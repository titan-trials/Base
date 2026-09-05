"""
Does any of the model_flags earn its place? Rolling-origin comparison.

    python flag_lab.py                      # every flag, one at a time, then all
    python flag_lab.py --players 150        # bigger pool (slower)
    python flag_lab.py --origins 3          # three cut points, not one
    python flag_lab.py --flags LOGIT_FEATURES,PLATOON_SPLIT

COMPLETELY INDEPENDENT OF THE LIVE MODEL
----------------------------------------
Every flag in model_flags.py defaults to what the scoring log was produced
with. This lab flips them in-process, rebuilds the features, refits the
per-PA models on the earlier part of the history and scores the later
part -- and then puts them back. Adopting a winner means editing
model_flags.py, which nothing here does for you.

WHAT IS COMPARED
----------------
For each of the four per-PA targets: Brier skill, AUC and expected
calibration error on the held-out period, plus the PAIRED bootstrap
difference in skill against the baseline (model_lab.paired_bootstrap_skill
-- both models score the same rows, so the shared noise cancels). The
same floor model_lab uses applies: a gain under 0.0005 skill is
"negligible" no matter how significant.

PER_HITTER_SHAPE is not a per-PA change, so it is scored at game level:
P(total bases over 0.5 / 1.5 / 2.5 / 3.5) for every held-out player-game,
with the ACTUAL plate-appearance count held fixed so that only the shape
is on trial, population shape versus per-hitter shape.

    K_BACKTEST_SCRATCH=<dir>   reuses the hitters_raw.pkl checkpoint that
                               backtest_k_props.py writes, which saves the
                               minute of CSV loading.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

import model_flags
from data.park_factors import get_park_factor
from features.pa_table import build_pa_table, build_game_totals, per_pa_contribution_distribution
from features.rate_features import (
    add_batter_rolling_rates, add_pitcher_rolling_rates, add_matchup_rates,
    rate_feature_cols, RATE_TARGETS,
)
from features.bases_features import (
    measure_bases_per_hit, measure_hit_mix, per_hitter_tb_distribution,
    expected_total_bases_per_pa,
)
from model.compound import (
    scale_contribution_distribution, compound_count_distribution, prob_over_line,
)
from model_lab import (
    load_pool, expected_calibration_error, paired_bootstrap_skill,
)
from train_props_v5 import train_rate_model, TB_LINES

DEFAULTS = {
    "LEAGUE_PRIOR_TRAILING_DAYS": None,
    "PITCHER_RATE_WINDOW_DAYS": None,
    "LOGIT_FEATURES": False,
    "PLATOON_SPLIT": False,
    "PER_HITTER_SHAPE": False,
}
TRIALS = {
    "LEAGUE_PRIOR_TRAILING_DAYS": 365,
    "PITCHER_RATE_WINDOW_DAYS": 365,
    "LOGIT_FEATURES": True,
    "PLATOON_SPLIT": True,
}
NEGLIGIBLE = 0.0005


def set_flags(**values):
    for k, v in DEFAULTS.items():
        setattr(model_flags, k, values.get(k, v))


def build_features(raw_pa: pd.DataFrame) -> pd.DataFrame:
    pa = add_batter_rolling_rates(raw_pa)
    pa = add_pitcher_rolling_rates(pa)
    pa = add_matchup_rates(pa)
    pa["park_hr_factor"] = pa["home_team"].apply(get_park_factor) / 100.0
    return pa


def score_config(pa: pd.DataFrame, cut_dates, targets) -> dict:
    """{(target, cut): (y, p)} for the held-out rows after each cut."""
    groups = rate_feature_cols()
    out = {}
    for cut in cut_dates:
        for target in targets:
            cols = groups[target]
            frame = pa.dropna(subset=cols + [target])
            train = frame[frame["game_date"] <= cut]
            test = frame[frame["game_date"] > cut]
            if len(test) < 3000 or test[target].nunique() < 2:
                continue
            model = train_rate_model(train, cols, target)
            out[(target, cut)] = (test[target].to_numpy(dtype=float),
                                  model.predict_proba(test[cols])[:, 1],
                                  test.index)
    return out


def skill(y, p):
    base = y.mean()
    return 1 - ((p - y) ** 2).mean() / (base * (1 - base))


def report_per_pa(baseline: dict, trial: dict, name: str):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  {'target':8s} {'cut':10s} {'base skill':>11} {'new skill':>10} "
          f"{'dAUC':>8} {'dECE':>8} {'gain':>9} {'95% CI':>20}  verdict")
    verdicts = []
    for key in sorted(baseline):
        if key not in trial:
            continue
        y, p0, idx0 = baseline[key]
        y1, p1, idx1 = trial[key]
        # Align on shared rows -- a flag can change which rows have
        # complete features (e.g. the pitcher window).
        common = idx0.intersection(idx1)
        if len(common) < 3000:
            continue
        m0 = pd.Series(p0, index=idx0).loc[common].to_numpy()
        m1 = pd.Series(p1, index=idx1).loc[common].to_numpy()
        yy = pd.Series(y, index=idx0).loc[common].to_numpy()
        boot = paired_bootstrap_skill(yy, m0, m1, n_boot=1000)
        gain = boot["point"]
        if boot["lo"] > 0 and gain >= NEGLIGIBLE:
            verdict = "BETTER"
        elif boot["hi"] < 0 and gain <= -NEGLIGIBLE:
            verdict = "WORSE"
        elif abs(gain) < NEGLIGIBLE:
            verdict = "negligible"
        else:
            verdict = "tie"
        verdicts.append(verdict)
        print(f"  {key[0]:8s} {str(key[1].date()):10s} {skill(yy, m0):+11.4f} "
              f"{skill(yy, m1):+10.4f} "
              f"{roc_auc_score(yy, m1) - roc_auc_score(yy, m0):+8.4f} "
              f"{expected_calibration_error(yy, m1) - expected_calibration_error(yy, m0):+8.4f} "
              f"{gain:+9.4f} [{boot['lo']:+.4f}, {boot['hi']:+.4f}]  {verdict}")
    return verdicts


def shape_trial(pa: pd.DataFrame, baseline: dict, cut_dates):
    """PER_HITTER_SHAPE at game level, actual PA held fixed."""
    print(f"\n{'=' * 78}\nPER_HITTER_SHAPE -- total bases, game level, actual PA fixed\n{'=' * 78}")
    tb_dist = per_pa_contribution_distribution(pa, "total_bases")
    league_bph, bases_per_hit, prior = measure_bases_per_hit(pa, verbose=False)
    league_mix, mix = measure_hit_mix(pa, prior, verbose=False)
    totals = build_game_totals(pa)
    for cut in cut_dates:
        need = [("is_hr", cut), ("is_hit", cut)]
        if any(k not in baseline for k in need):
            continue
        # Per-PA probabilities from the baseline models, first PA of each game.
        probs = pd.DataFrame({
            "p_hr": pd.Series(baseline[("is_hr", cut)][1], index=baseline[("is_hr", cut)][2]),
            "p_hit": pd.Series(baseline[("is_hit", cut)][1], index=baseline[("is_hit", cut)][2]),
        }).dropna()
        first = pa.loc[probs.index, ["batter", "game_pk"]].assign(
            p_hr=probs["p_hr"].to_numpy(), p_hit=probs["p_hit"].to_numpy())
        first = first.groupby(["batter", "game_pk"], as_index=False).first()
        games = totals.merge(first, on=["batter", "game_pk"])
        games = games[games["game_date"] > cut]
        if len(games) < 1000:
            continue
        bph = games["batter"].map(bases_per_hit).fillna(league_bph).to_numpy()
        means = expected_total_bases_per_pa(games["p_hr"].to_numpy(),
                                            games["p_hit"].to_numpy(), bph)
        pop_p = {line: [] for line in TB_LINES}
        own_p = {line: [] for line in TB_LINES}
        for pid, n, mean, p_hr, p_hit in zip(games["batter"], games["pa"], means,
                                            games["p_hr"], games["p_hit"]):
            n = int(np.clip(n, 1, 8))
            pop = compound_count_distribution(
                scale_contribution_distribution(tb_dist, float(mean)), n)
            own_dist = per_hitter_tb_distribution(
                p_hr, p_hit, mix.loc[pid] if pid in mix.index else league_mix)
            own = compound_count_distribution(own_dist, n)
            for line in TB_LINES:
                pop_p[line].append(prob_over_line(pop, line))
                own_p[line].append(prob_over_line(own, line))
        print(f"  cut {cut.date()}, {len(games):,} player-games")
        print(f"  {'line':>5} {'did':>6} {'pop said':>9} {'own said':>9} "
              f"{'pop skill':>10} {'own skill':>10} {'gain':>8} {'95% CI':>20}")
        for line in TB_LINES:
            y = (games["total_bases"].to_numpy() > line).astype(float)
            a, b = np.array(pop_p[line]), np.array(own_p[line])
            boot = paired_bootstrap_skill(y, a, b, n_boot=1000)
            print(f"  {line:>5} {y.mean():6.3f} {a.mean():9.3f} {b.mean():9.3f} "
                  f"{skill(y, a):+10.4f} {skill(y, b):+10.4f} {boot['point']:+8.4f} "
                  f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}]")


def load_raw(players: int) -> pd.DataFrame:
    scratch = os.environ.get("K_BACKTEST_SCRATCH")
    path = os.path.join(scratch, "hitters_raw.pkl") if scratch else None
    if path and os.path.exists(path):
        print(f"Loading hitters from {path}")
        return pd.read_pickle(path)
    return load_pool(players)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=120)
    parser.add_argument("--origins", type=int, default=1)
    parser.add_argument("--flags", type=str, default=None)
    parser.add_argument("--targets", type=str, default=",".join(RATE_TARGETS))
    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",")]
    flags = ([f.strip() for f in args.flags.split(",")] if args.flags
             else list(TRIALS) + ["PER_HITTER_SHAPE"])

    raw = load_raw(args.players)
    base_pa = build_pa_table(raw)
    qs = np.linspace(0.6, 0.8, args.origins) if args.origins > 1 else [0.75]
    cut_dates = [base_pa["game_date"].quantile(q) for q in qs]
    print(f"  {len(base_pa):,} plate appearances, {base_pa['batter'].nunique()} batters; "
          f"origins at {', '.join(str(c.date()) for c in cut_dates)}")

    set_flags()
    pa0 = build_features(base_pa.copy())
    baseline = score_config(pa0, cut_dates, targets)
    print("\nBASELINE (all flags off), held-out skill:")
    for key, (y, p, _) in sorted(baseline.items()):
        print(f"  {key[0]:8s} {key[1].date()}  skill {skill(y, p):+.4f}  "
              f"AUC {roc_auc_score(y, p):.4f}  ECE {expected_calibration_error(y, p):.4f}")

    summary = {}
    for flag in flags:
        if flag == "PER_HITTER_SHAPE":
            set_flags()
            shape_trial(pa0, baseline, cut_dates)
            continue
        if flag not in TRIALS:
            print(f"  Unknown flag {flag}; skipping.")
            continue
        set_flags(**{flag: TRIALS[flag]})
        trial = score_config(build_features(base_pa.copy()), cut_dates, targets)
        summary[flag] = report_per_pa(baseline, trial, f"{flag} = {TRIALS[flag]}")
    if len(flags) > 1 and all(f in TRIALS for f in flags if f != "PER_HITTER_SHAPE"):
        set_flags(**{f: TRIALS[f] for f in flags if f in TRIALS})
        trial = score_config(build_features(base_pa.copy()), cut_dates, targets)
        summary["ALL"] = report_per_pa(baseline, trial, "ALL per-PA flags together")
    set_flags()

    print("\n" + "=" * 78)
    print("SUMMARY (per-PA flags; verdicts across targets and origins)")
    print("=" * 78)
    for flag, verdicts in summary.items():
        counts = pd.Series(verdicts).value_counts().to_dict()
        print(f"  {flag:28s} {counts}")
    print("\n  A flag is worth turning on when it is BETTER on the targets you\n"
          "  care about and WORSE on none. Edit model_flags.py to adopt it;\n"
          "  the scoring log's running total describes the model as configured.")


if __name__ == "__main__":
    main()
