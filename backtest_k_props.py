"""
Rolling-origin backtest of the pitcher strikeout prop, ON THE DEPLOYED PATH.

    python backtest_k_props.py                # 2026 starts, month by month
    python backtest_k_props.py 2025-06-01     # origins from this month on

WHY THIS FILE EXISTS
--------------------
The K model doc reports +0.0635 / +0.0765 Brier skill on 1,101 held-out
starts. The script that produced that number was never committed, and on
2026-09-05 it turned out the number predict_slate actually compounded was
a different one -- the hitter model's `p_is_k_vs_starter`, fed a career
pitcher rate with no window. Live: -0.14 on 32 starts.

So this backtest runs the SAME functions predict_slate.build_pitcher_props
now calls -- WorkloadModel.fit, per_batter_k_probs, k_count_distribution --
against the real lineups each starter faced, with every input computed
as of the morning of the start. If predict_slate changes how it builds
the per-batter rate, this file has to change with it, and that coupling
is the point.

WHAT IS AND IS NOT KNOWN AT THE ORIGIN
--------------------------------------
  workload model      fit on starts strictly before the origin month
  pitcher K rate      12-month shrunk, as of the origin
  hitter K rate       his shrunk 600-PA rate on the morning of the game
                      (the PA table's shifted column), where the hitter
                      is in the pool; the pool's 12-month league rate
                      where he is not
  lineup              the first nine distinct batters he actually faced,
                      in order -- i.e. a confirmed lineup, the best case
  platoon factors     measured on the pool over the 12 months before
                      the origin

Two things are optimistic and stated: the lineup is the real one (on the
night it is usually confirmed by first pitch, so this is close), and the
hitter pool is today's pool, so a hitter who joined it in August has a
rate in April. Neither touches the pitcher side, which is where the live
failure was.

OUTPUT
------
Per origin month and pooled: predicted vs actual batters faced, expected
vs actual strikeouts, and for each line the mean prediction, the hit
rate, and Brier skill against the base rate. Also the same numbers with
the hitter adjustment switched OFF (one flat rate per pitcher), so the
lineup walk has to show it earns its place.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data.cache import cache_path
from data.refresh import load_player_cache
from data.pitcher_data import load_pitcher_cache, pitcher_pa_table
from features.pa_table import build_pa_table
from features.rate_features import add_batter_rolling_rates
from features.pitcher_workload import (
    identify_starts, WorkloadModel, k_count_distribution, prob_over,
    per_batter_k_probs, TRAILING_MONTHS,
)

LINES = (4.5, 5.5, 6.5, 7.5)
CACHE_DIR = os.path.dirname(cache_path("x"))


def _ids(prefix):
    out = []
    for path in glob.glob(os.path.join(CACHE_DIR, f"{prefix}*.csv")):
        stem = os.path.basename(path)[len(prefix):-4]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(set(out))


# Intermediate results can be checkpointed to a scratch directory so the
# run survives being split across several short shell sessions. Set
# K_BACKTEST_SCRATCH to a directory; stages already on disk are reused.
SCRATCH = os.environ.get("K_BACKTEST_SCRATCH")
HITTER_COLS = ["batter", "pitcher", "game_pk", "game_date", "game_type",
               "events", "stand", "p_throws", "inning", "at_bat_number",
               "inning_topbot", "home_team", "away_team", "bat_score",
               "post_bat_score", "on_1b", "on_2b", "on_3b", "outs_when_up"]


def _scratch(name):
    return os.path.join(SCRATCH, name) if SCRATCH else None


def load_hitter_raw(chunk=None, n_chunks=1, verbose=True):
    """Raw Statcast rows for the pool, optionally one chunk of the files."""
    ids = _ids("statcast_player_")
    if chunk is not None:
        ids = ids[chunk::n_chunks]
    frames = []
    for i, pid in enumerate(ids, 1):
        raw = load_player_cache(pid)
        if raw.empty:
            continue
        frames.append(raw[[c for c in HITTER_COLS if c in raw.columns]])
        if verbose and i % 100 == 0:
            print(f"  hitters: {i}/{len(ids)} loaded")
    return pd.concat(frames, ignore_index=True)


def load_hitter_rates(verbose=True):
    """(batter, game_pk) -> shrunk 600-PA strikeout rate on the morning of
    that game, plus the full PA table for league rates."""
    path = _scratch("hitter_rates.pkl")
    if path and os.path.exists(path):
        return pd.read_pickle(path)
    raw_path = _scratch("hitters_raw.pkl")
    if raw_path and os.path.exists(raw_path):
        combined = pd.read_pickle(raw_path)
    else:
        combined = load_hitter_raw(verbose=verbose)
    pa = build_pa_table(combined)
    pa = add_batter_rolling_rates(pa, targets=("is_k",))
    # The first PA of the game carries the window ending before the game.
    first = (pa.sort_values(["batter", "game_date", "game_pk", "at_bat_number"])
               .groupby(["batter", "game_pk"]).first().reset_index())
    rates = first.set_index(["batter", "game_pk"])["bat_is_k_600"].to_dict()
    out = (rates, pa[["batter", "game_date", "is_k", "platoon_edge", "stand"]])
    if path:
        pd.to_pickle(out, path)
    return out


def load_starts_with_lineups(verbose=True):
    path = _scratch("starts.pkl")
    if path and os.path.exists(path):
        return pd.read_pickle(path)
    ids = _ids("statcast_pitcher_")
    frames = []
    for pid in ids:
        raw = load_pitcher_cache(pid)
        if raw.empty:
            continue
        t = pitcher_pa_table(raw)
        if not t.empty:
            frames.append(t)
    pa = pd.concat(frames, ignore_index=True)
    starts = identify_starts(pa)
    starts = starts[~starts["is_opener"]].copy()
    if verbose:
        print(f"  {len(starts):,} starts from {starts['pitcher'].nunique()} pitchers")

    # Ordered batters faced per start, with handedness matchup.
    pa = pa.sort_values(["pitcher", "game_pk", "at_bat_number"])
    p_throws = pa.groupby("pitcher")["p_throws"].agg(lambda s: s.mode().iloc[0]
                                                     if len(s.mode()) else "R")
    lineups = {}
    for (pid, gpk), g in pa.groupby(["pitcher", "game_pk"], sort=False):
        seen, order = set(), []
        for b, st in zip(g["batter"], g["stand"]):
            if b not in seen:
                seen.add(b)
                order.append((int(b), st))
            if len(order) == 9:
                break
        lineups[(pid, gpk)] = order
    out = (starts, lineups, p_throws.to_dict())
    if path:
        pd.to_pickle(out, path)
    return out


def platoon_factors(pool_pa, as_of):
    cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=12)
    recent = pool_pa[(pool_pa["game_date"] >= cutoff) & (pool_pa["game_date"] < as_of)]
    if len(recent) < 5000:
        recent = pool_pa[pool_pa["game_date"] < as_of]
    overall = float(recent["is_k"].mean())
    same = recent[recent["platoon_edge"] == 0]["is_k"].mean()
    opp = recent[recent["platoon_edge"] == 1]["is_k"].mean()
    o = lambda p: p / (1 - p)
    return overall, {"same": float(o(same) / o(overall)),
                     "opposite": float(o(opp) / o(overall))}


def run(first_origin="2026-04-01"):
    print("Loading hitter pool...")
    hitter_rates, pool_pa = load_hitter_rates()
    print("Loading starters...")
    starts, lineups, throws = load_starts_with_lineups()

    starts["month"] = starts["game_date"].dt.to_period("M")
    origins = sorted(m for m in starts["month"].unique()
                     if m.start_time >= pd.Timestamp(first_origin))
    rows = []
    for month in origins:
        as_of = month.start_time
        history = starts[starts["game_date"] < as_of]
        test = starts[starts["month"] == month]
        if history.empty or test.empty:
            continue
        try:
            wm = WorkloadModel.fit(history, as_of=as_of, verbose=False)
        except ValueError:
            continue
        batter_league, plat = platoon_factors(pool_pa, as_of)
        for s in test.itertuples():
            lineup = lineups.get((s.pitcher, s.game_pk))
            if not lineup:
                continue
            hand = throws.get(s.pitcher, "R")
            b_rates, factors = [], []
            for b, st in lineup:
                r = hitter_rates.get((b, s.game_pk))
                b_rates.append(r if r is not None and np.isfinite(r) else batter_league)
                factors.append(plat["opposite"] if st != hand else plat["same"])
            support, bf_probs = wm.bf_pmf(s.pitcher)
            row = {"month": str(month), "pitcher": s.pitcher, "game_pk": s.game_pk,
                   "starts_seen": wm.starts_seen(s.pitcher),
                   "pred_bf": wm.expected_bf(s.pitcher), "actual_bf": s.bf,
                   "actual_k": s.k}
            for label, probs in (
                    ("walk", per_batter_k_probs(b_rates, batter_league,
                                                wm.k_rate(s.pitcher), factors)),
                    ("flat", np.full(len(lineup), wm.k_rate(s.pitcher)))):
                dist = k_count_distribution(probs, support, bf_probs)
                row[f"exp_k_{label}"] = float((np.arange(len(dist)) * dist).sum())
                for line in LINES:
                    row[f"p_{label}_{line}"] = prob_over(dist, line)
            rows.append(row)
        print(f"  {month}: {len(test)} starts, origin fit on {len(history):,}")

    res = pd.DataFrame(rows)
    if res.empty:
        print("Nothing to score.")
        return
    res.to_csv(cache_path("_k_backtest"), index=False)

    def report(frame, title):
        print(f"\n{title}  ({len(frame)} starts)")
        print(f"  batters faced: predicted {frame['pred_bf'].mean():.2f}, "
              f"actual {frame['actual_bf'].mean():.2f}")
        for label in ("walk", "flat"):
            print(f"  [{label}] strikeouts: expected {frame[f'exp_k_{label}'].mean():.3f}, "
                  f"actual {frame['actual_k'].mean():.3f}")
            for line in LINES:
                p = frame[f"p_{label}_{line}"]
                y = (frame["actual_k"] > line).astype(float)
                base = y.mean()
                brier = float(((p - y) ** 2).mean())
                ref = base * (1 - base)
                skill = 1 - brier / ref if ref > 0 else float("nan")
                print(f"     over {line}: said {p.mean():.3f}  did {base:.3f}  "
                      f"Brier skill {skill:+.4f}")

    for month, g in res.groupby("month"):
        report(g, f"origin {month}")
    report(res, "POOLED")
    report(res[res["starts_seen"] >= 5], "POOLED, pitchers with >= 5 starts in window")
    report(res[res["starts_seen"] < 5], "POOLED, pitchers with < 5 starts in window")
    print(f"\n  Rows: cache/_k_backtest.csv")


if __name__ == "__main__":
    # Checkpoint stages, for running in short sessions:
    #   python backtest_k_props.py --hitters I N   load chunk I of N hitter
    #                                              files into scratch
    #   python backtest_k_props.py --rates         build hitter rates
    #   python backtest_k_props.py --starts        build starts + lineups
    if len(sys.argv) > 1 and sys.argv[1] == "--hitters":
        i, n = int(sys.argv[2]), int(sys.argv[3])
        part = load_hitter_raw(chunk=i, n_chunks=n)
        pd.to_pickle(part, _scratch(f"hitters_raw_{i}.pkl"))
        parts = [_scratch(f"hitters_raw_{j}.pkl") for j in range(n)]
        if all(os.path.exists(p) for p in parts):
            pd.to_pickle(pd.concat([pd.read_pickle(p) for p in parts],
                                   ignore_index=True), _scratch("hitters_raw.pkl"))
            print("  all chunks present; hitters_raw.pkl written")
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--rates":
        load_hitter_rates()
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--starts":
        load_starts_with_lineups()
        raise SystemExit(0)
    run(sys.argv[1] if len(sys.argv) > 1 else "2026-04-01")
