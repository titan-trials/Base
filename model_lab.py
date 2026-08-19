"""
Is logistic regression the right model, or just the first one we tried?

    python model_lab.py                    # all models, all targets
    python model_lab.py --players 120      # bigger pool
    python model_lab.py --targets is_k     # one target

COMPLETELY INDEPENDENT OF THE LIVE MODEL
----------------------------------------
Imports the pipeline, changes nothing in it. Delete this file to remove
it entirely -- no git archaeology. Adopting a winner means editing
train_rate_model() in train_props_v5.py, which is a deliberate act nothing
here performs for you.

WHAT THIS FIXES FROM feature_lab.py
-----------------------------------
That lab reported `baseline`/`with` in Brier SKILL and `gain` in raw Brier
SCORE, side by side, as though they were comparable. They differ by a
factor of the reference variance -- about 0.034 for home runs -- so a real
skill change of 0.00013 printed as "0.00000" and the table looked
self-contradictory. Everything here is in SKILL units, so every column can
be read against every other.

It also ran 28 comparisons at 95% confidence and flagged the one that came
back marginal, when ~1.4 false positives were expected by chance. This
reports how many comparisons were made and what that implies.

WHY CALIBRATION IS REPORTED SEPARATELY FROM ACCURACY
----------------------------------------------------
The received wisdom is that gradient boosting RANKS better than logistic
regression and CALIBRATES worse. On this data it came out the other way
round -- on is_k, gbm_shallow ranked slightly WORSE (AUC 0.5957 vs 0.5968)
and calibrated much BETTER (ECE 0.0038 vs 0.0079). Which is exactly why
both are measured instead of assumed.

For this project that trade is bad. The V4 reframe concluded that the
printed percentage has to be honest, because a 12% home run chance that
happens 12% of the time is the product; a model that ranks hitters
beautifully but prints 25% for a 12% event is worse than useless. So each
model reports:

    BRIER SKILL   accuracy of the probability (the headline)
    AUC           ranking only, ignores whether the number is honest
    ECE           expected calibration error -- average gap between what
                  was printed and what happened, by decile

A model that wins on AUC and loses on ECE has not won.

Every boosted model is therefore also run WITH post-hoc calibration, which
is the fair comparison -- otherwise the contest is "a calibrated model
versus an uncalibrated one" rather than "which model class fits."
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from data.cache import cache_path
from data.park_factors import get_park_factor
from features.pa_table import build_pa_table
from features.rate_features import (
    add_batter_rolling_rates, add_pitcher_rolling_rates, add_matchup_rates,
    rate_feature_cols, RATE_TARGETS,
)
from model.hr_v4 import evaluate

TRAIN_FRAC = 0.75
N_BOOT = 2000
BASELINE = "lr_C0.1"        # what predict_slate.py runs today

# A gain has to be BIG ENOUGH TO CARE ABOUT, not merely resolvable.
#
# The first run of this file reported two models as BETTER: lr_C0.03 beat
# the baseline by +0.00001 on is_hit and +0.0000004 on is_k, the latter
# with a confidence interval of [0.00000, 0.00000]. Both were real -- C of
# 0.03 and 0.1 are nearly the same model, so the difference is consistent
# across 48,000 rows and never changes sign -- and both were worthless.
# +0.0000004 on a baseline of 0.0187 is a 0.02% relative change.
#
# Statistical significance measures whether an effect is DETECTABLE.
# Practical significance measures whether it MATTERS. With enough rows the
# first is satisfied by anything at all, so a floor is required.
#
# 0.0005 is roughly 15% of the home run baseline (0.0032) and 3% of the
# strikeout baseline (0.0187) -- small enough not to dismiss a genuine
# improvement, large enough that adopting it would be defensible.
MEANINGFUL_GAIN = 0.0005


def make_models():
    """
    Each entry builds a fresh estimator.

    The three logistic variants are here because V4 measured that
    REGULARISATION STRENGTH mattered more than model class -- moving C
    from 1.0 to 0.03 was worth about 0.007 Brier skill.

    MEASURED HERE, THAT NO LONGER HOLDS: C of 0.03, 0.1 and 1.0 all
    produce a Brier skill of 0.00318 on home runs. Identical to five
    decimal places.

    The explanation is that V4's finding was about V4's model, which had
    28 loosely-related features and therefore 28 coefficients for
    regularisation to shrink. V5 has 8 well-chosen ones, and there is
    almost nothing left to over-fit. A lesson learned on one feature set
    does not automatically transfer to another -- which is the reason to
    keep re-running the check rather than citing the old number.
    """
    return {
        "lr_C0.03": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.03))]),
        "lr_C0.1": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.1))]),
        "lr_C1.0": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0))]),
        # Shallow and heavily regularised on purpose. The signal here is
        # weak and smooth; a deep tree ensemble will memorise noise --
        # measured: gbm_deep costs -0.00077 on home runs and -0.00127 on
        # walks, against a baseline skill of 0.00318 and 0.00707.
        "gbm_shallow": lambda: HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=200, l2_regularization=1.0, random_state=0),
        "gbm_deep": lambda: HistGradientBoostingClassifier(
            max_depth=6, max_iter=400, learning_rate=0.05,
            min_samples_leaf=50, random_state=0),
        # The "fair" version -- and the WORST performer on home runs
        # (-0.00106, a third of the entire edge). Isotonic calibration on
        # a 3.5% event over-fits its own calibration curve: each cv fold
        # fits a step function to a few hundred positives.
        #
        # V4 hit this independently, where isotonic cost -0.045 Brier
        # skill and emitted p=1.000 for 16 test games. Twice measured, at
        # different levels of the model, on different data. Treat isotonic
        # calibration of rare events as a known hazard in this project.
        "gbm_calibrated": lambda: CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=200, l2_regularization=1.0, random_state=0),
            method="isotonic", cv=3),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=200,
            n_jobs=-1, random_state=0),
    }


def _verdict(boot) -> str:
    """
    Significant AND large enough to act on.

    "negligible" is deliberately distinct from "tie": it means the
    difference is real and measurable but too small to justify changing a
    model that already works. Collapsing the two would hide the fact that
    the test succeeded -- it found something, and the something was
    nothing much.
    """
    if boot["lo"] > 0:
        return "BETTER" if boot["point"] >= MEANINGFUL_GAIN else "negligible"
    if boot["hi"] < 0:
        return "WORSE" if -boot["point"] >= MEANINGFUL_GAIN else "negligible"
    return "tie"


def expected_calibration_error(y, p, n_bins=10) -> float:
    """
    Average gap between the printed probability and what actually
    happened, over equal-count bins.

    Equal-COUNT rather than equal-WIDTH bins: home run probabilities all
    live between 0.02 and 0.35, so equal-width bins would leave most of
    them empty and the average would be dominated by bins holding three
    rows each.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    gaps, weights = [], []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        gaps.append(abs(p[chunk].mean() - y[chunk].mean()))
        weights.append(len(chunk))
    return float(np.average(gaps, weights=weights))


def paired_bootstrap_skill(y, p_base, p_new, n_boot=N_BOOT, seed=0):
    """
    Difference in Brier SKILL, with a paired bootstrap.

    Both models score the same rows, so resampling rows cancels the shared
    noise and measures the difference directly. Reported in skill units by
    dividing through by the reference variance p(1-p) -- the same
    denominator the baseline column uses, so the two are comparable. That
    is precisely what feature_lab.py got wrong.
    """
    y = np.asarray(y, dtype=float)
    base_rate = y.mean()
    reference = base_rate * (1 - base_rate)
    if reference <= 0:
        return None

    diff = ((p_base - y) ** 2 - (p_new - y) ** 2) / reference
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    draws = diff[idx].mean(axis=1)
    return {"point": float(diff.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "win_rate": float((draws > 0).mean())}


def load_pool(max_players: int) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(os.path.dirname(cache_path("x")), "statcast_player_*.csv")))
    if not paths:
        raise SystemExit("No cached players. Run predict_slate.py first.")
    paths = paths[:max_players]
    print(f"Loading {len(paths)} cached hitters...")
    frames = []
    for i, path in enumerate(paths, 1):
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except Exception:
            continue
        if i % 40 == 0:
            print(f"  {i}/{len(paths)}")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=80)
    parser.add_argument("--targets", type=str, default=",".join(RATE_TARGETS))
    parser.add_argument("--models", type=str, default=None)
    args = parser.parse_args()

    models = make_models()
    if args.models:
        keep = {m.strip() for m in args.models.split(",")} | {BASELINE}
        models = {k: v for k, v in models.items() if k in keep}
    targets = [t.strip() for t in args.targets.split(",")]

    print("=" * 78)
    print("MODEL LAB -- can anything beat the logistic regression in production?")
    print("=" * 78)

    raw = load_pool(args.players)
    pa = build_pa_table(raw)
    pa = add_batter_rolling_rates(pa)
    pa = add_pitcher_rolling_rates(pa)
    pa = add_matchup_rates(pa)
    pa["park_hr_factor"] = pa["home_team"].apply(get_park_factor) / 100.0
    groups = rate_feature_cols()

    split_date = pa["game_date"].quantile(TRAIN_FRAC)
    print(f"\n  {len(pa):,} plate appearances, {pa['batter'].nunique()} batters")
    print(f"  Time split at {split_date.date()} -- train earlier, test later.")
    print(f"  Baseline is '{BASELINE}', which is what predict_slate.py runs.\n")

    rows = []
    for target in targets:
        cols = groups[target]
        frame = pa.dropna(subset=cols + [target])
        train = frame[frame["game_date"] <= split_date]
        test = frame[frame["game_date"] > split_date]
        y = test[target].to_numpy()
        if len(test) < 3000 or len(np.unique(y)) < 2:
            print(f"  {target}: not enough test data.")
            continue

        print("=" * 78)
        print(f"TARGET: {target}   (base rate {y.mean():.4f}, "
              f"{len(train):,} train / {len(test):,} test)")
        print("=" * 78)
        print(f"  {'model':16s} {'BrierSkill':>11} {'AUC':>7} {'ECE':>8} "
              f"{'vs baseline':>13} {'95% CI':>22}")

        predictions = {}
        for name, build in models.items():
            try:
                model = build()
                model.fit(train[cols], train[target])
                predictions[name] = model.predict_proba(test[cols])[:, 1]
            except Exception as e:
                print(f"  {name:16s} FAILED ({type(e).__name__}: {e})")

        if BASELINE not in predictions:
            print(f"  baseline '{BASELINE}' failed -- cannot compare.")
            continue

        for name, p in predictions.items():
            result = evaluate(y, p, label="")
            ece = expected_calibration_error(y, p)
            if name == BASELINE:
                comparison, interval = "  (baseline)", ""
            else:
                boot = paired_bootstrap_skill(y, predictions[BASELINE], p)
                comparison = f"{boot['point']:+.5f}"
                interval = f"[{boot['lo']:+.5f}, {boot['hi']:+.5f}]"
                rows.append({"target": target, "model": name,
                             "brier_skill": result["brier_skill"],
                             "auc": result["auc"], "ece": ece,
                             "gain_vs_baseline": boot["point"],
                             "lo": boot["lo"], "hi": boot["hi"],
                             "verdict": _verdict(boot)})
            print(f"  {name:16s} {result['brier_skill']:>+11.5f} "
                  f"{result['auc']:>7.4f} {ece:>8.5f} {comparison:>13} "
                  f"{interval:>22}")

        base_ece = expected_calibration_error(y, predictions[BASELINE])
        worse_cal = [n for n, p in predictions.items()
                     if n != BASELINE and expected_calibration_error(y, p) > base_ece * 1.5]
        if worse_cal:
            print(f"\n  Calibration warning: {', '.join(worse_cal)} print "
                  f"percentages at least 50% further from reality than the")
            print(f"  baseline does. Ranking is not the product here -- the "
                  f"printed number is.")
        print()

    if not rows:
        raise SystemExit("Nothing evaluated.")

    table = pd.DataFrame(rows)
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    n = len(table)
    better = table[table.verdict == "BETTER"]
    negligible = table[table.verdict == "negligible"]
    if len(negligible):
        print(f"\n  {len(negligible)} comparison(s) were statistically real but "
              f"smaller than {MEANINGFUL_GAIN:.4f} skill, so they are reported")
        print(f"  as 'negligible' rather than as wins. With ~48,000 test rows "
              f"a difference of 0.00001 is detectable and meaningless.")
    expected_false = 0.05 * n
    print(f"\n  {len(better)} of {n} comparisons beat the baseline with the "
          f"interval excluding zero.")
    print(f"  At 95% confidence across {n} comparisons, roughly "
          f"{expected_false:.1f} false positives are expected by chance.")
    if len(better) <= expected_false:
        print("  That is at or below the chance rate -- treat any single "
              "winner here as unproven.")
    else:
        print("  That is above the chance rate, so at least some of these "
              "are real. Check ECE before adopting:")
        for _, r in better.iterrows():
            print(f"    {r.target:9s} {r.model:16s} "
                  f"skill {r.gain_vs_baseline:+.5f}, ECE {r.ece:.5f}")

    out = cache_path("model_lab_results")
    table.to_csv(out, index=False)
    print(f"\n  Saved to {out}")
    print("  Nothing in the live model changed. Delete model_lab.py to "
          "remove this entirely.")


if __name__ == "__main__":
    main()
