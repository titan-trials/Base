"""
V4: a CALIBRATED home-run probability model.

    python train_hr_v4.py

WHAT CHANGED FROM V1-V3
-----------------------
Earlier versions chased a classifier that could call a home run correctly,
and topped out at AUC 0.565. That ceiling is real: a single-game home run
is a rare event dominated by an error term (exact pitch location, timing
to the millisecond, where the ball is caught) that no observable feature
set explains. Pushing for "60%+ confident" was chasing a number the
problem does not contain.

V4 keeps the same target but changes what counts as success. Instead of
"was the call right," the question is "is the PROBABILITY honest" -- when
the model prints 23.6%, does that happen about 23.6% of the time. That is
a question this data can actually answer, and a well-calibrated 24% is a
more useful output than a falsely confident yes.

Two new feature families are added on top of the V3 pitch-matchup work:

  CONTACT QUALITY (features/contact_quality.py)
    Sweet-spot rate, barrel rate, hard-hit rate, exit velocity on air
    balls, pull-side air rate. These measure the SKILL under the home run
    rather than the home run itself -- roughly 10x the sample size per
    window, because a hitter puts ~30 balls in play per 10 games but hits
    ~2 home runs. This is the same principle that made walk rate the
    project's best player-level target.

  CONTEXT (features/context_features.py + data/game_context.py)
    Park splits, home/away, day-of-week, day/night, plus real game-time
    temperature and wind from MLB's own Stats API. Every split is
    empirical-Bayes shrunk toward the player's own baseline, so a 5-game
    sample can't claim a 40% home run rate. The day-of-week group is then
    run through a permutation test, so "he homers on Sundays" gets a
    p-value instead of a shrug.

WHAT THE OUTPUT MEANS
---------------------
The headline is the BRIER SKILL SCORE, not AUC. It compares the model
against the honest null of always predicting the base rate. Positive means
the features add information. Around zero means the model is decoration on
the base rate -- which, given the screenshot showing 23.6% against a 23.6%
base rate in the training pool, is exactly the failure mode worth watching
for here.

RUNTIME
-------
First run is slow: pitch mixes per opposing starter (cached) plus one
Stats API call per unique game for weather (cached, checkpointed every 100
games, resumable). Later runs read entirely from cache/.

Set SKIP_WEATHER = True below to run everything except the weather pull --
useful for a fast first look before committing to the full fetch.
"""
import os
import pandas as pd

from config import PLAYERS, START_DATE, END_DATE
from data.loader import load_batter_statcast_cached
from data.game_context import attach_game_context
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.fatigue_features import add_fatigue_features
from features.contact_quality import (
    build_contact_quality_log,
    add_contact_quality_rolling,
    contact_quality_feature_cols,
    DEFAULT_WINDOWS,
)
from features.context_features import add_context_features, CALENDAR_SPLIT_COLS, WEATHER_COLS
from features.pitch_mix_features import (
    get_starting_pitcher_per_game,
    build_batter_bucket_log,
    add_bucket_rolling_rates,
    build_pitcher_mix_table,
    BUCKETS,
)
from model.hr_v4 import (
    chronological_split, fit_calibrated, predict_calibrated, evaluate,
    print_evaluation, compare_results, permutation_test,
    print_permutation_result, coefficient_report, print_reliability_by_quantile,
    bootstrap_brier_skill, print_bootstrap, rolling_origin_evaluation,
    print_rolling_origin, univariate_ranking, print_univariate_ranking,
)
from model.calibration import print_calibration_report

SKIP_WEATHER = False       # True = skip the Stats API pull (fast dry run)
RUN_PERMUTATION_TESTS = True
N_PERMUTATIONS = 200
BUCKET_WINDOW = 20

# Regularisation strength. 0.03 was chosen on the calibration slice of the
# real data; sklearn's default of 1.0 measurably overfits this signal.
# Set to None to re-select it on the calibration slice each run.
MODEL_C = 0.03

# --- Feature groups, added one at a time in the ablation ---------------
FORM_FEATURES = [
    "hr_rate_10", "hr_rate_20", "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20", "career_hr_rate", "pa_20",
]
FATIGUE_FEATURES = ["rest_days", "games_last_7_days"]
MATCHUP_FEATURES = ["expected_hr_exposure"]
CONTACT_FEATURES = contact_quality_feature_cols(DEFAULT_WINDOWS, primary_window=20)
CALENDAR_FEATURES = list(CALENDAR_SPLIT_COLS)
WEATHER_FEATURES = list(WEATHER_COLS)


def build_dataset() -> pd.DataFrame:
    """Everything from raw cached Statcast to one model-ready frame."""
    print("=" * 70)
    print("Loading player data (cached where available)...")
    all_statcast = []
    for first, last in PLAYERS:
        try:
            df = load_batter_statcast_cached(first, last, START_DATE, END_DATE)
            all_statcast.append(df)
        except Exception as e:
            print(f"  Skipping {first} {last}: {e}")
    if not all_statcast:
        raise RuntimeError("No player data loaded -- check config.PLAYERS and cache/.")
    combined = pd.concat(all_statcast, ignore_index=True)

    print("Building game log and rolling form features...")
    game_log = build_multi_game_log(combined)
    features_df = add_rolling_features_multi(game_log)
    features_df = add_fatigue_features(features_df)

    print("Building contact-quality features (sweet spot / barrel / hard hit)...")
    contact_log = build_contact_quality_log(combined)
    contact_rates = add_contact_quality_rolling(contact_log)
    keep = ["batter", "game_pk", "game_date"] + [
        c for c in contact_rates.columns
        if c not in contact_log.columns or c in ("batter", "game_pk", "game_date")
    ]
    features_df = features_df.merge(
        contact_rates[list(dict.fromkeys(keep))],
        on=["batter", "game_pk", "game_date"], how="left",
    )

    print("Building pitch-type matchup feature...")
    bucket_log = build_batter_bucket_log(combined)
    bucket_rates = add_bucket_rolling_rates(bucket_log)
    bucket_cols = [f"hr_rate_vs_{b}_{BUCKET_WINDOW}" for b in BUCKETS]
    features_df = features_df.merge(
        bucket_rates[["batter", "game_pk", "game_date"] + bucket_cols],
        on=["batter", "game_pk", "game_date"], how="left",
    )

    starters = get_starting_pitcher_per_game(combined)
    features_df = features_df.merge(starters, on="game_pk", how="left")
    features_df["mix_season"] = features_df["game_date"].dt.year - 1
    pairs = features_df[["opp_pitcher_id", "mix_season"]].dropna().drop_duplicates()
    pairs = list(pairs.itertuples(index=False, name=None))
    print(f"  Pitch mix for {len(pairs)} unique (pitcher, season) pairs...")
    mix_table = build_pitcher_mix_table(pairs)

    if not mix_table.empty:
        features_df = features_df.merge(
            mix_table, left_on=["opp_pitcher_id", "mix_season"],
            right_on=["opp_pitcher_id", "season"], how="left",
        )
        features_df["expected_hr_exposure"] = (
            features_df[f"hr_rate_vs_fastball_{BUCKET_WINDOW}"] * features_df["fastball_pct"]
            + features_df[f"hr_rate_vs_breaking_{BUCKET_WINDOW}"] * features_df["breaking_pct"]
            + features_df[f"hr_rate_vs_offspeed_{BUCKET_WINDOW}"] * features_df["offspeed_pct"]
        )
    else:
        print("  No pitch mix returned -- expected_hr_exposure set to 0.")
        features_df["expected_hr_exposure"] = 0.0

    if not SKIP_WEATHER:
        print("Attaching venue / day-night / weather from MLB Stats API...")
        features_df = attach_game_context(features_df)
    else:
        print("SKIP_WEATHER = True -- venue falls back to Statcast home_team, "
              "weather features will be neutral.")

    print("Building shrunk context splits (venue / weekday / home / day-night)...")
    features_df = add_context_features(features_df, combined)

    return features_df


def main():
    features_df = build_dataset()

    all_features = (
        FORM_FEATURES + FATIGUE_FEATURES + MATCHUP_FEATURES
        + CONTACT_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES
    )
    missing = [c for c in all_features if c not in features_df.columns]
    if missing:
        raise KeyError(
            f"Missing expected feature columns: {missing}. If this is a "
            f"KeyError after adding a feature, clear the relevant cache/ "
            f"file -- old cached pulls won't have the new column."
        )

    before = len(features_df)
    features_df = features_df.dropna(subset=all_features + ["hit_hr"]).reset_index(drop=True)
    print(f"\nRows with every feature available: {len(features_df)} "
          f"(dropped {before - len(features_df)} incomplete rows)")
    print(f"Overall HR-per-game rate in this pool: {features_df['hit_hr'].mean():.3f}")

    train_df, calib_df, test_df = chronological_split(features_df)
    print(f"Chronological split -> train {len(train_df)} "
          f"({train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}) | "
          f"calibrate {len(calib_df)} | test {len(test_df)} "
          f"({test_df['game_date'].min().date()} to {test_df['game_date'].max().date()})")

    # ---------------- Ablation ladder ---------------------------------
    print("\n" + "=" * 70)
    print("ABLATION -- each row adds one feature group to the row above it.")
    print("Watch BRIER SKILL, not AUC: it says whether the probability got")
    print("more honest, which is the whole point of V4.")
    print("=" * 70)

    ladder = [
        ("1. Rolling form only (V1-style)", FORM_FEATURES),
        ("2. + fatigue + pitch matchup (V3 best)", FORM_FEATURES + FATIGUE_FEATURES + MATCHUP_FEATURES),
        ("3. + contact quality (sweet spot / barrel)", FORM_FEATURES + FATIGUE_FEATURES + MATCHUP_FEATURES + CONTACT_FEATURES),
        ("4. + weather (temp / wind)", FORM_FEATURES + FATIGUE_FEATURES + MATCHUP_FEATURES + CONTACT_FEATURES + WEATHER_FEATURES),
        ("5. + calendar & park splits (shrunk)", all_features),
    ]

    results = []
    fitted = {}
    for label, cols in ladder:
        model, calibrator, method = fit_calibrated(train_df, calib_df, cols, C=MODEL_C)
        probs = predict_calibrated(model, calibrator, test_df[cols])
        result = evaluate(test_df["hit_hr"], probs, label=label)
        result["calibration"] = method
        results.append(result)
        fitted[label] = (model, calibrator, cols, probs)
        print_evaluation(result)
        print(f"  Calibration chosen    : {method}")

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    summary = compare_results(results)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_label = summary.loc[summary["brier_skill"].idxmax(), "label"]
    print(f"\nBest by Brier skill: {best_label}")
    if summary["brier_skill"].max() <= 0:
        print("\nWARNING: no configuration beats simply predicting the base rate.")
        print("That is a real result, not a bug -- it would mean the printed")
        print("percentage carries no information beyond 'this pool of sluggers")
        print("homers about X% of the time'. Report it as such.")

    # ---------------- Best model: is the edge real? --------------------
    best_model, best_calibrator, best_cols, best_probs = fitted[best_label]

    print("\n" + "=" * 70)
    print("IS THE BEST MODEL'S EDGE REAL?")
    print("Two independent checks. A weak signal needs both -- one window")
    print("with a tight CI proves very little on its own.")
    print("=" * 70)
    print_bootstrap(bootstrap_brier_skill(test_df["hit_hr"], best_probs))
    print_rolling_origin(rolling_origin_evaluation(features_df, best_cols, C=MODEL_C))

    print("\n--- What the edge is worth in plain terms ---")
    base_rate = float(test_df["hit_hr"].mean())
    print(f"  Predicting the base rate for every game : {base_rate:.1%}")
    print(f"  This model's range across test games    : "
          f"{best_probs.min():.1%} to {best_probs.max():.1%}")
    print(f"  Standard deviation of its predictions   : {best_probs.std():.4f}")
    print(f"  So it moves the number roughly +/-{2 * best_probs.std():.1%} around")
    print(f"  the base rate. Any card showing a number AT the base rate is")
    print(f"  showing you the base rate, whatever it is labelled.")

    # ---------------- Reliability of the full model --------------------
    full_model, full_calibrator, full_cols, full_probs = fitted[ladder[-1][0]]
    print_calibration_report(
        test_df["hit_hr"], full_probs, n_bins=8,
        label="V4 full model (calibrated)",
    )
    # Second view of the same question. Isotonic output clusters on a few
    # discrete values, which leaves the equal-width table above with
    # near-empty buckets; equal-count buckets keep every row interpretable.
    print_reliability_by_quantile(
        test_df["hit_hr"], full_probs, n_bins=8,
        label="V4 full model (calibrated)",
    )

    print("\n" + "=" * 70)
    print("WHICH FACTORS ACTUALLY MOVE THE NUMBER")
    print("(standardised coefficients -- log-odds change per 1 SD of the feature)")
    print("=" * 70)
    print(coefficient_report(full_model, full_cols).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print_univariate_ranking(univariate_ranking(features_df, all_features))

    # ---------------- Are the superstitions real? ----------------------
    if RUN_PERMUTATION_TESTS:
        print("\n" + "=" * 70)
        print("PERMUTATION TESTS -- is a feature group signal or folklore?")
        print("Each shuffles that group's columns and refits, building a null")
        print("distribution. If the real model isn't better than its own")
        print("shuffled self, the group is noise.")
        print("=" * 70)

        for name, suspect in [
            ("Calendar & park splits (venue / weekday / home / day-night)", CALENDAR_FEATURES),
            ("Weather (temperature / wind)", WEATHER_FEATURES),
            ("Contact quality (sweet spot / barrel / hard hit)", CONTACT_FEATURES),
        ]:
            result = permutation_test(
                train_df, calib_df, test_df, all_features, suspect,
                n_permutations=N_PERMUTATIONS,
            )
            print_permutation_result(name, result)

    # ---------------- Persist ------------------------------------------
    try:
        import joblib
        os.makedirs("model/artifacts", exist_ok=True)
        joblib.dump(
            {
                "model": full_model,
                "calibrator": full_calibrator,
                "feature_cols": full_cols,
                "trained_through": str(calib_df["game_date"].max().date()) if len(calib_df) else None,
                "test_metrics": results[-1],
            },
            "model/artifacts/hr_v4.joblib",
        )
        print("\nSaved fitted model to model/artifacts/hr_v4.joblib")
    except ImportError:
        print("\n(joblib not installed -- model not saved. `pip install joblib` to enable.)")

    print("\nRemember what this model is and isn't: it produces an HONEST")
    print("probability, not a confident call. Read the Brier skill score, and")
    print("read the reliability table, before trusting any single percentage.")


if __name__ == "__main__":
    main()
