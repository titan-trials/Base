"""
Score a past slate against what actually happened.

    python score_slate.py 2026-08-18      # after the games are final

WHY THIS IS THE ONLY TEST THAT REALLY COUNTS
--------------------------------------------
Every number quoted for this model so far comes from a BACKTEST, and that
backtest is mildly optimistic for an unavoidable reason: several modelling
choices were made while looking at test-set results. The regularisation
strength, the overdispersion correction, the per-PA shape method, when to
stop tuning -- each of those decisions saw the answer before it was made.
That's researcher degrees of freedom, and it inflates measured skill even
when nobody intends it to.

A prediction written down BEFORE the games, then scored after, has none of
that. Nothing about tonight's outcome could have leaked backwards into a
file that already existed. This is the honest measurement.

ONE SLATE IS NOISE. TWENTY IS A MEASUREMENT.
--------------------------------------------
A single day is ~270 hitters, of whom maybe 60 homer. The Brier skill from
one slate has an error bar wide enough to swallow the entire effect --
recall the backtest edge is around +0.02, and a one-day estimate will
scatter far wider than that.

So every scored slate appends to `cache/scoring_log.csv`, and the report
shows both the day and the RUNNING TOTAL across every slate scored so far.
The running total is the number to watch. Expect the first several days to
bounce around and mean nothing.

WHAT GETS SCORED
----------------
Each binary prop (home run, hit, walk, strikeout) and each hits+RBI+walks
line, against three references:

    base rate    always predict the day's observed rate -- the null
    model        what was written down before the games
    calibration  did the printed percentage match the observed frequency

A player who didn't actually play is DROPPED, not counted as a failure.
The model predicted "if he bats"; a scratched hitter is a lineup question,
not a modelling error. How many were dropped is reported, because a
projection that keeps guessing the wrong nine hitters is a real problem
even though it isn't this model's problem.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import START_DATE
from data.cache import cache_path
from data.refresh import refresh_players
from features.pa_table import build_pa_table, build_game_totals
from data.batting_lines import get_batting_lines
from features.run_features import build_run_training_frame
from model.hr_v4 import evaluate, bootstrap_brier_skill, reliability_by_quantile

SCORING_LOG_KEY = "scoring_log"

# prediction column -> (actual column, human label, is_count_line)
BINARY_PROPS = {
    "prob_hr": ("got_hr", "at least 1 home run"),
    "prob_hit": ("got_hit", "at least 1 hit"),
    "prob_walk": ("got_walk", "at least 1 walk"),
}
HRR_PREFIX = "prob_hrr_over_"


def load_predictions(game_date: str) -> pd.DataFrame:
    path = cache_path(f"slate_{game_date}")
    if not os.path.exists(path):
        raise SystemExit(
            f"No predictions found at cache/slate_{game_date}.csv -- "
            f"run `python predict_slate.py {game_date}` first."
        )
    predictions = pd.read_csv(path)
    if "player_id" not in predictions.columns:
        raise SystemExit(
            "This slate file predates player_id being exported. Re-run "
            f"`python predict_slate.py {game_date}` to regenerate it "
            "(predictions will differ -- that no longer counts as a clean "
            "out-of-sample test for this date)."
        )
    return predictions


def load_actuals(predictions: pd.DataFrame, game_date: str,
                 verbose: bool = True) -> pd.DataFrame:
    """Actual per-game outcomes for the hitters we predicted."""
    player_ids = sorted(predictions["player_id"].dropna().astype(int).unique())
    names = dict(zip(predictions["player_id"], predictions["name"]))

    if verbose:
        print(f"Refreshing Statcast so {game_date} results are present...")
    combined = refresh_players(player_ids, START_DATE, game_date, names=names)

    pa = build_pa_table(combined)
    totals = build_game_totals(pa)
    totals["game_date"] = pd.to_datetime(totals["game_date"])
    played = totals[totals["game_date"] == pd.Timestamp(game_date)]

    if played.empty:
        raise SystemExit(
            f"No completed plate appearances found for {game_date}. Either the "
            f"games haven't finished, or Statcast hasn't posted them yet "
            f"(usually a few hours after the final out)."
        )

    # ---- Official hits / runs / RBI -----------------------------------
    #
    # The H+R+RBI prop is graded on the OFFICIAL line, not on a Statcast
    # reconstruction. Runs scored aren't in Statcast at all, and the RBI
    # derived from score changes credits runs an official scorer would
    # charge to an error. Grading a model against an approximation of the
    # thing it predicted measures the approximation as much as the model.
    #
    # These games are force-refetched: a line cached while the game was
    # still in progress must not become the number this is scored against.
    if verbose:
        print("Fetching official boxscore lines for the graded games...")
    todays_games = played["game_pk"].unique()
    lines = get_batting_lines(todays_games, verbose=verbose, refetch=todays_games)
    if lines.empty:
        raise SystemExit(
            f"No official batting lines for {game_date}. The boxscores may not "
            f"be final yet -- try again once the last game has ended."
        )

    played = build_run_training_frame(played, lines)
    missing = int(played["hrr"].isna().sum())
    if missing and verbose:
        print(f"  WARNING: {missing} of {len(played)} hitters have no official "
              f"line and cannot be scored on H+R+RBI.")
    if verbose:
        matched = played["hrr"].notna()
        print(f"  Official H+R+RBI for {int(matched.sum())} hitters, "
              f"mean {played.loc[matched, 'hrr'].mean():.3f}.")
    return played


def score_prop(merged: pd.DataFrame, prediction_col: str, actual_col: str,
               label: str) -> dict:
    subset = merged.dropna(subset=[prediction_col, actual_col])
    if len(subset) < 30 or subset[actual_col].nunique() < 2:
        return None

    result = evaluate(subset[actual_col], subset[prediction_col], label=label)
    result["prediction_col"] = prediction_col
    return result


def main(game_date: str = None):
    if game_date is None:
        raise SystemExit("Usage: python score_slate.py YYYY-MM-DD")

    print("=" * 72)
    print(f"SCORING SLATE: {game_date}")
    print("=" * 72)

    # Check the calendar BEFORE doing anything expensive.
    #
    # load_actuals() refreshes ~270 players over the network, then checks
    # whether any results came back. Run against a future date that means
    # several minutes of pulling to arrive at "there is nothing to score" --
    # an answer available instantly from the date alone.
    today = pd.Timestamp.today().normalize()
    target = pd.Timestamp(game_date).normalize()
    if target > today:
        raise SystemExit(
            f"\n  {game_date} is in the future -- those games haven't been "
            f"played yet.\n"
            f"  Today is {today.date()}. Come back after the games are final.\n\n"
            f"  Statcast posts a game a few hours after the last out, so for a\n"
            f"  night slate the safe time to score is the following morning.\n\n"
            f"  In the meantime `python eval_rbi.py` runs fine -- it only uses\n"
            f"  history, so it needs no future games."
        )
    if target == today:
        print(f"  NOTE: {game_date} is today. Late games may not be final, and\n"
              f"  Statcast lags a few hours behind the last out. Any hitter\n"
              f"  whose game isn't posted yet will be dropped as 'did not bat',\n"
              f"  which understates coverage. Scoring tomorrow is cleaner.\n")

    predictions = load_predictions(game_date)
    print(f"  {len(predictions)} predictions written before the games.")
    if "lineup_status" in predictions.columns:
        print(f"  Lineup status when written: "
              f"{predictions['lineup_status'].value_counts().to_dict()}")

    # Verify the prediction actually predates the games. Without this the
    # scoring loop can be silently poisoned: regenerate a slate after the
    # fact and the "out-of-sample" result becomes a measure of hindsight.
    if "predicted_at_utc" in predictions.columns and "start_time_utc" in predictions.columns:
        written = pd.to_datetime(predictions["predicted_at_utc"], utc=True,
                                 errors="coerce").min()
        first_pitch = pd.to_datetime(predictions["start_time_utc"], utc=True,
                                     errors="coerce").min()
        if pd.notna(written) and pd.notna(first_pitch):
            if written >= first_pitch:
                print("\n  *** WARNING: predictions were written AFTER first pitch ***")
                print(f"  written {written}, first pitch {first_pitch}.")
                print("  This is NOT a clean out-of-sample test -- the model's")
                print("  rolling rates may already contain these games' outcomes.")
                print("  Scoring anyway, but do not add this to the running total.\n")
            else:
                lead = (first_pitch - written).total_seconds() / 3600.0
                print(f"  Written {lead:.1f}h before first pitch -- clean "
                      f"out-of-sample.")
    else:
        print("  NOTE: this slate has no predicted_at_utc stamp, so it cannot be")
        print("  verified as pre-game. Treat the result with caution.")

    actuals = load_actuals(predictions, game_date)

    merged = predictions.merge(
        actuals.rename(columns={"batter": "player_id"}),
        on="player_id", how="inner", suffixes=("", "_actual"),
    )
    dropped = len(predictions) - len(merged)
    print(f"\n  {len(merged)} of {len(predictions)} predicted hitters actually "
          f"batted ({dropped} did not).")
    if dropped and len(predictions):
        print(f"  Those {dropped} are excluded, not counted as misses -- the "
              f"model predicted 'if he bats'.")
        print(f"  A high number here means the LINEUP PROJECTION is off, which "
              f"is worth fixing separately.")

    if len(merged) < 30:
        raise SystemExit("Too few matched hitters to score anything meaningful.")

    # ---- Binary props -------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS -- model versus 'just quote the base rate'")
    print("=" * 72)

    rows = []
    for prediction_col, (actual_col, label) in BINARY_PROPS.items():
        if prediction_col not in merged or actual_col not in merged:
            continue
        result = score_prop(merged, prediction_col, actual_col, label)
        if result:
            rows.append(result)

    # ---- Count lines --------------------------------------------------
    # Each prop: (prediction-column prefix, actual column, label). Total
    # bases and hits come straight off the PA table -- both settle inside
    # the plate appearance, so unlike H+R+RBI they need no official
    # boxscore join to be scored honestly.
    COUNT_PROPS = [
        (HRR_PREFIX, "hrr", "H+R+RBI"),
        ("prob_tb_over_", "total_bases", "Total bases"),
        ("prob_hits_over_", "hits", "Hits"),
    ]
    for prefix, actual_col, label in COUNT_PROPS:
        if actual_col not in merged.columns:
            print(f"  (no `{actual_col}` column -- skipping {label})")
            continue
        for column in [c for c in merged.columns if c.startswith(prefix)]:
            line = float(column[len(prefix):])
            flag = f"_actual_{actual_col}_over_{line}"
            merged[flag] = (merged[actual_col] > line).astype(int)
            result = score_prop(merged, column, flag, f"{label} over {line}")
            if result:
                rows.append(result)

    if not rows:
        raise SystemExit("Nothing scoreable in this slate file.")

    table = pd.DataFrame(rows)[
        ["label", "n", "base_rate", "mean_pred", "auc", "brier", "brier_skill"]
    ]
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  base_rate  = what actually happened")
    print("  mean_pred  = what the model said on average (should be close)")
    print("  brier_skill = the headline. Positive beats quoting the base rate.")

    # ---- Is one day enough? (no) --------------------------------------
    headline = next((r for r in rows if r["label"].startswith("H+R+RBI over 1.5")),
                    rows[0])
    subset = merged.dropna(subset=[headline["prediction_col"]])
    actual_col = ("_actual_over_1.5" if headline["label"].startswith("H+R+RBI")
                  else BINARY_PROPS[headline["prediction_col"]][0])
    if actual_col in subset:
        boot = bootstrap_brier_skill(subset[actual_col],
                                     subset[headline["prediction_col"]])
        print(f"\n  {headline['label']}: Brier skill {boot['point']:+.4f}, "
              f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
        print("  On one slate that interval will be wide. That is expected and")
        print("  is exactly why the running total below is the number to read.")

    # ---- Calibration --------------------------------------------------
    if "prob_hr" in merged and "got_hr" in merged:
        print("\n--- Home run reliability (did the printed % match reality?) ---")
        relia = reliability_by_quantile(merged["got_hr"], merged["prob_hr"], n_bins=4)
        print(relia.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- Append to the running log ------------------------------------
    log_path = cache_path(SCORING_LOG_KEY)
    entry = table.copy()
    entry.insert(0, "game_date", game_date)
    if os.path.exists(log_path):
        previous = pd.read_csv(log_path)
        previous = previous[previous["game_date"] != game_date]  # allow re-scoring
        log = pd.concat([previous, entry], ignore_index=True)
    else:
        log = entry
    log.to_csv(log_path, index=False)

    print("\n" + "=" * 72)
    print(f"RUNNING TOTAL across {log['game_date'].nunique()} scored slate(s)")
    print("=" * 72)
    running = log.groupby("label").apply(
        lambda g: pd.Series({
            "slates": g["game_date"].nunique(),
            "n": g["n"].sum(),
            # Weight by sample size: a 269-hitter slate should not count
            # the same as a rained-out 80-hitter one.
            "auc": np.average(g["auc"], weights=g["n"]),
            "brier_skill": np.average(g["brier_skill"], weights=g["n"]),
        })
    ).reset_index()
    print(running.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if log["game_date"].nunique() < 5:
        print(f"\n  Only {log['game_date'].nunique()} slate(s) scored. Treat all of")
        print("  this as noise until roughly 15-20 slates have accumulated.")
    else:
        print("\n  Compare the running Brier skill to the backtest figures in")
        print("  CONTEXT.md. If live is meaningfully lower, the backtest was")
        print("  optimistic -- which is worth knowing and worth writing down.")

    print(f"\n  Log: cache/{SCORING_LOG_KEY}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
