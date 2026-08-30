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
from data.pitching_lines import get_pitching_lines
from features.run_features import build_run_training_frame
from model.hr_v4 import evaluate, bootstrap_brier_skill, reliability_by_quantile

SCORING_LOG_KEY = "scoring_log"

# Per-row flag: was this hitter's prediction written before HIS game started?
# Leading underscore so it never collides with a real slate column and is
# obvious as internal if it ever leaks into a printout.
CLEAN_COL = "_clean_row"

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


def clean_mask(frame) -> pd.Series:
    """
    True for rows whose prediction was written before THAT row's game began.

    NaT on either side yields False rather than NaN: a row we cannot date is
    a row we cannot vouch for, and the safe default for "is this evidence"
    is no.
    """
    if "predicted_at_utc" not in frame.columns or "start_time_utc" not in frame.columns:
        return pd.Series(False, index=frame.index)
    written = pd.to_datetime(frame["predicted_at_utc"], utc=True, errors="coerce")
    start = pd.to_datetime(frame["start_time_utc"], utc=True, errors="coerce")
    return (written < start).fillna(False)


# Each count prop: (prediction-column prefix, actual column, label). Total
# bases and hits come straight off the PA table -- both settle inside the
# plate appearance, so unlike H+R+RBI they need no official boxscore join
# to be scored honestly.
COUNT_PROPS = [
    (HRR_PREFIX, "hrr", "H+R+RBI"),
    ("prob_tb_over_", "total_bases", "Total bases"),
    ("prob_hits_over_", "hits", "Hits"),
]


def iter_scoreable(frame):
    """
    Yield (label, prediction_col, truth) for every prop this frame can score.

    One definition of "what the props are and how each one is graded",
    used by the main scorer and by the form-marker self-test. Two functions
    deriving the same thing from the same source is how they quietly drift
    apart -- the lesson already written into data/batting_lines.py.
    """
    for prediction_col, (actual_col, label) in BINARY_PROPS.items():
        if prediction_col in frame and actual_col in frame:
            yield label, prediction_col, frame[actual_col]

    for prefix, actual_col, label in COUNT_PROPS:
        if actual_col not in frame.columns:
            continue
        for column in [c for c in frame.columns if c.startswith(prefix)]:
            line = float(column[len(prefix):])
            yield (f"{label} over {line}", column,
                   (frame[actual_col] > line).astype(int))


def score_frame(frame) -> list:
    """
    Every prop scored against one frame. Returns a list of result dicts.

    Pulled out of main() so the identical scoring can run twice -- once over
    the whole slate and once over the clean subset -- with no chance of the
    two drifting into slightly different definitions.
    """
    work = frame.copy()
    rows = []
    for label, prediction_col, truth in iter_scoreable(work):
        flag = f"_truth_{label}"
        work[flag] = truth.to_numpy()
        result = score_prop(work, prediction_col, flag, label)
        if result:
            rows.append(result)
    return rows


FORM_LOG_KEY = "form_log"
PITCHER_LOG_KEY = "pitcher_scoring_log"
K_LINES = (5.5, 6.5)


def score_pitchers(game_date: str):
    """
    Grade last night's starter strikeout props.

    Separate from the hitter scorer because the units are different -- one
    row per starter rather than per hitter, so about fifteen rows a slate
    against 250. That matters for how the results are read: a single
    slate's Brier skill on fifteen starters is almost pure noise, and only
    the running total is worth anything.

    Batters faced is graded alongside strikeouts on purpose. It is half the
    model and the half more likely to be wrong, and during the build a
    1.50-batter error in it sat hidden behind a 6% error in the strikeout
    rate running the other way. Two numbers that can each be checked beat
    one number that can only be checked jointly.
    """
    path = cache_path(f"pitchers_{game_date}")
    if not os.path.exists(path):
        return None
    props = pd.read_csv(path)
    if props.empty or "prob_k_over_5.5" not in props.columns:
        return None

    print("\n" + "=" * 72)
    print(f"PITCHER STRIKEOUTS -- {game_date}")
    print("=" * 72)

    lines = get_pitching_lines(props["game_pk"].unique(),
                               refetch=props["game_pk"].unique())
    if lines.empty:
        print("  No official pitching lines available yet. Try again later.")
        return None

    merged = props.merge(
        lines[lines["is_starter"] == 1][
            ["game_pk", "pitcher_id", "batters_faced", "strikeouts",
             "innings_pitched"]],
        on=["game_pk", "pitcher_id"], how="inner")

    dropped = len(props) - len(merged)
    if dropped:
        # A projected starter who was scratched, or an opener the official
        # line disagrees with. Excluded rather than counted as a miss --
        # the model predicted "if he starts".
        print(f"  {dropped} of {len(props)} projected starters did not "
              f"actually start. Excluded, not counted as misses.")
    if merged.empty:
        print("  None of the projected starters started. Nothing to score.")
        return None

    merged[CLEAN_COL] = clean_mask(merged)
    n_clean = int(merged[CLEAN_COL].sum())
    if n_clean and n_clean < len(merged):
        print(f"  {n_clean} of {len(merged)} written before their own first "
              f"pitch. The running total uses those.")
    basis = merged[merged[CLEAN_COL]] if n_clean >= 5 else merged
    basis_name = "clean" if basis is not merged else "all"

    print(f"\n  Batters faced: predicted {basis['expected_bf'].mean():.2f}, "
          f"actual {basis['batters_faced'].mean():.2f} "
          f"({basis['expected_bf'].mean() - basis['batters_faced'].mean():+.2f})")
    print(f"  Strikeouts:    predicted {basis['expected_k'].mean():.2f}, "
          f"actual {basis['strikeouts'].mean():.2f} "
          f"({basis['expected_k'].mean() - basis['strikeouts'].mean():+.2f})")

    rows = []
    for line in K_LINES:
        col = f"prob_k_over_{line}"
        if col not in basis.columns:
            continue
        truth = (basis["strikeouts"] > line).astype(int)
        said = float(basis[col].mean())
        did = float(truth.mean())
        brier = float(((basis[col] - truth) ** 2).mean())
        ref = did * (1.0 - did)
        rows.append({
            "game_date": game_date, "line": line, "n": len(basis),
            "basis": basis_name, "mean_pred": said, "base_rate": did,
            "brier": brier,
            "brier_skill": 1.0 - brier / ref if ref > 0 else float("nan"),
            "pred_bf": float(basis["expected_bf"].mean()),
            "actual_bf": float(basis["batters_faced"].mean()),
            "pred_k": float(basis["expected_k"].mean()),
            "actual_k": float(basis["strikeouts"].mean()),
        })
    if not rows:
        return None

    entry = pd.DataFrame(rows)
    print()
    print(entry[["line", "n", "mean_pred", "base_rate", "brier_skill"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  On fifteen starters a single night's skill is almost all "
          "noise.")
    print("  The running total below is the number that means something.")

    log_path = cache_path(PITCHER_LOG_KEY)
    if os.path.exists(log_path):
        previous = pd.read_csv(log_path)
        previous = previous[previous["game_date"] != game_date]
        log = pd.concat([previous, entry], ignore_index=True)
    else:
        log = entry
    log.to_csv(log_path, index=False)

    print("\n" + "-" * 72)
    print(f"RUNNING TOTAL across {log['game_date'].nunique()} slate(s)")
    print("-" * 72)

    def _pool(g):
        w = g["n"].to_numpy(dtype=float)
        base = float(np.average(g["base_rate"], weights=w))
        brier = float(np.average(g["brier"], weights=w))
        ref = base * (1.0 - base)
        return pd.Series({
            "slates": g["game_date"].nunique(), "n": int(w.sum()),
            "said": float(np.average(g["mean_pred"], weights=w)),
            "did": base,
            # Pooled, not an average of per-slate skills -- see the note in
            # the hitter running total for why those differ.
            "brier_skill": 1.0 - brier / ref if ref > 0 else float("nan"),
            "pred_bf": float(np.average(g["pred_bf"], weights=w)),
            "actual_bf": float(np.average(g["actual_bf"], weights=w)),
        })

    running = log.groupby("line")[
        ["n", "base_rate", "brier", "mean_pred", "pred_bf", "actual_bf",
         "game_date"]].apply(_pool).reset_index()
    print(running.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Backtest on 1,101 held-out starts gave +0.0635 (over 5.5) "
          f"and +0.0765 (over 6.5).")
    print(f"  If live settles well below that, the backtest was optimistic "
          f"and it is worth knowing.")
    print(f"\n  Log: cache/{PITCHER_LOG_KEY}.csv")
    return running


def score_form_marker(frame, game_date: str):
    """
    Does the hot/cold marker predict anything?

    The marker feeds no model, so it cannot be graded by Brier skill the
    way a prop is. The question it answers is narrower and simpler: among
    hitters flagged Hot, did MORE happen than the model predicted for them,
    and among those flagged Cold, less?

    That comparison is the right one precisely because the model does not
    know about the flag. Its probabilities are an unbiased forecast made in
    ignorance of form, so any consistent gap between predicted and actual
    inside a flagged group is information the model is missing.

    Reported in standard errors, because on one slate a dozen flagged
    hitters will drift a couple of points in some direction regardless.

    HOW LONG THIS TAKES TO ANSWER
    -----------------------------
    Simulated, 500 runs per point, 270 hitters a slate, no optional
    stopping -- the smallest number of slates giving 80% power to clear
    2 sigma on the SLOPE test:

        true slope   a 2sd-hot hitter beats forecast by   slates
          +0.005                1.0 points                 >120
          +0.010                2.0 points                   90
          +0.015                3.0 points                   45
          +0.020                4.0 points                   20
          +0.030                6.0 points                   12

    With a true slope of zero the test fires 5.4% of the time at 30
    slates, which is what a 2-sigma threshold should do and confirms the
    pooling is calibrated rather than merely plausible.

    So: if form carries an effect as large as the hot-hand literature
    suggests, twenty slates or so. If it is a one-point effect, this will
    never resolve it and the honest answer will be "too small to matter",
    which is also worth knowing.

    Appends to cache/form_log.csv so the answer accumulates.
    """
    if "form_state" not in frame.columns or "form_z" not in frame.columns:
        return None

    rows = []
    for state in ("Hot", "Cold"):
        sub = frame[frame["form_state"] == state]
        # Anything non-empty is logged. A group of four gives a noisy
        # per-slate z and a perfectly good contribution to the pooled
        # average, and discarding it would throw away evidence that never
        # comes back.
        if sub.empty:
            continue
        for label, prediction_col, truth in iter_scoreable(sub):
            said = float(sub[prediction_col].mean())
            did = float(truth.mean())
            n = len(sub)
            se = (said * (1.0 - said) / n) ** 0.5 if 0 < said < 1 else float("nan")
            rows.append({
                "game_date": game_date, "state": state, "label": label,
                "n": n, "said": said, "did": did,
                "gap": did - said,
                "z": (did - said) / se if se and se > 0 else float("nan"),
            })

    # The comparison above is the readable one and the WEAK one. Roughly
    # twelve hitters a slate get flagged, so detecting an effect the size
    # Green and Zwiebel measured would take several hundred slates -- not a
    # self-test, a career.
    #
    # The strong version uses every hitter. Regress each hitter's residual
    # (what happened minus what the model said) on his continuous form_z.
    # If form carries information the model lacks, that slope is positive:
    # hitters running hot beat their forecast in proportion to how hot.
    # 270 rows a slate instead of 12 is more than twenty times the
    # evidence, and it uses the size of the deviation rather than throwing
    # it away at a threshold.
    slopes = []
    have_z = frame[frame["form_z"].notna()]
    if len(have_z) >= 30:
        x = have_z["form_z"].to_numpy(dtype=float)
        xc = x - x.mean()
        sxx = float((xc ** 2).sum())
        for label, prediction_col, truth in iter_scoreable(have_z):
            y = (truth.to_numpy(dtype=float)
                 - have_z[prediction_col].to_numpy(dtype=float))
            if sxx <= 0 or len(y) < 30:
                continue
            slope = float((xc * (y - y.mean())).sum() / sxx)
            resid = y - y.mean() - slope * xc
            dof = len(y) - 2
            se = ((resid ** 2).sum() / (dof * sxx)) ** 0.5 if dof > 0 else float("nan")
            slopes.append({
                "game_date": game_date, "state": "SLOPE", "label": label,
                "n": len(y), "said": float("nan"), "did": float("nan"),
                "gap": slope, "z": slope / se if se and se > 0 else float("nan"),
                "slope_se": se,
            })
    rows.extend(slopes)

    if not rows:
        return None

    entry = pd.DataFrame(rows)
    log_path = cache_path(FORM_LOG_KEY)
    if os.path.exists(log_path):
        previous = pd.read_csv(log_path)
        previous = previous[previous["game_date"] != game_date]
        log = pd.concat([previous, entry], ignore_index=True)
    else:
        log = entry
    log.to_csv(log_path, index=False)

    n_hot = int((frame["form_state"] == "Hot").sum())
    n_cold = int((frame["form_state"] == "Cold").sum())
    print("\n" + "=" * 72)
    print(f"FORM MARKER SELF-TEST -- {n_hot} hot, {n_cold} cold "
          f"(model does not use this)")
    print("=" * 72)

    # Pool across every slate scored so far, weighting by group size.
    def _pool(g):
        w = g["n"].to_numpy(dtype=float)
        said = float(np.average(g["said"], weights=w))
        did = float(np.average(g["did"], weights=w))
        total = float(w.sum())
        se = (said * (1.0 - said) / total) ** 0.5 if 0 < said < 1 else float("nan")
        return pd.Series({"slates": g["game_date"].nunique(), "n": total,
                          "said": said, "did": did, "gap": did - said,
                          "z": (did - said) / se if se and se > 0 else float("nan")})

    # ---- The headline: does form_z predict the residual? ---------------
    slope_log = log[log["state"] == "SLOPE"].copy()
    if not slope_log.empty and "slope_se" in slope_log.columns:
        # Inverse-variance pooling across slates -- the standard way to
        # combine independent estimates of the same quantity, and correct
        # here because each slate is a separate sample of the same slope.
        out = []
        for label, g in slope_log.groupby("label"):
            g = g[g["slope_se"].notna() & (g["slope_se"] > 0)]
            if g.empty:
                continue
            w = 1.0 / g["slope_se"].to_numpy(dtype=float) ** 2
            b = float((g["gap"].to_numpy(dtype=float) * w).sum() / w.sum())
            se = float((1.0 / w.sum()) ** 0.5)
            out.append({"label": label, "slates": g["game_date"].nunique(),
                        "n": int(g["n"].sum()), "slope": b, "se": se,
                        "z": b / se if se > 0 else float("nan")})
        if out:
            table = pd.DataFrame(out).sort_values("z", ascending=False)
            print("\n  Does form predict the residual? "
                  "(slope of actual-minus-predicted on form_z)")
            print(table.to_string(index=False,
                                  float_format=lambda v: f"{v:+.4f}"))
            print("\n  slope = how much better a hitter does per 1 sd of form.")
            print("          +0.02 would mean a hitter 2 sd hot beats his")
            print("          forecast by about 4 percentage points.")
            print("  Positive and growing |z| means the marker is real.")

    # ---- The readable version, on the flagged groups only --------------
    pooled = (log[log["state"] != "SLOPE"]
              .groupby(["state", "label"])[["n", "said", "did", "game_date"]]
              .apply(_pool).reset_index())
    for state, sign in (("Hot", +1), ("Cold", -1)):
        part = pooled[pooled["state"] == state]
        if part.empty:
            continue
        print(f"\n  {state} hitters, pooled across "
              f"{int(part['slates'].max())} slate(s):")
        print(part[["label", "n", "said", "did", "gap", "z"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        agree = float((np.sign(part["gap"]) == sign).mean())
        print(f"  {agree:.0%} of props move in the direction the marker "
              f"predicts ({'above' if sign > 0 else 'below'} forecast).")

    print("\n  Read the SLOPE table, not the group table. Only about a dozen")
    print("  hitters a slate get flagged, so the group comparison would need")
    print("  hundreds of slates to resolve anything; the slope uses all 270.")
    print("  And note the props are near-restatements of each other, so a")
    print("  column of agreeing signs is closer to one observation than to")
    print("  fourteen.")
    print(f"\n  Log: cache/{FORM_LOG_KEY}.csv")
    return pooled


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

    # Cleanliness is a property of each GAME, not of the slate.
    #
    # Lineups get posted at different times and games start at different
    # times, so a normal night means re-running the predictor a few times as
    # cards are confirmed. A prediction written at 6:34pm is worthless for
    # the 12:35pm game -- that one is already inside the rolling rates -- and
    # perfectly clean for the 10:40pm game, which has not been played yet.
    #
    # The previous version of this compared ONE write time against the
    # EARLIEST first pitch and condemned the entire night on that basis. On
    # 2026-08-19 that meant discarding 162 legitimate out-of-sample rows in
    # order to avoid 108 tainted ones.
    #
    # Worse, it printed "do not add this to the running total" and then added
    # it to the running total anyway, four hundred lines further down. The
    # instruction was addressed to the human and ignored by the program.
    predictions[CLEAN_COL] = clean_mask(predictions)
    if CLEAN_COL not in predictions or not predictions[CLEAN_COL].notna().any():
        print("  NOTE: this slate has no predicted_at_utc stamp, so it cannot be")
        print("  verified as pre-game. Treat the result with caution.")
    else:
        n_clean = int(predictions[CLEAN_COL].sum())
        games_all = predictions["game_pk"].nunique()
        games_clean = predictions.loc[predictions[CLEAN_COL], "game_pk"].nunique()
        if n_clean == len(predictions):
            print(f"  All {n_clean} hitters were written before their game's "
                  f"first pitch -- fully clean slate.")
        elif n_clean == 0:
            print("\n  *** EVERY game had already started when this was "
                  "written ***")
            print("  There is no clean subset here. The full-slate numbers are")
            print("  reported below and logged, but they measure hindsight, not")
            print("  prediction. Do not read them as model performance.\n")
        else:
            print(f"  Clean: {n_clean} hitters across {games_clean} of "
                  f"{games_all} games, written before their own first pitch.")
            print(f"  Tainted: {len(predictions) - n_clean} hitters across "
                  f"{games_all - games_clean} games already underway.")
            print(f"  Both are scored below. The headline and the running "
                  f"total use the clean rows.")

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

    for missing in ("hrr", "total_bases", "hits"):
        if missing not in merged.columns:
            print(f"  (no `{missing}` column -- skipping those lines)")

    rows = score_frame(merged)
    if not rows:
        raise SystemExit("Nothing scoreable in this slate file.")

    COLS = ["label", "n", "base_rate", "mean_pred", "auc", "brier", "brier_skill"]
    table = pd.DataFrame(rows)[COLS]

    # The same scoring over the rows that were genuinely predicted in
    # advance. score_prop returns None below 30 rows, so a night with only
    # one or two early games simply produces no clean table rather than a
    # meaningless one computed from 14 hitters.
    clean_frame = merged[merged[CLEAN_COL]] if CLEAN_COL in merged else merged.iloc[0:0]
    rows_clean = score_frame(clean_frame) if len(clean_frame) >= 30 else []
    table_clean = (pd.DataFrame(rows_clean)[COLS] if rows_clean
                   else pd.DataFrame(columns=COLS))

    fully_clean = len(clean_frame) == len(merged) and len(merged) > 0
    print()
    if fully_clean:
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        # Show them next to each other. The whole point of keeping both is
        # that the difference between them is the size of the leakage, and
        # a number you have to go and look up somewhere else is a number
        # nobody looks up.
        side = table.merge(table_clean, on="label", how="left",
                           suffixes=("", "_clean"))
        view = side[["label", "n", "brier_skill", "n_clean", "brier_skill_clean"]]
        view = view.rename(columns={"n": "n_all", "brier_skill": "skill_all",
                                    "n_clean": "n_clean",
                                    "brier_skill_clean": "skill_clean"})
        view["gap"] = view["skill_all"] - view["skill_clean"]
        print(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\n  skill_all   = every hitter, including games already underway")
        print("  skill_clean = only hitters predicted before their game began")
        print("  gap         = how much the tainted rows flattered the result.")
        print("                Large and positive means hindsight was doing the work.")
        print("\n  Full detail on the clean rows:")
        if rows_clean:
            print(table_clean.to_string(index=False,
                                        float_format=lambda v: f"{v:.4f}"))
        else:
            print("  (too few clean rows to score -- fewer than 30)")

    print("\n  base_rate  = what actually happened")
    print("  mean_pred  = what the model said on average (should be close)")
    print("  brier_skill = the headline. Positive beats quoting the base rate.")

    # ---- Is one day enough? (no) --------------------------------------
    # Everything below reports the CLEAN rows when there are enough of them,
    # because these are the numbers that describe the model rather than the
    # clock. `basis` names which, so the printout can never be mistaken.
    basis = clean_frame if len(clean_frame) >= 30 else merged
    basis_name = "clean rows" if basis is clean_frame else "all rows"
    rows_basis = rows_clean if basis is clean_frame else rows

    headline = next((r for r in rows_basis
                     if r["label"].startswith("H+R+RBI over 1.5")),
                    rows_basis[0] if rows_basis else None)
    if headline is not None:
        # Rebuild the outcome column here rather than reaching for one left
        # behind by the scoring loop. The old code looked for
        # "_actual_over_1.5" while the loop wrote "_actual_hrr_over_1.5", so
        # the `in subset` test was always False and this whole block quietly
        # printed nothing -- a missing confidence interval reads as "not
        # implemented yet", not as "bug".
        prediction_col = headline["prediction_col"]
        work = basis.dropna(subset=[prediction_col]).copy()
        if headline["label"].startswith("H+R+RBI") and "hrr" in work.columns:
            truth = (work["hrr"] > 1.5).astype(int)
        elif prediction_col in BINARY_PROPS and BINARY_PROPS[prediction_col][0] in work:
            truth = work[BINARY_PROPS[prediction_col][0]]
        else:
            truth = None
        if truth is not None and truth.nunique() > 1:
            boot = bootstrap_brier_skill(truth, work[prediction_col])
            print(f"\n  {headline['label']} ({basis_name}, n={len(work)}): "
                  f"Brier skill {boot['point']:+.4f}, "
                  f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
            print("  On one slate that interval will be wide. That is expected and")
            print("  is exactly why the running total below is the number to read.")

    # ---- Calibration --------------------------------------------------
    if "prob_hr" in basis and "got_hr" in basis:
        print(f"\n--- Home run reliability, {basis_name} "
              f"(did the printed % match reality?) ---")
        relia = reliability_by_quantile(basis["got_hr"], basis["prob_hr"], n_bins=4)
        print(relia.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- Append to the running log ------------------------------------
    # Both sets of numbers are kept. The unprefixed columns stay exactly as
    # they were -- every hitter scored -- so rows written by the old version
    # remain meaningful. The clean_* columns are the ones the dashboard
    # reads, and are NaN on a night with too few pre-game rows to score.
    log_path = cache_path(SCORING_LOG_KEY)
    entry = table.copy()
    entry.insert(0, "game_date", game_date)
    clean_cols = table_clean.rename(
        columns={c: f"clean_{c}" for c in COLS if c != "label"})
    entry = entry.merge(clean_cols, on="label", how="left")
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
    # The running total is the clean rows only. A slate that was entirely
    # re-run after first pitch contributes nothing to it -- which is the
    # point, and is different from the slate being deleted: its full-slate
    # numbers are still in the log and still readable.
    #
    # Rows written before clean_* existed have no clean columns at all; for
    # those, fall back to the unprefixed values rather than dropping the
    # slate. Their predictions predate the merge-on-rerun change, so the
    # distinction did not exist to be recorded.
    POOLED = ("n", "base_rate", "auc", "brier", "brier_skill")
    for col in POOLED:
        if f"clean_{col}" not in log.columns:
            log[f"clean_{col}"] = np.nan
    usable = log.copy()
    for col in POOLED:
        usable[f"clean_{col}"] = usable[f"clean_{col}"].fillna(usable[col])
    usable = usable[usable["clean_n"].fillna(0) > 0]

    def _weighted(g, value_col, weight_col):
        w = g[weight_col].to_numpy(dtype=float)
        if w.sum() <= 0:
            return float("nan")
        return float(np.average(g[value_col].to_numpy(dtype=float), weights=w))

    if usable.empty:
        print("  No clean rows on any scored slate yet -- nothing to total.")
    else:
        def _pool(g):
            # Pool the BRIER SCORES and recompute skill from them, rather
            # than averaging the per-slate skills.
            #
            # Skill is 1 - brier/(p(1-p)), and the average of a ratio is not
            # the ratio of the averages: each slate divides by its own base
            # rate, so averaging the results silently weights slates by how
            # extreme their base rate happened to be. The dashboard has
            # always pooled correctly, which is why it read +1.79% for home
            # runs while this printed +1.65% -- one quantity, two answers,
            # and no way to tell which to trust.
            #
            # Weight by sample size throughout: a 270-hitter slate should
            # not count the same as a rained-out 80-hitter one.
            base = _weighted(g, "clean_base_rate", "clean_n")
            brier = _weighted(g, "clean_brier", "clean_n")
            ref = base * (1.0 - base)
            return pd.Series({
                "slates": g["game_date"].nunique(),
                "n": g["clean_n"].sum(),
                # AUC has no closed-form pooling without the raw outcomes,
                # so this stays a weighted mean and is approximate.
                "auc": _weighted(g, "clean_auc", "clean_n"),
                "brier_skill": (1.0 - brier / ref) if ref > 0 else float("nan"),
            })

        running = usable.groupby("label").apply(_pool).reset_index()
        print(running.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\n  Clean rows only -- hitters whose prediction was written")
        print("  before their own game started.")

    if log["game_date"].nunique() < 5:
        print(f"\n  Only {log['game_date'].nunique()} slate(s) scored. Treat all of")
        print("  this as noise until roughly 15-20 slates have accumulated.")
    else:
        print("\n  Compare the running Brier skill to the backtest figures in")
        print("  CONTEXT.md. If live is meaningfully lower, the backtest was")
        print("  optimistic -- which is worth knowing and worth writing down.")

    # Graded on the CLEAN rows for the same reason everything else is: a
    # prediction made after the game started is not a forecast the marker
    # can be credited or blamed for.
    score_form_marker(basis, game_date)

    # Starters are graded from their own file and their own log --
    # fifteen rows against 250, so mixing them into the hitter totals
    # would let a noisy handful move a number built from hundreds.
    score_pitchers(game_date)

    print(f"\n  Log: cache/{SCORING_LOG_KEY}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
