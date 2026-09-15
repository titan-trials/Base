"""
Recent form for tonight's starters, small enough to commit.

    python pitcher_form.py                 # today
    python pitcher_form.py 2026-09-13
    python pitcher_form.py --starts 15     # deeper history per pitcher

WHY THIS IS A SEPARATE FILE AND NOT A DASHBOARD QUERY
-----------------------------------------------------
The dashboard is deployed. `.gitignore` tracks `cache/slate_*.csv`,
`cache/pitchers_*.csv` and the accumulating logs, and deliberately does
NOT track `cache/statcast_pitcher_*.csv` -- those are ~1.2 GB and
regenerable, and putting them in git is what took .git to 1.1 GB once
already.

So the deployed app cannot read a pitcher's game log, however much it
would like to. Anything the Pitchers tab wants to show about recent form
has to be computed HERE, on the machine that has the statcast caches, and
written to a small tracked file.

Thirty starters at ten starts each is 300 rows, about 25 KB a night. That
is the same bargain `pitchers_{date}.csv` already makes.

WHAT IT DOES NOT CONTAIN, AND WHY
---------------------------------
No versus-opponent column. It was measured (`lab_opponent.py`) and it does
not survive a split-half test: r = +0.019 against a null of -0.001 +/-
0.030 across 869 pitcher-opponent pairings. There IS real overdispersion
in the residuals -- about 2.3 points of strikeout rate, z = +5.3 -- but it
does not repeat, which means it is drift and within-start clustering
rather than a property of the pairing. A column showing it would be
showing noise with a confident label on it.

Home and away DO have sample: the lighter of a pitcher's two halves still
holds a median 46 starts and 1,069 plate appearances across the 50 cached
starters. That split is real, and it is here. Whether it PREDICTS anything
has not been tested -- it is shown as a fact about his record, not as a
signal, and it should not be fed to anything until it has been through the
same persistence test the opponent split failed.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

K_EVENTS = {"strikeout", "strikeout_double_play"}
# Columns read from a ~100-column, 6 MB statcast cache. Reading only these
# cuts both by an order of magnitude, and this script opens thirty files.
USE = ["game_pk", "game_date", "game_type", "home_team", "away_team",
       "inning", "inning_topbot", "events", "player_name",
       "pitch_type", "release_speed"]
# Below this a "start" is a relief outing that happens to be in the cache,
# and it would drag a starter's recent-form line down for no reason.
MIN_BF = 8

# ---- the velocity form marker ---------------------------------------
#
# Measured, not assumed. lab_pitcher_form.py ran both candidate signals
# over 4,599 cached starts with a de-meaned within-pitcher test, a
# shuffled null and the league's seasonal arc removed:
#
#     recent K rate, z     r = +0.019  p = 0.210   <- noise
#     fastball mph         r = +0.052  p = 0.001   <- real, z = +3.2
#
# So this marker is built on VELOCITY and deliberately NOT on recent
# strikeout rate, which is the obvious version and was tested and failed.
# A physical measurement beats an outcome filtered through defence and
# luck -- the same argument context-features-availability.md makes for
# exit velocity on the hitter side.
#
# Four-seam, sinker, cutter only. A changeup's speed moves with the
# fastball and a curveball's does not, so mixing them in measures pitch
# selection rather than arm speed.
FASTBALLS = {"FF", "SI", "FT", "FC"}
# Three starts is roughly 70 batters, above the ~60 PA where a strikeout
# rate starts to mean something, and short enough to still be "recent".
VELO_WINDOW = 3
# Half a mile an hour off his OWN norm. Chosen from the data rather than
# picked round: at +/-0.5 the gap between cold and hot starts is 1.85
# points of strikeout rate at z = +3.8, the best of the cuts tried, and
# it fires on 40% of starts (20% each way) rather than on almost all of
# them like the hitter marker does.
VELO_CUT = 0.5


# ---- the strikeout form marker --------------------------------------
#
# Velocity measures how hard he is throwing. This measures how he has
# actually been DOING. They are not the same thing and they disagree:
# Landen Roupp on 2026-09-15 sat at velocity z = -0.21 (normal) while his
# last four starts ran 11 K in 89 batters against his own 21.8% -- a
# strikeout-form z of -2.15. The velocity marker had nothing to say about
# him. This one does.
#
# DISPLAY ONLY. It is not fed to predict_slate and must not be. Recent K
# rate IS real (lab_pitcher_form.py, r = +0.0335, z = +3.54 over 11,148
# starts) but it does NOT survive next to velocity -- the two correlate at
# +0.19, and fitted jointly recent form drops to z = +1.91 and about 0.08 K
# while velocity is untouched. Feeding both would be counting most of the
# same signal twice. This exists to say "look closer", not to move a
# number.
#
# WINDOW AND CUT, MEASURED
# ------------------------
# Over ~10,500 starts, de-meaned within pitcher, the gap in NEXT-start K
# rate between flagged-hot and flagged-cold:
#
#     window 3, cut 1.50   fires 20%   +2.59%   z = +6.05
#     window 4, cut 1.25   fires 30%   +1.81%   z = +5.13   <- this
#     window 4, cut 1.50   fires 21%   +1.63%   z = +3.92
#     window 5, cut 1.25   fires 31%   +1.64%   z = +4.69
#
# Those cells are inside each other's error bars (all +/- 0.3 to 0.4), so
# this is not a claim that 4/1.25 beats 3/1.50. Four starts is ~90 batters
# rather than ~70, which is the more stable window, and it is the stretch a
# person actually eyeballs.
#
# THE NUMBER THIS IS NOT
# ----------------------
# The first version of this measurement reported +4.5% at z = +12.67 and
# was wrong. `z` is computed against his baseline and the outcome was
# measured against the SAME baseline, so baseline noise pushed both the
# same way. De-meaning within pitcher removes it and the effect drops by a
# factor of three. Same shared-denominator trap as the order-penalty lab.
KFORM_WINDOW = 4
KFORM_CUT = 1.25


def per_start(pitcher_id, keep):
    """
    His most recent outings, newest first, or None.

    One row per game: date, opponent, side, pitch count, batters faced,
    strikeouts, and the last inning he appeared in.
    """
    path = os.path.join(CACHE, f"statcast_pitcher_{int(pitcher_id)}.csv")
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_csv(path, usecols=lambda c: c in USE, low_memory=False)
    except Exception:
        return None
    if raw.empty or "inning_topbot" not in raw.columns:
        return None
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"] == "R"]
    if raw.empty:
        return None

    # The pitcher is on the FIELDING side, so Top of the inning -- the away
    # team batting -- means HIS team is the home team and his opponent is
    # the away team. Getting this backwards silently prints a pitcher's own
    # team as his opponent, which looks plausible and is wrong every time.
    at_home = raw["inning_topbot"].eq("Top")
    raw = raw.assign(
        opp=np.where(at_home, raw["away_team"], raw["home_team"]),
        home=at_home,
        is_k=raw["events"].isin(K_EVENTS),
        # A plate appearance is a pitch that carries an outcome. Every
        # other row in the file is a pitch inside one.
        is_pa=raw["events"].notna() & raw["events"].ne(""))

    g = (raw.groupby("game_pk")
            .agg(game_date=("game_date", "first"), opp=("opp", "first"),
                 home=("home", "first"), pitches=("events", "size"),
                 bf=("is_pa", "sum"), k=("is_k", "sum"),
                 last_inning=("inning", "max"))
            .reset_index())
    # Fastball-only mean speed, joined separately so a start with no
    # fastball in it comes back NaN rather than as the mean of his
    # breaking stuff.
    if {"pitch_type", "release_speed"} <= set(raw.columns):
        fb = (raw[raw["pitch_type"].isin(FASTBALLS)]
              .groupby("game_pk")["release_speed"].mean().rename("velo"))
        g = g.merge(fb, on="game_pk", how="left")
    else:
        g["velo"] = float("nan")
    g = g[g["bf"] >= MIN_BF]
    if g.empty:
        return None
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g = g.dropna(subset=["game_date"]).sort_values("game_date", ascending=False)
    g["pitcher_id"] = int(pitcher_id)

    # ---- the velocity marker, computed on the FULL cache -------------
    #
    # Before the head(keep) below, because the baseline has to be his
    # whole record and the file only keeps the last ten starts.
    #
    # The baseline EXCLUDES the recent window. An earlier version averaged
    # everything including the last three starts, which puts the thing
    # being measured inside its own yardstick and shrinks every drop
    # toward zero -- the same shape of mistake as a rolling feature that
    # forgets to shift. lab_pitcher_form.py uses shift(1).expanding(), and
    # this is that.
    velo = g["velo"].dropna()               # already newest-first
    if len(velo) >= VELO_WINDOW + 3:
        recent = float(velo.iloc[:VELO_WINDOW].mean())
        base = float(velo.iloc[VELO_WINDOW:].mean())
        sd = float(velo.iloc[VELO_WINDOW:].std(ddof=1))
        drop = recent - base
        state = ("Hot" if drop >= VELO_CUT
                 else "Cold" if drop <= -VELO_CUT else "Normal")
    else:
        recent = base = drop = sd = float("nan")
        state = "Unknown"                   # too little history to say
    g["velo_recent"] = recent
    g["velo_base"] = base
    g["velo_drop"] = drop
    g["velo_state"] = state
    # The drop in units of his OWN start-to-start variation, which is what
    # the projection consumes. A mile an hour means something different for
    # an arm that ranges two ticks a night than for one that never moves,
    # and lab_pitcher_form.py measured the effect per standard deviation.
    # Measured on the baseline starts only, for the same reason the mean is:
    # including the window puts the thing being measured inside its own
    # yardstick.
    g["velo_sd"] = sd
    g["velo_z"] = (drop / sd) if (np.isfinite(sd) and sd > 0) else float("nan")

    # ---- strikeout form, same shape, different question --------------
    #
    # Computed on the FULL cache for the same reason velocity is: the file
    # only keeps the last ten starts and the baseline has to be his whole
    # record. Baseline EXCLUDES the recent window -- including it puts the
    # thing being measured inside its own yardstick.
    #
    # The z is against binomial error on his own rate, not against a
    # spread of past z-scores, so a short window over few batters is
    # correctly harder to flag than a long one.
    if len(g) >= KFORM_WINDOW + 5:
        rec = g.iloc[:KFORM_WINDOW]
        old_g = g.iloc[KFORM_WINDOW:]
        rec_bf, old_bf = float(rec["bf"].sum()), float(old_g["bf"].sum())
        if rec_bf >= 30 and old_bf > 0:
            k_base = float(old_g["k"].sum()) / old_bf
            k_rec = float(rec["k"].sum()) / rec_bf
            k_se = np.sqrt(k_base * (1 - k_base) / rec_bf)
            k_z = (k_rec - k_base) / k_se if k_se > 0 else float("nan")
        else:
            k_base = k_rec = k_z = float("nan")
    else:
        k_base = k_rec = k_z = float("nan")
    if not np.isfinite(k_z):
        k_state = "Unknown"
    elif k_z >= KFORM_CUT:
        k_state = "Hot"
    elif k_z <= -KFORM_CUT:
        k_state = "Cold"
    else:
        k_state = "Normal"
    g["kform_base"] = k_base
    g["kform_recent"] = k_rec
    g["kform_z"] = k_z
    g["kform_state"] = k_state
    g["name"] = str(raw["player_name"].iloc[0]) if "player_name" in raw else ""
    return g.head(keep)



# ---- the velocity adjustment --------------------------------------------
#
# A starter whose fastball is down off his OWN norm misses fewer bats.
# Measured in lab_pitcher_form.py over 11,148 starts: r = +0.0815 at
# z = +8.59 against a 200-permutation null, surviving the seasonal-arc
# control, and NOT predicting how long he lasts (r = +0.008, p = 0.41) --
# so it moves the rate and must not touch the batters-faced distribution.
#
# The slope below is re-measured in the exact form it ships, which is not
# the same number the lab prints:
#
#   lab, expanding baseline, uncapped          +0.0114 per sd
#   12-month baseline (what k_rate uses)       +0.0106 per sd
#   12-month baseline, production window defn  +0.0093 per sd   <- this
#
# Both corrections matter. `k_rate` is a trailing-12-month rate, so it has
# already absorbed part of a recent decline and the full expanding-baseline
# slope would count it twice. And `pitcher_form.per_start` excludes the
# recent window from its own baseline, which the lab does not -- shipping a
# slope measured on one definition while feeding it a feature built on
# another is the mismatch this project keeps catching.
#
# n = 9,846, r = +0.0752, p = 8e-14.
VELO_K_SLOPE = 0.0093
# Clamped so a pitcher with thin or erratic velocity history cannot swing
# his own projection on very little. 4.9% of starts reach the clamp, and at
# it the adjustment is about 0.43 K over 23 batters.
VELO_Z_CAP = 2.0


def apply_velocity(k_rate, velo_z):
    """His strikeout rate, nudged by how his arm is throwing right now."""
    if velo_z is None or not np.isfinite(velo_z):
        return k_rate              # Unknown: no history, no opinion
    z = float(np.clip(velo_z, -VELO_Z_CAP, VELO_Z_CAP))
    return float(np.clip(k_rate + VELO_K_SLOPE * z, 0.01, 0.60))


# ---------------------------------------------------------------------
# what the projection calls
# ---------------------------------------------------------------------
def velocity_for(pitcher_ids, window=None):
    """
    The velocity marker for a set of pitchers, without needing a slate file.

    WHY THIS EXISTS AS A FUNCTION
    -----------------------------
    The marker used to be produced only as a side effect of `main()`, which
    reads `cache/pitchers_{date}.csv` -- a file `predict_slate` writes. So
    the marker could only ever exist AFTER the projection it should have
    been informing. It was a thing the dashboard displayed next to a number
    it had not been allowed to affect.

    `predict_slate` now calls this before it projects. Nothing about
    run_slate's step order changes, and `main()` below is untouched, so the
    dashboard file keeps being written exactly as before.

    Returns a DataFrame indexed by pitcher_id with velo_base, velo_recent,
    velo_sd, velo_drop, velo_z and velo_state. A pitcher with too little
    history is present with NaN and state "Unknown" rather than missing,
    so a caller joining on this cannot silently drop him.
    """
    keep = window or 1
    rows = []
    for pid in pd.unique(pd.Series(list(pitcher_ids)).dropna()):
        try:
            got = per_start(int(pid), keep)
        except Exception:
            got = None
        if got is None or got.empty:
            rows.append({"pitcher_id": int(pid), "velo_base": float("nan"),
                         "velo_recent": float("nan"), "velo_sd": float("nan"),
                         "velo_drop": float("nan"), "velo_z": float("nan"),
                         "velo_state": "Unknown"})
            continue
        r = got.iloc[0]
        rows.append({"pitcher_id": int(pid),
                     "velo_base": float(r["velo_base"]),
                     "velo_recent": float(r["velo_recent"]),
                     "velo_sd": float(r["velo_sd"]),
                     "velo_drop": float(r["velo_drop"]),
                     "velo_z": float(r["velo_z"]),
                     "velo_state": str(r["velo_state"])})
    return pd.DataFrame(rows).set_index("pitcher_id")


def main():
    ap = argparse.ArgumentParser(
        description="Recent-form rows for tonight's starters.")
    ap.add_argument("date", nargs="?", default=None,
                    help="YYYY-MM-DD (default: today, local)")
    ap.add_argument("--starts", type=int, default=10,
                    help="how many recent starts to keep per pitcher")
    args = ap.parse_args()
    date = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")

    pit_path = os.path.join(CACHE, f"pitchers_{date}.csv")
    if not os.path.exists(pit_path):
        print(f"No cache/pitchers_{date}.csv -- run predict_slate first.")
        return 1
    pit = pd.read_csv(pit_path)
    if "pitcher_id" not in pit.columns:
        print("pitchers file has no pitcher_id column.")
        return 1

    ids = pit["pitcher_id"].dropna().astype(int).unique()
    print(f"PITCHER FORM {date}\n  {len(ids)} starters named.")

    frames, missing = [], []
    for pid in ids:
        got = per_start(pid, args.starts)
        if got is None or got.empty:
            missing.append(int(pid))
        else:
            frames.append(got)

    if not frames:
        print("  No statcast pitcher caches matched. Nothing written.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    # Name from the slate file where we have it: statcast writes
    # "Burnes, Corbin" and every other file in this project writes
    # "Corbin Burnes". The dashboard joins on pitcher_id, but a human
    # reading the csv should not have to.
    names = dict(zip(pit["pitcher_id"].astype("Int64"), pit.get("pitcher", "")))
    out["name"] = [names.get(pd.NA if pd.isna(p) else int(p), n)
                   for p, n in zip(out["pitcher_id"], out["name"])]
    out["game_date"] = out["game_date"].dt.strftime("%Y-%m-%d")
    marker = (out.drop_duplicates("pitcher_id")
                 [["pitcher_id", "velo_base", "velo_recent", "velo_drop",
                   "velo_state", "kform_base", "kform_recent", "kform_z",
                   "kform_state"]])
    out = out[["pitcher_id", "name", "game_date", "opp", "home", "pitches",
               "bf", "k", "last_inning", "velo"]]

    path = os.path.join(CACHE, f"pitcher_form_{date}.csv")
    out.to_csv(path, index=False)

    # ---- write the marker back into pitchers_{date}.csv ---------------
    #
    # Additive columns only, on the file predict_slate just wrote. It goes
    # HERE rather than in predict_slate because the velocity history lives
    # in the statcast caches, which this script already opens and which
    # the deployed dashboard can never read.
    #
    # And it has to land in pitchers_{date}.csv specifically, not just in
    # the form file: score_slate's _log_pitcher_rows copies from there
    # into pitcher_row_log.csv, which is the only place the marker ever
    # sits next to the strikeout residual it has to be graded against.
    # A marker that cannot grade itself is decoration.
    try:
        pit_now = pd.read_csv(pit_path)
        pit_now = pit_now.drop(columns=[c for c in marker.columns
                                        if c != "pitcher_id"
                                        and c in pit_now.columns])
        pit_now = pit_now.merge(marker, on="pitcher_id", how="left")
        pit_now["velo_state"] = pit_now["velo_state"].fillna("Unknown")
        pit_now["kform_state"] = pit_now["kform_state"].fillna("Unknown")
        pit_now.to_csv(pit_path, index=False)
        counts = pit_now["velo_state"].value_counts().to_dict()
        print(f"  Velocity marker written into cache/pitchers_{date}.csv: "
              + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        kc = pit_now["kform_state"].value_counts().to_dict()
        print(f"  Strikeout-form marker: "
              + ", ".join(f"{v} {k}" for k, v in sorted(kc.items())))
    except Exception as exc:
        print(f"  Could not write the velocity marker back "
              f"({type(exc).__name__}: {exc}). The form file is still "
              f"written; the marker just will not be gradeable.")

    have = out["pitcher_id"].nunique()
    print(f"  {len(out)} starts for {have} pitchers "
          f"({out.groupby('pitcher_id').size().median():.0f} each, median).")
    if missing:
        # Not an error. A debutant has no cache, and neither does a
        # pitcher predict_slate named before his statcast pull ran.
        print(f"  {len(missing)} had no usable cache and are simply absent "
              f"from the file: {', '.join(str(m) for m in missing[:8])}"
              f"{' ...' if len(missing) > 8 else ''}")
    print(f"  wrote cache/pitcher_form_{date}.csv "
          f"({os.path.getsize(path) / 1024:.0f} KB)")

    # A quick sanity line, because a silently wrong opponent column is the
    # failure mode this script is most likely to have.
    ex = out.iloc[0]
    print(f"\n  e.g. {ex['name']}: {ex['game_date']} "
          f"{'vs' if ex['home'] else 'at'} {ex['opp']}, "
          f"{int(ex['pitches'])} pitches, {int(ex['bf'])} batters, "
          f"{int(ex['k'])} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
