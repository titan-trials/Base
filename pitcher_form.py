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
       "inning", "inning_topbot", "events", "player_name"]
# Below this a "start" is a relief outing that happens to be in the cache,
# and it would drag a starter's recent-form line down for no reason.
MIN_BF = 8


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
    g = g[g["bf"] >= MIN_BF]
    if g.empty:
        return None
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g = g.dropna(subset=["game_date"]).sort_values("game_date", ascending=False)
    g["pitcher_id"] = int(pitcher_id)
    g["name"] = str(raw["player_name"].iloc[0]) if "player_name" in raw else ""
    return g.head(keep)


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
    out = out[["pitcher_id", "name", "game_date", "opp", "home", "pitches",
               "bf", "k", "last_inning"]]

    path = os.path.join(CACHE, f"pitcher_form_{date}.csv")
    out.to_csv(path, index=False)

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
