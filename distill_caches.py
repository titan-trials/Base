"""
Reduce the statcast caches to the few small tables the labs actually read.

    python distill_caches.py
    python distill_caches.py --force     # rebuild even if up to date

WHY
---
`cache/statcast_pitcher_*.csv` is 196 files and about 1.25 GB. Every lab in
this project opens all of them, parses ~100 columns, and then immediately
throws away all but six or seven. That costs minutes per run on this
machine, and it makes the labs unrunnable anywhere else: the cloud sandbox
that writes them held 50 of the 196 caches, which is why every measurement
taken on 2026-09-14 ran on a quarter of the evidence without anyone
noticing until `pull_caches.py --verify` printed the real number.

What the labs consume is tiny:

    starts.csv      one row per start        ~18,000 rows
    matchups.csv    one row per hitter seen  ~90,000 rows
                    three times in a start
    arsenal.csv     one row per pitcher      ~200 rows

Call it 4 MB against 1.25 GB. Small enough to move, small enough to keep,
and small enough that a lab reading it starts in under a second.

This is not a cache of convenience. It is the difference between a
measurement that can be reproduced and checked and one that quietly runs
on whatever subset of the files happened to be sitting next to it.

WHAT IS AND IS NOT IN HERE
--------------------------
Only aggregates. No pitch-level rows, no plate-appearance rows. That is
deliberate: lab_log5.py needs plate appearances and is already answered at
288k of them, and lab_tto.py needs them and is already decisive at
z = -8.84. Neither is worth carrying 15 MB for. If a future question needs
plate appearances, add a fourth table then rather than now.

Nothing is recomputed or adjusted. Every number here is a straight
aggregation of what statcast recorded.
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
OUT = os.path.join(CACHE, "distilled")

K_EVENTS = {"strikeout", "strikeout_double_play"}
TTO_COL = "n_priorpa_thisgame_player_at_bat"
USE = ["game_pk", "game_date", "game_type", "events", "batter", "inning",
       "inning_topbot", "home_team", "away_team", "at_bat_number",
       "pitch_number", "pitch_type", "release_speed", "player_name",
       TTO_COL]
FASTBALLS = {"FF", "SI", "FT", "FC"}
MIN_BF = 8


def one(path):
    pid = int(re.search(r"(\d+)", os.path.basename(path)).group(1))
    try:
        raw = pd.read_csv(path, usecols=lambda c: c in USE, low_memory=False)
    except Exception:
        return None, None, None
    if raw.empty or "events" not in raw.columns:
        return None, None, None
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"] == "R"]
    if raw.empty:
        return None, None, None

    # ---- arsenal, over every pitch ---------------------------------
    ars = None
    if "pitch_type" in raw.columns:
        mix = raw["pitch_type"].dropna()
        mix = mix[mix.ne("") & ~mix.isin({"PO", "IN", "UN"})]
        if len(mix) >= 500:
            share = mix.value_counts(normalize=True)
            ars = {"pitcher": pid,
                   "n_pitch_types": int((share >= 0.10).sum()),
                   "mix_entropy": float(-(share * np.log(share)).sum()),
                   "fastball_share": float(
                       share[share.index.isin(FASTBALLS)].sum()),
                   "pitches": int(len(mix))}

    raw = raw.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    # The pitcher is on the FIELDING side, so Top of the inning -- the away
    # team batting -- means HIS team is home. Backwards, this silently
    # prints a pitcher's own team as his opponent, which looks plausible
    # and is wrong every time. pitcher_form.py makes the same note.
    if "inning_topbot" in raw.columns:
        at_home = raw["inning_topbot"].eq("Top")
        raw = raw.assign(opp=np.where(at_home, raw.get("away_team"),
                                      raw.get("home_team")),
                         home=at_home)
    else:
        raw = raw.assign(opp="", home=False)
    raw["is_k"] = raw["events"].isin(K_EVENTS)
    raw["is_pa"] = raw["events"].notna() & raw["events"].ne("")

    agg = {"date": ("game_date", "first"), "opp": ("opp", "first"),
           "home": ("home", "first"), "pitches": ("events", "size"),
           "bf": ("is_pa", "sum"), "k": ("is_k", "sum")}
    if "inning" in raw.columns:
        agg["last_inning"] = ("inning", "max")
    starts = raw.groupby("game_pk").agg(**agg).reset_index()

    # Fastball-only speed, joined separately so a start with no fastball
    # comes back NaN rather than as the mean of his breaking stuff.
    if {"pitch_type", "release_speed"} <= set(raw.columns):
        fb = (raw[raw["pitch_type"].isin(FASTBALLS)]
              .groupby("game_pk")["release_speed"].mean().rename("velo"))
        starts = starts.merge(fb, on="game_pk", how="left")
    else:
        starts["velo"] = np.nan
    starts = starts[starts["bf"] >= MIN_BF]
    if starts.empty:
        return None, None, ars
    starts["pitcher"] = pid
    if "player_name" in raw.columns and len(raw):
        starts["name"] = str(raw["player_name"].iloc[0])
    else:
        starts["name"] = ""

    # ---- matchups: a hitter seen exactly three times in one start ----
    mu = None
    if TTO_COL in raw.columns:
        pa = raw[raw["is_pa"]].copy()
        pa["tto"] = pd.to_numeric(pa[TTO_COL], errors="coerce") + 1
        pa = pa.dropna(subset=["tto"])
        # Order matters and is not cosmetic. lab_tto_who.py applies the
        # 12-batter start filter to ALL plate appearances and only then
        # restricts to the first three looks, so a start that reached a
        # fourth time through counts those batters toward the threshold.
        # Restricting first drops a handful of starts the lab keeps, and
        # the distilled table stops reproducing the lab it feeds. Caught
        # by 22,978 matchups here against 22,979 there.
        keep = pa.groupby("game_pk")["is_k"].transform("size") >= 12
        pa = pa[keep]
        pa = pa[pa["tto"].between(1, 3)]
        if not pa.empty:
            grp = pa.groupby(["game_pk", "batter"])["is_k"]
            pa = pa[grp.transform("size") == 3]
        if not pa.empty:
            mu = (pa.pivot_table(index=["game_pk", "batter"], columns="tto",
                                 values="is_k", aggfunc="first")
                    .rename(columns={1.0: "k1", 2.0: "k2", 3.0: "k3"})
                    .dropna().reset_index())
            mu["pitcher"] = pid
    return starts, mu, ars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv")))
    if not paths:
        print("No statcast pitcher caches found.")
        return 1
    os.makedirs(OUT, exist_ok=True)

    newest = max(os.path.getmtime(p) for p in paths)
    target = os.path.join(OUT, "starts.csv")
    if os.path.exists(target) and os.path.getmtime(target) > newest \
            and not args.force:
        print(f"Up to date ({len(paths)} caches). --force to rebuild.")
        return 0

    print(f"distilling {len(paths)} caches...")
    S, M, A = [], [], []
    for i, p in enumerate(paths, 1):
        s, m, a = one(p)
        if s is not None:
            S.append(s)
        if m is not None:
            M.append(m)
        if a is not None:
            A.append(a)
        if i % 25 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")

    if not S:
        print("Nothing usable.")
        return 1

    starts = pd.concat(S, ignore_index=True)
    starts["date"] = pd.to_datetime(starts["date"], errors="coerce")
    starts = starts.dropna(subset=["date"]).sort_values(["pitcher", "date"])
    starts["date"] = starts["date"].dt.strftime("%Y-%m-%d")
    starts.to_csv(target, index=False)

    mu = pd.concat(M, ignore_index=True) if M else pd.DataFrame()
    if not mu.empty:
        mu["start"] = (mu["pitcher"].astype(str) + "_"
                       + mu["game_pk"].astype(str))
        mu["d31"] = mu["k3"] - mu["k1"]
        mu["d21"] = mu["k2"] - mu["k1"]
        mu.to_csv(os.path.join(OUT, "matchups.csv"), index=False)
    ars = pd.DataFrame(A)
    if not ars.empty:
        ars.to_csv(os.path.join(OUT, "arsenal.csv"), index=False)

    def mb(name):
        p = os.path.join(OUT, name)
        return os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0

    raw_gb = sum(os.path.getsize(p) for p in paths) / 1e9
    total = mb("starts.csv") + mb("matchups.csv") + mb("arsenal.csv")
    print(f"\n  starts.csv    {len(starts):>7,} rows  {mb('starts.csv'):>6.1f} MB"
          f"   {starts.pitcher.nunique()} pitchers")
    print(f"  matchups.csv  {len(mu):>7,} rows  {mb('matchups.csv'):>6.1f} MB")
    print(f"  arsenal.csv   {len(ars):>7,} rows  {mb('arsenal.csv'):>6.1f} MB")
    print(f"\n  {total:.1f} MB from {raw_gb:.2f} GB "
          f"({total / (raw_gb * 1000):.2%} of the source)")
    print(f"  written to cache/distilled/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
