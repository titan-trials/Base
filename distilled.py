"""
Read the small tables distill_caches.py writes, if they are there.

Every loader here returns None when the distilled file is missing, so a lab
importing this keeps its original path to the raw statcast caches and
nothing breaks on a machine that has never run the distiller.

WHY A LAB SHOULD PREFER THESE
-----------------------------
Speed is the small reason: seconds instead of minutes, over 1.25 GB of
files that get parsed and then almost entirely discarded.

The real reason is reproducibility. On 2026-09-14 every lab in this project
ran against whatever `cache/statcast_pitcher_*.csv` happened to be sitting
next to it, and the cloud sandbox held 50 of the 196 files that existed.
Nothing failed. Nothing warned. The measurements simply used a quarter of
the evidence, and `lab_log5.py` reported a -0.72% effect at z = -1.27 that
turned out to be nothing at all once the other 146 caches were included.

A distilled table is small enough to move between machines, which means two
runs of the same lab can be made to mean the same thing. Every loader below
prints what it is handing over for exactly that reason -- a silent loader
is how the first mistake stayed invisible.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "distilled")


def _read(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def starts(announce=True):
    """One row per start: pitcher, date, bf, k, velo, opp, home."""
    df = _read("starts.csv")
    if df is None or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["pitcher", "date"])
    if announce:
        print(f"  [distilled] {len(df):,} starts, "
              f"{df.pitcher.nunique()} pitchers")
    return df


def matchups(announce=True):
    """
    One row per hitter seen exactly three times in a start.

    k1/k2/k3 are his outcome on each look; d31 and d21 are the paired
    differences with the hitter, pitcher, park, day and umpire all
    differenced away.
    """
    df = _read("matchups.csv")
    if df is None or df.empty:
        return None
    if announce:
        print(f"  [distilled] {len(df):,} three-look matchups, "
              f"{df.pitcher.nunique()} pitchers")
    return df


def arsenal(announce=False):
    """One row per pitcher: pitch-mix entropy, types thrown, fastball share."""
    df = _read("arsenal.csv")
    if df is None or df.empty:
        return None
    if announce:
        print(f"  [distilled] arsenal for {len(df)} pitchers")
    return df
