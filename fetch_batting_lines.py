"""
One-time (resumable) fetch of official batting lines.

    python fetch_batting_lines.py

Pulls hits / runs / RBI per player per game from MLB boxscores for every
game your cached hitters appear in. Needed because the H+R+RBI prop
requires RUNS SCORED, which Statcast only records as free text.

Roughly 30 minutes for a full 13,800-game backfill, checkpointed every
100 games. Safe to stop with Ctrl+C and restart -- it resumes from the
cache and only fetches what's missing.

Run this once. After that `predict_slate.py` reads from cache and adding
each night's new games costs a handful of calls.
"""
import glob
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from data.batting_lines import get_batting_lines
from data.cache import cache_path


def main():
    print("=" * 70)
    print("FETCHING OFFICIAL BATTING LINES (hits / runs / RBI)")
    print("=" * 70)

    paths = sorted(glob.glob(
        os.path.join(os.path.dirname(cache_path("x")), "statcast_player_*.csv")))
    if not paths:
        raise SystemExit(
            "No cached players found. Run predict_slate.py or "
            "build_wide_pool.py first."
        )

    print(f"  Scanning {len(paths)} cached players for game ids...")
    games = set()
    for path in paths:
        try:
            df = pd.read_csv(path, usecols=["game_pk"], low_memory=False)
        except Exception:
            continue
        games.update(df["game_pk"].dropna().astype("int64").unique().tolist())

    print(f"  {len(games):,} unique games.\n")
    lines = get_batting_lines(sorted(games))

    if lines.empty:
        raise SystemExit("Nothing fetched. Check network access to statsapi.mlb.com.")

    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    real = lines[lines["player_id"] > 0]
    print(f"  {len(real):,} batting lines across {real['game_pk'].nunique():,} games")
    print(f"  mean per player-game:  {real['hits'].mean():.3f} hits, "
          f"{real['runs'].mean():.3f} runs, {real['rbi'].mean():.3f} RBI")
    print(f"  mean H+R+RBI        :  "
          f"{(real['hits'] + real['runs'] + real['rbi']).mean():.3f}")

    composite = real["hits"] + real["runs"] + real["rbi"]
    print(f"\n  H+R+RBI distribution per player-game:")
    counts = composite.value_counts().sort_index()
    for value, n in counts.head(9).items():
        print(f"    {value}: {n / len(real):.4f}")
    for line in (0.5, 1.5, 2.5, 3.5):
        print(f"  P(over {line}) = {(composite > line).mean():.4f}")

    print("\n  Compare these to a real sportsbook line to sanity-check the")
    print("  definition before trusting any model built on it.")
    print(f"\n  Cached to cache/batting_lines.csv")
    print("  Now re-run: python predict_slate.py 2026-08-18")


if __name__ == "__main__":
    main()
