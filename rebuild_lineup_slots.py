"""
Rebuild cache/lineup_slots.csv from the Statcast caches -- no network.

    python rebuild_lineup_slots.py            # writes cache/lineup_slots.csv
    python rebuild_lineup_slots.py --check    # only compares against _pa_export

WHY THIS EXISTS
---------------
On 2026-09-05 test_slate_dryrun.py overwrote the real lineup cache with a
fabricated one (its `_write_synthetic_slots` wrote to the live path -- now
fixed to a temp file). The real file was 222k rows of slots fetched from
MLB boxscores over several weeks. Rather than re-pull 14,000 boxscores,
this reconstructs the same information from data already on disk.

HOW A LINEUP FALLS OUT OF STATCAST
----------------------------------
`at_bat_number` counts plate appearances through the whole game, both
teams, from 1. Within one side (Top = away batting, Bot = home batting)
the first nine DISTINCT batters, in at_bat_number order, ARE the starting
lineup in slot order -- that is what a batting order means. The only way
to get it wrong is a substitution before a starter's first plate
appearance, which is rare enough to ignore.

The catch is coverage: a hitter's cache holds only his own plate
appearances. But every plate appearance is in exactly two caches -- the
batter's and the pitcher's -- and the pitcher caches hold EVERY batter a
pitcher faced. Pooling every cached row and de-duplicating on
(game_pk, at_bat_number) gives, for most games since 2022, the complete
sequence of plate appearances. A side's lineup is written only when every
at_bat_number from 1 up to that side's ninth distinct batter is present,
so nothing is inferred across a gap.

Validated against cache/_pa_export.csv, an August 30 export of 44,840
real (batter, game, slot) rows from the original boxscore pull.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from data.cache import cache_path
from data.refresh import load_player_cache
from data.pitcher_data import load_pitcher_cache
from features.pa_table import NON_PA_EVENTS

COLS = ["game_pk", "at_bat_number", "batter", "inning_topbot", "game_date"]
CACHE_DIR = os.path.dirname(cache_path("x"))


def _ids(prefix):
    out = []
    for path in glob.glob(os.path.join(CACHE_DIR, f"{prefix}*.csv")):
        stem = os.path.basename(path)[len(prefix):-4]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(set(out))


def load_all_pa(verbose=True):
    frames = []
    scratch = os.environ.get("K_BACKTEST_SCRATCH")
    raw_pkl = os.path.join(scratch, "hitters_raw.pkl") if scratch else None
    if raw_pkl and os.path.exists(raw_pkl):
        raw = pd.read_pickle(raw_pkl)
        frames.append(raw[raw["events"].notna() & ~raw["events"].isin(NON_PA_EVENTS)][COLS])
        if verbose:
            print(f"  hitters: {len(frames[-1]):,} plate appearances from scratch pickle")
    else:
        ids = _ids("statcast_player_")
        for i, pid in enumerate(ids, 1):
            raw = load_player_cache(pid)
            if raw.empty or "at_bat_number" not in raw.columns:
                continue
            frames.append(raw[raw["events"].notna() & ~raw["events"].isin(NON_PA_EVENTS)][COLS])
            if verbose and i % 100 == 0:
                print(f"  hitters: {i}/{len(ids)}")
    for i, pid in enumerate(_ids("statcast_pitcher_"), 1):
        raw = load_pitcher_cache(pid)
        if raw.empty or "at_bat_number" not in raw.columns:
            continue
        frames.append(raw[raw["events"].notna() & ~raw["events"].isin(NON_PA_EVENTS)][COLS])
    pa = pd.concat(frames, ignore_index=True)
    pa = pa.dropna(subset=["at_bat_number", "batter"])
    pa["at_bat_number"] = pa["at_bat_number"].astype(int)
    pa["batter"] = pa["batter"].astype(int)
    pa = pa.drop_duplicates(subset=["game_pk", "at_bat_number"])
    if verbose:
        print(f"  {len(pa):,} distinct plate appearances across "
              f"{pa['game_pk'].nunique():,} games")
    return pa


def reconstruct(pa: pd.DataFrame, verbose=True) -> pd.DataFrame:
    pa = pa.sort_values(["game_pk", "at_bat_number"])
    rows, sides_ok, sides_seen = [], 0, 0
    for gpk, g in pa.groupby("game_pk", sort=False):
        present = set(g["at_bat_number"].tolist())
        for half, side in g.groupby("inning_topbot"):
            sides_seen += 1
            seen, order, needed_to = set(), [], None
            for ab, b in zip(side["at_bat_number"], side["batter"]):
                if b not in seen:
                    seen.add(b)
                    order.append(b)
                if len(order) == 9:
                    needed_to = ab
                    break
            if needed_to is None:
                continue
            if not all(n in present for n in range(1, needed_to + 1)):
                continue
            sides_ok += 1
            for slot, b in enumerate(order, start=1):
                rows.append((int(gpk), b, slot))
    out = pd.DataFrame(rows, columns=["game_pk", "batter", "lineup_slot"])
    out["is_starter"] = 1
    # Only games with BOTH lineups reconstructed are written. The boxscore
    # fetcher (data/lineup_slots.get_lineup_slots) decides what to fetch by
    # game_pk, so a game written with one side would never get its other
    # side filled in.
    per_game = out.groupby("game_pk").size()
    complete = per_game[per_game == 18].index
    out = out[out["game_pk"].isin(complete)].reset_index(drop=True)
    if verbose:
        print(f"  {sides_ok:,} of {sides_seen:,} team-games reconstructed; "
              f"{len(complete):,} games with both lineups written "
              f"({len(out):,} starter rows). The rest are left for the "
              f"boxscore fetcher on the next predict_slate run.")
        if "game_date" in pa.columns:
            years = (pa.drop_duplicates("game_pk").set_index("game_pk")["game_date"]
                       .reindex(complete))
            years = pd.to_datetime(years).dt.year.value_counts().sort_index()
            print("  complete games by season: "
                  + ", ".join(f"{y}: {n}" for y, n in years.items()))
    return out


def check(slots: pd.DataFrame) -> None:
    path = cache_path("_pa_export")
    if not os.path.exists(path):
        print("  (no _pa_export.csv to validate against)")
        return
    truth = pd.read_csv(path)[["batter", "game_pk", "lineup_slot", "is_starter"]]
    truth = truth[truth["is_starter"] == 1]
    m = truth.merge(slots, on=["game_pk", "batter"], how="left",
                    suffixes=("_true", "_rebuilt"))
    covered = m["lineup_slot_rebuilt"].notna()
    agree = (m.loc[covered, "lineup_slot_true"] == m.loc[covered, "lineup_slot_rebuilt"])
    print(f"  Validation against _pa_export: {covered.mean():.1%} of "
          f"{len(m):,} real starter rows covered; slot agrees on "
          f"{agree.mean():.2%} of those.")
    if agree.mean() < 0.98:
        bad = m[covered & ~agree.reindex(m.index, fill_value=False)]
        print(bad.head(10).to_string(index=False))


if __name__ == "__main__":
    print("Loading plate appearances...")
    pa = load_all_pa()
    slots = reconstruct(pa)
    check(slots)
    if "--check" in sys.argv:
        raise SystemExit(0)
    out = cache_path("lineup_slots")
    if os.path.exists(out):
        existing = pd.read_csv(out)
        # Keep any real rows for games this rebuild could not cover.
        keep = existing[~existing["game_pk"].isin(slots["game_pk"].unique())]
        slots = pd.concat([slots, keep], ignore_index=True)
        print(f"  Kept {len(keep):,} rows for games not covered by the rebuild.")
    slots.to_csv(out, index=False)
    print(f"  Wrote {len(slots):,} rows to cache/lineup_slots.csv")
