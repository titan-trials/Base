"""
One-time (slow, resumable) data build for the wide player pool.

    python build_wide_pool.py

Run this once and walk away. It does the three network-bound jobs that
everything else then reads from cache:

  1. Pick a STRATIFIED pool of hitters spanning elite to replacement level
     (one API call), and pull each one's Statcast history.
  2. Pull batting-order slots for every game those hitters appear in.
  3. Pull game context (venue, day/night, weather) for the same games.

Every step is cached per item and checkpointed, so killing this halfway
and restarting picks up where it left off. Nothing is re-fetched.

EXPECT THIS TO TAKE A WHILE. The Statcast pulls are the slow part --
roughly one rate-limited request per player. Sixty players is on the order
of an hour on a first run; every run afterwards is seconds.

WHY A WIDE POOL, IN ONE PARAGRAPH
---------------------------------
The nine-slugger pool capped every model in this project. A model's job is
to rank hitters apart, and nine similar hitters give it almost nothing to
say -- the per-PA hit model scored AUC 0.511 despite hit rate being the
most repeatable skill measured (split-half r = +0.92). The skill was real;
the spread wasn't there. The same artifact made lineup slot look
irrelevant (0.9% of PA variance) because all nine bat first through
fourth. Widening the pool doesn't just add rows, it makes the question
answerable.
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from config import START_DATE, END_DATE
from data.player_pool import get_player_pool, load_pool_statcast
from data.lineup_slots import get_lineup_slots
from data.game_context import get_game_context

# --- Settings -----------------------------------------------------------
POOL_SEASON = 2025      # season whose stats define the tiers
N_PLAYERS = 60          # 60 across 3 tiers = 20 elite, 20 middle, 20 replacement
N_TIERS = 3
RANK_BY = "ops"         # "ops" for overall hitting quality, "iso" for raw power
MIN_PA = 250

FETCH_LINEUPS = True
FETCH_WEATHER = True


def main():
    print("=" * 72)
    print("STEP 1 -- choose a stratified pool and pull its Statcast history")
    print("=" * 72)
    pool = get_player_pool(
        POOL_SEASON, n_players=N_PLAYERS, n_tiers=N_TIERS,
        rank_by=RANK_BY, min_pa=MIN_PA,
    )
    print(f"\n  Pool spans OPS {pool['ops'].min():.3f} to {pool['ops'].max():.3f} "
          f"and ISO {pool['iso'].min():.3f} to {pool['iso'].max():.3f}")
    print("\n  Pulling Statcast per player (cached individually -- resumable)...")
    statcast = load_pool_statcast(pool, START_DATE, END_DATE)

    game_pks = statcast["game_pk"].dropna().unique()
    print(f"\n  {len(game_pks):,} unique games across the pool.")

    if FETCH_LINEUPS:
        print("\n" + "=" * 72)
        print("STEP 2 -- batting-order slots (one call per game, all players)")
        print("=" * 72)
        slots = get_lineup_slots(game_pks)
        if not slots.empty:
            merged = statcast[["batter", "game_pk"]].drop_duplicates().merge(
                slots, on=["batter", "game_pk"], how="inner"
            )
            print(f"\n  Matched {len(merged):,} player-games to a lineup slot.")
            print("  Slot distribution across the pool:")
            print(merged["lineup_slot"].value_counts().sort_index().to_string())

    if FETCH_WEATHER:
        print("\n" + "=" * 72)
        print("STEP 3 -- venue / day-night / weather")
        print("=" * 72)
        get_game_context(game_pks)

    print("\n" + "=" * 72)
    print("DONE. Everything is cached. Now run:")
    print("    python train_props_v5.py     (set USE_WIDE_POOL = True)")
    print("=" * 72)


if __name__ == "__main__":
    main()
