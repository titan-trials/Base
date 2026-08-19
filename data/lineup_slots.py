"""
Batting-order slot per player per game, from MLB's own boxscore feed.

WHY THIS FILE EXISTS
--------------------
The V5 diagnostic found that knowing tonight's plate-appearance count
exactly would be worth about +0.046 Brier skill on "gets a hit" --
roughly TWELVE TIMES the combined contribution of every batter-skill
feature built across five versions of this project.

Plate appearances are not random. They are mostly determined by one thing
that has never been in this dataset: where the hitter bats in the order.
A leadoff hitter gets a guaranteed first-inning plate appearance and comes
up again roughly every nine batters. A number-nine hitter frequently gets
one fewer turn. Over a season that's ~90 plate appearances -- an enormous
difference in opportunity that no rate statistic captures.

Statcast has no lineup information at all. MLB's boxscore endpoint does.

THE ENCODING, WHICH IS NOT OBVIOUS
----------------------------------
`battingOrder` comes back as a three-character string:

    "100"  bats 1st, and started the game there
    "300"  bats 3rd, started
    "301"  bats 3rd, but entered as the FIRST substitute in that slot
    "902"  bats 9th, second substitute in that slot

So slot = int(value) // 100, and a value ending in "00" means the player
started. The distinction matters: a substitute who entered in the seventh
inning occupies the 3-slot but will get one plate appearance, not four.
Training a plate-appearance model on substitute appearances as though they
were starts would teach it that the 3-slot sometimes yields one PA, which
is true but useless -- before a game you know who is STARTING, so that's
what the model should be trained on.

`is_starter` is therefore kept as a column, and the PA model filters to
starters.

EFFICIENCY NOTE
---------------
One call per GAME returns every player in that game. With a wide player
pool the same games are shared across many hitters, so the number of calls
grows with games, not with players -- widening the pool from 9 to 60
hitters costs proportionally far less here than it does for the Statcast
pulls.
"""
import os
import json
import time
import urllib.request
import urllib.error

import pandas as pd

from data.cache import cache_path

CACHE_KEY = "lineup_slots"
CHECKPOINT_EVERY = 100
REQUEST_DELAY_SEC = 0.12
TIMEOUT_SEC = 20

# Field filtering keeps each response tiny -- the unfiltered boxscore is
# hundreds of kilobytes of information we don't need.
URL = (
    "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    "?fields=teams,home,away,players,person,id,battingOrder"
)

SLOT_COLUMNS = ["game_pk", "batter", "lineup_slot", "is_starter"]


def _fetch_one(game_pk: int):
    """
    Every player's batting-order slot in one game.

    Returns a list of rows on success (possibly empty if the game really
    has no lineup data), or None if the REQUEST ITSELF failed.

    That distinction matters more than it looks. An earlier version
    returned [] for both cases and then wrote a permanent "nothing here"
    sentinel to the cache. One network hiccup partway through a 5,000-game
    fetch would therefore poison those games forever -- they'd be skipped
    on every future run, and the only cure would be deleting the whole
    cache and re-fetching everything. Network failures are transient and
    must stay retryable; genuinely empty games are permanent and shouldn't
    be re-requested.
    """
    try:
        request = urllib.request.Request(
            URL.format(game_pk=int(game_pk)),
            headers={"User-Agent": "baseball_predictor/1.0"},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None  # transient: leave it uncached so the next run retries

    rows = []
    for side in ("home", "away"):
        team = (payload.get("teams") or {}).get(side) or {}
        for entry in (team.get("players") or {}).values():
            order = entry.get("battingOrder")
            person = entry.get("person") or {}
            if not order or not person.get("id"):
                continue  # pitchers and unused bench players have no slot
            try:
                order_int = int(order)
            except (TypeError, ValueError):
                continue
            rows.append({
                "game_pk": int(game_pk),
                "batter": int(person["id"]),
                "lineup_slot": order_int // 100,
                "is_starter": int(order_int % 100 == 0),
            })
    return rows


def _load_existing() -> pd.DataFrame:
    path = cache_path(CACHE_KEY)
    if not os.path.exists(path):
        return pd.DataFrame(columns=SLOT_COLUMNS)
    return pd.read_csv(path)[SLOT_COLUMNS]


def get_lineup_slots(game_pks, verbose: bool = True) -> pd.DataFrame:
    """
    One row per (game, player) with lineup slot and starter flag.

    Cached and checkpointed exactly like data/game_context.py -- games
    already fetched are skipped, and a run interrupted halfway resumes
    from the last checkpoint rather than starting over.

    Games that fail are recorded with a sentinel row (slot -1) so they
    aren't retried forever. Delete cache/lineup_slots.csv to force a full
    refetch.
    """
    wanted = sorted({int(pk) for pk in pd.Series(list(game_pks)).dropna().unique()})
    cached = _load_existing()
    have = set(cached["game_pk"].astype(int)) if not cached.empty else set()
    missing = [pk for pk in wanted if pk not in have]

    if verbose:
        print(f"  Lineup slots: {len(wanted)} games wanted, "
              f"{len(have & set(wanted))} cached, {len(missing)} to fetch.")

    if not missing:
        return cached[cached["game_pk"].isin(wanted) & (cached["lineup_slot"] > 0)]

    new_rows, network_failures = [], 0
    for i, game_pk in enumerate(missing, start=1):
        rows = _fetch_one(game_pk)

        if rows is None:
            network_failures += 1
            # Bail out rather than grind through thousands of failing
            # requests: if the first handful all fail, the endpoint is
            # unreachable (no network, firewall, API down) and continuing
            # just wastes minutes to arrive at the same answer.
            if network_failures >= 20 and network_failures == i:
                print(f"    Aborting: the first {i} requests all failed. "
                      f"Check network access to statsapi.mlb.com.")
                break
            continue

        if not rows:
            # Genuinely no lineup data for this game (postponed, spring
            # training). Safe to remember permanently.
            rows = [{"game_pk": game_pk, "batter": -1,
                     "lineup_slot": -1, "is_starter": 0}]
        new_rows.extend(rows)
        time.sleep(REQUEST_DELAY_SEC)

        if i % CHECKPOINT_EVERY == 0 and new_rows:
            pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True) \
                .to_csv(cache_path(CACHE_KEY), index=False)
            if verbose:
                print(f"    ...{i}/{len(missing)} games fetched (checkpointed)")

    if network_failures and verbose:
        print(f"    {network_failures} request(s) failed and were NOT cached "
              f"-- rerun to retry just those.")
    if not new_rows:
        return cached[cached["game_pk"].isin(wanted) & (cached["lineup_slot"] > 0)]

    combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_pk", "batter"], keep="last")
    combined.to_csv(cache_path(CACHE_KEY), index=False)
    if verbose:
        real = (combined["lineup_slot"] > 0).sum()
        print(f"  Cached {real:,} player-game slots to cache/{CACHE_KEY}.csv")

    result = combined[combined["game_pk"].isin(wanted)]
    return result[result["lineup_slot"] > 0].reset_index(drop=True)


def attach_lineup_slots(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Left-join lineup slot onto any frame keyed by (batter, game_pk).

    Rows with no match keep NaN rather than a filled-in guess. A missing
    slot usually means the player pinch-hit, and quietly imputing "he
    probably batted fifth" would be inventing the single most important
    feature in the model.
    """
    slots = get_lineup_slots(df["game_pk"].unique(), verbose=verbose)
    if slots.empty:
        df = df.copy()
        df["lineup_slot"] = float("nan")
        df["is_starter"] = 0
        return df

    merged = df.copy()
    merged["game_pk"] = merged["game_pk"].astype("int64")
    merged["batter"] = merged["batter"].astype("int64")
    slots["game_pk"] = slots["game_pk"].astype("int64")
    slots["batter"] = slots["batter"].astype("int64")
    return merged.merge(slots, on=["game_pk", "batter"], how="left")
