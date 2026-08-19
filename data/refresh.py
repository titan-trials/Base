"""
Incremental Statcast top-up -- keeping a player's history current without
re-pulling years of data every time.

THE PROBLEM
-----------
The original cache key was `statcast_id{id}_{start}_{end}`. That works for
a one-off backfill and breaks the moment you want today's numbers: the end
date is baked into the filename, so "2022-01-01 to 2026-08-01" and
"2022-01-01 to 2026-08-18" are different files, and asking for the second
re-pulls four and a half years to add seventeen days.

For a slate runner that has to be current, that's fatal. Sixty players
re-pulling full histories is an hour of waiting to add two weeks of games.

THE FIX
-------
One canonical file per player, `statcast_player_{id}.csv`, holding
everything known so far. Topping up means:

    1. read what's cached
    2. find the newest game_date in it
    3. pull only from that date forward
    4. append, de-duplicate, save

Re-pulling from the last cached date (rather than the day after) is
deliberate -- Statcast occasionally revises a game's rows shortly after it
finishes, and overlapping by one day lets corrections land. Duplicates are
removed on the pitch-level key afterwards, so the overlap costs nothing.

MIGRATION
---------
Old date-stamped files are read once and folded into the new canonical
file, so the hours already spent pulling are not thrown away. The old
files are left alone rather than deleted -- this project has no way to
undo a bad delete, and they're only a few megabytes.

STALENESS -- AND THE BUG THAT MADE THIS USELESS
-----------------------------------------------
The first version measured staleness as:

    (date being predicted) - (newest game in the cache)

which is wrong in a way that quietly defeated the whole point of
incremental refresh. Running on Aug 17 to predict the Aug 18 slate, the
newest game data that CAN exist is Aug 16 -- so the gap is 2 days, every
player reads as stale, and all 270 get re-pulled. Next run, same thing.
The check could never pass, because it compared against a date whose games
had not been played yet.

The fix is to track when each player was last CHECKED, not how old his
newest game is. Those are different questions, and the second one is the
wrong one twice over:

  - A hitter on the injured list has no games for three weeks. His cache
    is perfectly current; there is simply nothing to add. Under the old
    rule he looked maximally stale and got re-pulled every run forever.
  - A hitter who played last night has fresh data the moment we pull it,
    regardless of what date we happen to be predicting.

So `cache/refresh_log.csv` records a last-checked date per player, and a
player needs a pull only if he hasn't been checked today. Re-running the
same slate twice in an evening -- which is exactly what you do when
lineups post -- now costs zero network calls.

A failed pull deliberately does NOT update the log, so a network hiccup
stays retryable. Same principle as the negative-cache fix in
data/lineup_slots.py.
"""
import os
import glob
import time

import pandas as pd

from data.cache import cache_path, load_cached, save_cache

# Statcast pitch rows are uniquely identified by this combination. Used to
# drop overlap after a top-up.
DEDUPE_KEYS = ["game_pk", "at_bat_number", "pitch_number", "batter"]

REQUEST_DELAY_SEC = 0.3


def canonical_key(player_id: int) -> str:
    return f"statcast_player_{int(player_id)}"


def _find_legacy_files(player_id: int) -> list:
    """Old date-stamped caches for this player, from before the canonical
    key existed."""
    pattern = os.path.join(os.path.dirname(cache_path("x")),
                           f"statcast_id{int(player_id)}_*.csv")
    return sorted(glob.glob(pattern))


def load_player_cache(player_id: int) -> pd.DataFrame:
    """
    Everything cached for this player, migrating legacy files in on first
    use. Returns an empty frame when nothing is cached yet.
    """
    frames = []

    cached = load_cached(canonical_key(player_id))
    if cached is not None and not cached.empty:
        frames.append(cached)

    for path in _find_legacy_files(player_id):
        try:
            legacy = pd.read_csv(path, low_memory=False)
            if not legacy.empty:
                frames.append(legacy)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    combined = combined.dropna(subset=["game_date"])

    keys = [k for k in DEDUPE_KEYS if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    return combined.sort_values("game_date").reset_index(drop=True)


REFRESH_LOG_KEY = "refresh_log"


def _load_refresh_log(log_key: str = None) -> dict:
    """
    {player_id: last_checked_date} from cache/{log_key}.csv.

    `log_key` is a parameter so pitchers can keep their own log
    (cache/refresh_log_pitchers.csv) using the identical mechanism. The
    alternative -- pitchers keeping the old "newest game vs target date"
    check -- left the same never-satisfiable staleness bug in place on the
    pitcher side, re-pulling all ~28 starters on every single run.
    """
    path = cache_path(log_key or REFRESH_LOG_KEY)
    if not os.path.exists(path):
        return {}
    try:
        log = pd.read_csv(path)
        return {
            int(r.player_id): pd.Timestamp(r.last_checked).normalize()
            for r in log.itertuples()
            if pd.notna(r.last_checked)
        }
    except Exception:
        return {}


def _save_refresh_log(log: dict, log_key: str = None):
    if not log:
        return
    pd.DataFrame(
        [{"player_id": pid, "last_checked": ts.strftime("%Y-%m-%d")}
         for pid, ts in sorted(log.items())]
    ).to_csv(cache_path(log_key or REFRESH_LOG_KEY), index=False)


def checked_today(entity_id: int, log_key: str, as_of=None,
                  max_age_days: int = 0) -> bool:
    """Has this id been checked recently enough, per its own log?"""
    log = _load_refresh_log(log_key)
    checked = log.get(int(entity_id))
    if checked is None:
        return False
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None \
        else pd.Timestamp.today().normalize()
    return (as_of - checked).days <= max_age_days


def mark_checked(entity_id: int, log_key: str, as_of=None):
    """Record that this id was successfully checked."""
    log = _load_refresh_log(log_key)
    log[int(entity_id)] = pd.Timestamp(as_of).normalize() if as_of is not None \
        else pd.Timestamp.today().normalize()
    _save_refresh_log(log, log_key)


def data_age_days(player_id: int, as_of=None) -> float:
    """
    How many days of GAMES this player's cache is missing.

    Reported for information -- it is NOT the staleness test. A player who
    hasn't appeared in three weeks has a large age and nothing to fetch.
    Use `needs_refresh` to decide whether to pull.
    """
    cached = load_player_cache(player_id)
    if cached.empty:
        return float("inf")
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    return float((as_of - cached["game_date"].max()).days)


def needs_refresh(player_id: int, as_of=None, max_age_days: int = 0,
                  refresh_log: dict = None) -> bool:
    """
    Has this player been checked recently enough?

    `as_of` should be TODAY, not the date being predicted -- checking
    against a future slate date is what made the original version
    re-pull everything forever.
    """
    if load_player_cache(player_id).empty:
        return True
    log = refresh_log if refresh_log is not None else _load_refresh_log()
    checked = log.get(int(player_id))
    if checked is None:
        return True
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None \
        else pd.Timestamp.today().normalize()
    return (as_of - checked).days > max_age_days


def refresh_player(player_id: int, start_date: str, end_date: str,
                   max_age_days: int = 0, verbose: bool = True,
                   refresh_log: dict = None) -> tuple:
    """
    Bring one player's cache current, pulling only what's missing.

    Returns (dataframe, did_check) -- `did_check` is True when a pull
    actually succeeded, so the caller knows whether to stamp the refresh
    log. A failed pull returns False and stays retryable.

    `end_date` is capped at today: asking Statcast for games that haven't
    been played is pointless, and it was the source of the never-satisfied
    staleness loop.
    """
    from pybaseball import statcast_batter

    cached = load_player_cache(player_id)
    today = pd.Timestamp.today().normalize()
    effective_end = min(pd.Timestamp(end_date), today)

    if not needs_refresh(player_id, as_of=today, max_age_days=max_age_days,
                         refresh_log=refresh_log):
        return cached, False

    if cached.empty:
        pull_from = start_date
    else:
        # Overlap by one day so same-day Statcast revisions land.
        pull_from = cached["game_date"].max().strftime("%Y-%m-%d")

    try:
        fresh = statcast_batter(pull_from, effective_end.strftime("%Y-%m-%d"),
                                int(player_id))
    except Exception as e:
        if verbose:
            print(f"      top-up failed for {player_id}: {type(e).__name__} "
                  f"-- keeping cached data, will retry next run")
        return cached, False

    time.sleep(REQUEST_DELAY_SEC)

    if fresh is None or fresh.empty:
        # Checked successfully; there was simply nothing new (didn't play,
        # or Statcast hasn't posted last night's game yet). Still write the
        # canonical file so the legacy-key migration persists.
        if not cached.empty and load_cached(canonical_key(player_id)) is None:
            save_cache(canonical_key(player_id), cached)
        return cached, True

    fresh["game_date"] = pd.to_datetime(fresh["game_date"], errors="coerce")
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    combined = combined.dropna(subset=["game_date"])

    keys = [k for k in DEDUPE_KEYS if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)

    save_cache(canonical_key(player_id), combined)
    if verbose:
        added = len(combined) - len(cached)
        print(f"      +{added:,} new pitches (through "
              f"{combined['game_date'].max().date()})")
    return combined, True


CHECKPOINT_EVERY = 25


def refresh_players(player_ids, start_date: str, end_date: str,
                    max_age_days: int = 0, names=None,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Top up a whole slate's worth of hitters and return their combined
    pitch-level data.

    Only players not already checked today hit the network. Re-running the
    same slate after lineups post is therefore free -- which is the point,
    since that's the run you actually care about.

    Players who fail are skipped rather than aborting the run: on a
    15-game slate one unreachable player should not cost you the other 250.
    """
    frames, failed = [], []
    names = names or {}
    today = pd.Timestamp.today().normalize()

    refresh_log = _load_refresh_log()
    to_check = [p for p in player_ids
                if needs_refresh(p, as_of=today, max_age_days=max_age_days,
                                 refresh_log=refresh_log)]

    if verbose:
        print(f"    {len(to_check)} of {len(player_ids)} need a check today; "
              f"{len(player_ids) - len(to_check)} already current.")

    checked_count = 0
    for i, player_id in enumerate(player_ids, start=1):
        label = names.get(player_id, str(player_id))
        will_check = player_id in set(to_check)

        if verbose and will_check:
            age = data_age_days(player_id, as_of=today)
            state = "no cache" if age == float("inf") else \
                f"last game {age:.0f}d ago"
            print(f"    [{i}/{len(player_ids)}] {label} ({state})")

        try:
            df, did_check = refresh_player(
                player_id, start_date, end_date, max_age_days=max_age_days,
                verbose=verbose, refresh_log=refresh_log,
            )
        except Exception as e:
            print(f"      {label}: {type(e).__name__} -- skipped")
            failed.append(label)
            continue

        if did_check:
            refresh_log[int(player_id)] = today
            checked_count += 1
            if checked_count % CHECKPOINT_EVERY == 0:
                _save_refresh_log(refresh_log)

        if df is not None and not df.empty:
            frames.append(df)
        else:
            failed.append(label)

    _save_refresh_log(refresh_log)

    if failed and verbose:
        print(f"    {len(failed)} player(s) had no usable data: "
              f"{', '.join(map(str, failed[:6]))}"
              f"{'...' if len(failed) > 6 else ''}")

    if not frames:
        raise RuntimeError("No player data available after refresh.")
    return pd.concat(frames, ignore_index=True)
