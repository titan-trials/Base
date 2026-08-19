"""
Official per-player, per-game batting lines from MLB boxscores.

WHY THIS EXISTS
---------------
The prop people actually bet is HITS + RUNS + RBI. Statcast gives hits
directly and RBI approximately, but runs SCORED are not available in any
usable form: the only trace is free text in the `des` field --

    "Aaron Judge singles on a fly ball to center. Paul Goldschmidt scores."

-- which names the scorer but not his player id. Matching names back to
ids across half a million rows is precisely the accented-name problem this
project already hit once (Yordan Alvarez stored as "álvarez"), and getting
it subtly wrong would be invisible.

MLB's boxscore reports the official line directly. Same endpoint already
used for lineup slots, just more fields.

IT ALSO FIXES RBI
-----------------
features/pa_table.py derives RBI from `post_bat_score - bat_score`, which
is close but not the official statistic: it credits a run that scored on
an error or a wild pitch during the plate appearance, which the official
scorer would not. The boxscore RBI is the real number. Where both are
available, prefer this one.

CACHE
-----
`cache/batting_lines.csv`, one row per (game_pk, player_id), fetched once
per game and checkpointed every 100. Transient network failures are NOT
cached, so a hiccup stays retryable -- same rule as data/lineup_slots.py
and data/refresh.py.
"""
import os
import json
import time
import urllib.request
import urllib.error

import pandas as pd

from data.cache import cache_path

CACHE_KEY = "batting_lines"
CHECKPOINT_EVERY = 100
REQUEST_DELAY_SEC = 0.12
TIMEOUT_SEC = 20

# A game in progress returns a PERFECTLY VALID partial boxscore. Nothing in
# the response says "this is only five innings so far", so caching it would
# freeze half a game in as though it were the final line -- and because the
# cache is keyed on game_pk alone, it would never be refetched. Every hitter
# in that game would then be scored against a total that stopped early.
#
# Both teams together take roughly 74 plate appearances in a nine-inning
# game and never fewer than about 54 (a home team that never bats in the
# ninth, with no baserunners at all). Anything under this threshold is
# treated as "not finished yet" and is not cached.
#
# The cost of the guard is that a genuinely shortened game -- called for
# rain in the sixth, and therefore official -- gets refetched on every run
# instead of being remembered. That is a handful of games a season and one
# extra HTTP call each, which is the cheaper mistake by a wide margin.
MIN_FINAL_PA = 54

# Field-filtered so each response is small. `batting` carries the whole
# line; we keep the pieces the model uses.
URL = (
    "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    "?fields=teams,home,away,players,person,id,stats,batting,"
    "hits,runs,rbi,baseOnBalls,hitByPitch,strikeOuts,homeRuns,"
    "plateAppearances,atBats"
)

LINE_COLUMNS = [
    "game_pk", "player_id", "pa", "at_bats", "hits", "runs", "rbi",
    "walks", "hbp", "strikeouts", "home_runs",
]


def _fetch_one(game_pk: int):
    """
    Official batting lines for one game.

    Returns a list of rows on success (possibly empty), or None if the
    REQUEST failed -- the caller must not cache a network failure as
    "this game has no data".
    """
    try:
        request = urllib.request.Request(
            URL.format(game_pk=int(game_pk)),
            headers={"User-Agent": "baseball_predictor/1.0"},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None

    rows = []
    for side in ("home", "away"):
        team = (payload.get("teams") or {}).get(side) or {}
        for entry in (team.get("players") or {}).values():
            person = entry.get("person") or {}
            batting = ((entry.get("stats") or {}).get("batting")) or {}
            if not person.get("id") or not batting:
                continue
            pa = batting.get("plateAppearances")
            # A player with no plate appearances is a defensive
            # replacement or an unused pitcher -- not a batting line.
            if not pa:
                continue
            rows.append({
                "game_pk": int(game_pk),
                "player_id": int(person["id"]),
                "pa": int(pa),
                "at_bats": int(batting.get("atBats") or 0),
                "hits": int(batting.get("hits") or 0),
                "runs": int(batting.get("runs") or 0),
                "rbi": int(batting.get("rbi") or 0),
                "walks": int(batting.get("baseOnBalls") or 0),
                "hbp": int(batting.get("hitByPitch") or 0),
                "strikeouts": int(batting.get("strikeOuts") or 0),
                "home_runs": int(batting.get("homeRuns") or 0),
            })

    # Too few plate appearances to be a completed game -- almost always a
    # game still being played. Report it as a REQUEST failure so the caller
    # leaves it uncached and retryable, rather than as "this game has no
    # batting lines", which would be remembered forever.
    if rows and sum(r["pa"] for r in rows) < MIN_FINAL_PA:
        return None
    return rows


def _load_existing() -> pd.DataFrame:
    path = cache_path(CACHE_KEY)
    if not os.path.exists(path):
        return pd.DataFrame(columns=LINE_COLUMNS)
    try:
        return pd.read_csv(path)[LINE_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=LINE_COLUMNS)


def get_batting_lines(game_pks, verbose: bool = True,
                      refetch=None) -> pd.DataFrame:
    """
    One row per (game, player) with the official batting line.

    Only games not already cached are fetched. Checkpointed every 100, so
    an interrupted run resumes rather than restarting.

    `refetch` is an iterable of game_pks to pull again even though they are
    cached. score_slate.py passes the date it is scoring, so a line that
    was cached while the game was still being played can never silently
    become the number the model is graded against.
    """
    wanted = sorted({int(pk) for pk in pd.Series(list(game_pks)).dropna().unique()})
    cached = _load_existing()
    have = set(cached["game_pk"].dropna().astype(int)) if not cached.empty else set()
    if refetch:
        stale = {int(pk) for pk in pd.Series(list(refetch)).dropna().unique()}
        have -= stale
        cached = cached[~cached["game_pk"].isin(stale)] if not cached.empty else cached
    missing = [pk for pk in wanted if pk not in have]

    if verbose:
        print(f"  Batting lines: {len(wanted):,} games wanted, "
              f"{len(have & set(wanted)):,} cached, {len(missing):,} to fetch.")
        if missing:
            print(f"    ~{len(missing) * REQUEST_DELAY_SEC / 60:.0f} min. "
                  f"Checkpointed every {CHECKPOINT_EVERY} -- safe to stop and resume.")

    if not missing:
        return cached[cached["game_pk"].isin(wanted)].reset_index(drop=True)

    new_rows, failures = [], 0
    for i, game_pk in enumerate(missing, start=1):
        rows = _fetch_one(game_pk)

        if rows is None:
            failures += 1
            if failures >= 20 and failures == i:
                print(f"    Aborting: the first {i} requests all failed. "
                      f"Check network access to statsapi.mlb.com.")
                break
            continue

        if not rows:
            # Genuinely no batting lines (postponed game). Remember it as a
            # sentinel so it isn't refetched forever.
            rows = [{c: 0 for c in LINE_COLUMNS}]
            rows[0]["game_pk"] = game_pk
            rows[0]["player_id"] = -1
        new_rows.extend(rows)
        time.sleep(REQUEST_DELAY_SEC)

        if i % CHECKPOINT_EVERY == 0 and new_rows:
            pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True) \
                .to_csv(cache_path(CACHE_KEY), index=False)
            if verbose:
                print(f"    ...{i:,}/{len(missing):,} games fetched (checkpointed)")

    if failures and verbose:
        print(f"    {failures} request(s) failed and were NOT cached -- "
              f"rerun to retry just those.")
    if not new_rows:
        return cached[cached["game_pk"].isin(wanted)].reset_index(drop=True)

    combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_pk", "player_id"], keep="last")
    combined.to_csv(cache_path(CACHE_KEY), index=False)
    if verbose:
        real = (combined["player_id"] > 0).sum()
        print(f"  Cached {real:,} batting lines to cache/{CACHE_KEY}.csv")

    result = combined[combined["game_pk"].isin(wanted)]
    return result[result["player_id"] > 0].reset_index(drop=True)


# NOTE: joining these lines onto a per-(batter, game) frame lives in
# features/run_features.build_run_training_frame, not here. There used to
# be an attach_batting_lines() in this file doing roughly the same thing
# with a slightly different set of renames; two functions computing the
# same composite from the same source is how the two quietly drift apart.
# One of them owns it, and it is the one that also derives the times-on-base
# denominator the run model needs.
