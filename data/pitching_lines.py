"""
Official per-pitcher, per-game pitching lines from MLB boxscores.

WHY NOT STATCAST
----------------
Strikeouts and batters faced are both recoverable from a Statcast pull,
and features/pitcher_workload.py does exactly that for the training
history. But grading last night's slate that way means refreshing the
Statcast cache for every starter who threw, and Statcast lags a few hours
behind the last out. The boxscore is final within minutes of the game
ending and is one small request per game.

Same endpoint data/batting_lines.py already uses -- a different section of
the same payload -- so the shape of this file deliberately mirrors that
one. Two fetches of the same URL per scored date is about fifteen extra
HTTP calls, which is cheaper than the refactor that would avoid them and
much cheaper than getting the H+R+RBI path wrong while attempting it.

WHO IS THE STARTER
------------------
`gamesStarted` is 1 for the starter and 0 for everyone else, straight
from the official line. That is better than the inning-1 heuristic
features/pitcher_workload.py has to use on Statcast, which cannot tell a
starter from an opener without also looking at how long the outing was.

CACHE
-----
`cache/pitching_lines.csv`, one row per (game_pk, pitcher_id). Transient
network failures are NOT cached, so a hiccup stays retryable -- same rule
as data/batting_lines.py.
"""
import os
import json
import time
import urllib.request
import urllib.error

import pandas as pd

from data.cache import cache_path

CACHE_KEY = "pitching_lines"
CHECKPOINT_EVERY = 100
REQUEST_DELAY_SEC = 0.12
TIMEOUT_SEC = 20

# A game in progress returns a perfectly valid partial boxscore, and
# nothing in the response says so. Caching one would freeze a starter at
# four strikeouts through five innings as though that were his final line,
# and because the cache is keyed on game_pk it would never be refetched.
#
# Both teams together face roughly 74 batters in a nine-inning game and
# never fewer than about 54. Anything below that is treated as "not
# finished" and left uncached. Same threshold and same reasoning as
# data/batting_lines.MIN_FINAL_PA.
MIN_FINAL_BF = 54

URL = (
    "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    "?fields=teams,home,away,players,person,id,stats,pitching,"
    "strikeOuts,battersFaced,inningsPitched,gamesStarted,"
    "baseOnBalls,hits,earnedRuns,homeRuns,pitchesThrown"
)

LINE_COLUMNS = [
    "game_pk", "pitcher_id", "is_starter", "batters_faced", "strikeouts",
    "innings_pitched", "walks", "hits_allowed", "home_runs_allowed",
    "pitches",
]


def _innings_to_float(value) -> float:
    """
    "6.2" means six innings and TWO OUTS, not six and one-fifth.

    Averaging the raw string as a decimal understates a pitcher by up to
    a fifth of an inning every time, which is small, consistent, and
    invisible.
    """
    try:
        whole, _, outs = str(value).partition(".")
        return float(whole or 0) + float(outs or 0) / 3.0
    except (TypeError, ValueError):
        return 0.0


def _fetch_one(game_pk: int):
    """
    Official pitching lines for one game.

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
            pitching = ((entry.get("stats") or {}).get("pitching")) or {}
            if not person.get("id") or not pitching:
                continue
            faced = pitching.get("battersFaced")
            if not faced:
                # A position player listed with an empty pitching line, or
                # a reliever who warmed up and never entered.
                continue
            rows.append({
                "game_pk": int(game_pk),
                "pitcher_id": int(person["id"]),
                "is_starter": int(pitching.get("gamesStarted") or 0),
                "batters_faced": int(faced),
                "strikeouts": int(pitching.get("strikeOuts") or 0),
                "innings_pitched": _innings_to_float(
                    pitching.get("inningsPitched")),
                "walks": int(pitching.get("baseOnBalls") or 0),
                "hits_allowed": int(pitching.get("hits") or 0),
                "home_runs_allowed": int(pitching.get("homeRuns") or 0),
                "pitches": int(pitching.get("pitchesThrown") or 0),
            })

    if rows and sum(r["batters_faced"] for r in rows) < MIN_FINAL_BF:
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


def get_pitching_lines(game_pks, verbose: bool = True,
                       refetch=None) -> pd.DataFrame:
    """
    One row per (game, pitcher) with the official pitching line.

    `refetch` is an iterable of game_pks to pull again even though they are
    cached, so a line cached while the game was still being played can
    never silently become the number the model is graded against.
    """
    wanted = sorted({int(pk) for pk in pd.Series(list(game_pks)).dropna().unique()})
    cached = _load_existing()
    have = set(cached["game_pk"].dropna().astype(int)) if not cached.empty else set()
    # `is not None` and `len()`, never plain truthiness -- callers pass
    # numpy arrays here and `if some_array:` raises rather than doing
    # anything useful. The same line in data/batting_lines.py was a live
    # bug once.
    if refetch is not None and len(refetch) > 0:
        stale = {int(pk) for pk in pd.Series(list(refetch)).dropna().unique()}
        have -= stale
        cached = cached[~cached["game_pk"].isin(stale)] if not cached.empty else cached
    missing = [pk for pk in wanted if pk not in have]

    if verbose:
        print(f"  Pitching lines: {len(wanted):,} games wanted, "
              f"{len(have & set(wanted)):,} cached, {len(missing):,} to fetch.")

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
            rows = [{c: 0 for c in LINE_COLUMNS}]
            rows[0]["game_pk"] = game_pk
            rows[0]["pitcher_id"] = -1
        new_rows.extend(rows)
        time.sleep(REQUEST_DELAY_SEC)

        if i % CHECKPOINT_EVERY == 0 and new_rows:
            pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True) \
                .to_csv(cache_path(CACHE_KEY), index=False)

    if failures and verbose:
        print(f"    {failures} request(s) failed and were NOT cached -- "
              f"rerun to retry just those.")
    if not new_rows:
        return cached[cached["game_pk"].isin(wanted)].reset_index(drop=True)

    combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_pk", "pitcher_id"],
                                        keep="last")
    combined.to_csv(cache_path(CACHE_KEY), index=False)
    if verbose:
        real = (combined["pitcher_id"] > 0).sum()
        print(f"  Cached {real:,} pitching lines to cache/{CACHE_KEY}.csv")

    result = combined[combined["game_pk"].isin(wanted)]
    return result[result["pitcher_id"] > 0].reset_index(drop=True)
