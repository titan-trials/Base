"""
Final scores, one row per team per game.

WHY A SEPARATE FETCHER
----------------------
The team run model needs the distribution of runs a team scores in a
game, and that turns out to be surprisingly hard to get from what this
project already caches:

  - The PA table only holds the hitters in the pool. Summing their runs
    gives a team's runs only when the pool happens to cover that whole
    lineup, which historically it does not -- checked on 19,458
    game-teams and ZERO had a near-complete side.
  - cache/batting_lines.csv has runs per player but no team column, so
    the totals cannot be assembled from it either.

Boxscores would work but cost one request per game, and there are eleven
thousand of them. The schedule endpoint returns every game in a DATE
RANGE in a single request, with final scores attached. A whole season is
one call.

    python -m data.team_lines 2025-01-01 2026-12-31

REGULAR SEASON ONLY, for the same reason everything else here is: spring
training scores are a different sport. `gameType=R` asks the API to
filter rather than doing it afterwards.
"""
import os
import json
import urllib.request
import urllib.error

import pandas as pd

from data.cache import cache_path

CACHE_KEY = "team_lines"
TIMEOUT_SEC = 30

URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&gameType=R&startDate={start}&endDate={end}"
    "&fields=dates,date,games,gamePk,teams,away,home,team,name,score,"
    "status,detailedState"
)

COLUMNS = ["game_pk", "game_date", "team", "opponent", "is_home",
           "runs", "runs_allowed"]

# A game that was suspended, postponed or called early is not a normal
# nine innings and would drag the run distribution down. Only finished
# games count.
FINAL_STATES = {"Final", "Completed Early", "Game Over"}


def _get(url: str):
    try:
        with urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "baseball_predictor/1.0"}),
                timeout=TIMEOUT_SEC) as response:
            return json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError,
            OSError) as exc:
        print(f"  Request failed: {exc}")
        return None


def fetch_team_lines(start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    """
    Two rows per game -- one per team -- with runs scored and allowed.

    Both sides are emitted because the model needs "how many runs does
    this team score" and "how many does this team allow", and deriving
    the second from the first later means a join that can go wrong.
    """
    if verbose:
        print(f"  Fetching final scores {start} to {end}...")
    payload = _get(URL.format(start=start, end=end))
    if not payload:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    skipped = 0
    for day in payload.get("dates") or []:
        date = day.get("date")
        for game in day.get("games") or []:
            state = ((game.get("status") or {}).get("detailedState") or "")
            if state not in FINAL_STATES:
                skipped += 1
                continue
            teams = game.get("teams") or {}
            home, away = teams.get("home") or {}, teams.get("away") or {}
            h_score, a_score = home.get("score"), away.get("score")
            if h_score is None or a_score is None:
                skipped += 1
                continue
            h_name = ((home.get("team") or {}).get("name"))
            a_name = ((away.get("team") or {}).get("name"))
            pk = game.get("gamePk")
            rows.append({"game_pk": pk, "game_date": date, "team": h_name,
                         "opponent": a_name, "is_home": 1,
                         "runs": int(h_score), "runs_allowed": int(a_score)})
            rows.append({"game_pk": pk, "game_date": date, "team": a_name,
                         "opponent": h_name, "is_home": 0,
                         "runs": int(a_score), "runs_allowed": int(h_score)})

    out = pd.DataFrame(rows, columns=COLUMNS)
    if verbose:
        print(f"  {len(out) // 2:,} completed games "
              f"({skipped} skipped as not final).")
    return out


def get_team_lines(start: str, end: str, refresh: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """Cached. Pass refresh=True to re-pull."""
    path = cache_path(CACHE_KEY)
    if os.path.exists(path) and not refresh:
        try:
            cached = pd.read_csv(path)
            if not cached.empty:
                if verbose:
                    print(f"  Team lines: {len(cached) // 2:,} games cached.")
                return cached
        except Exception:
            pass
    out = fetch_team_lines(start, end, verbose=verbose)
    if not out.empty:
        out.to_csv(path, index=False)
        if verbose:
            print(f"  Cached to cache/{CACHE_KEY}.csv")
    return out


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-12-31"
    df = get_team_lines(start, end, refresh=True)
    if not df.empty:
        r = df["runs"]
        print(f"\n  TEAM RUNS: mean {r.mean():.3f}  var {r.var():.3f}  "
              f"var/mean {r.var() / r.mean():.3f}")
        print(f"  home {df[df.is_home == 1]['runs'].mean():.3f} vs "
              f"away {df[df.is_home == 0]['runs'].mean():.3f}")
        print("  distribution:",
              {int(k): round(v, 4)
               for k, v in r.value_counts(normalize=True).sort_index()
               .head(13).items()})
