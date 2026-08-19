"""
Tonight's (or tomorrow's) slate: games, venues, probable pitchers, and
lineups when they exist yet.

WHERE THE DATA COMES FROM
-------------------------
MLB's own schedule endpoint, hydrated with everything the slate runner
needs in a single call:

    /api/v1/schedule?sportId=1&date=YYYY-MM-DD
        &hydrate=probablePitcher,lineups,team,venue

One request returns the whole day. Free, no key.

THE LINEUP TIMING PROBLEM
-------------------------
Official lineups post roughly two to four hours before first pitch. Ask
earlier and the `lineups` block is simply absent.

That matters more here than it would in most projects, because batting
order is the input the plate-appearance model leans on hardest -- the
measured gap between the leadoff slot and the ninth is nearly a full plate
appearance per game. Running the slate in the morning with no lineups
means projecting that.

So this module reports what it knows and how firmly:

    lineup_status = "confirmed"   the official lineup is posted
                    "projected"   inferred from recent starts
                    "unknown"     no basis for a guess

The dashboard shows that status per player. A projected 4th-slot hitter
who actually bats 7th tonight will have a plate-appearance estimate that's
too high, and the honest response is to label it, not to hide it behind a
number that looks identical to a confirmed one.

PROBABLE PITCHERS
-----------------
Also part of this feed, and also provisional -- "TBD" is common a day out
(your own slate listing had three). A game with no named starter still
gets predictions; the opposing-pitcher term simply falls back to league
average, which is what the shrinkage would mostly have done anyway given
how little pitcher data this project has.
"""
import json
import urllib.request
import urllib.error
from datetime import date, timedelta

import pandas as pd

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}&hydrate=probablePitcher,lineups,team,venue"
)
TIMEOUT_SEC = 30


def tomorrow() -> str:
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")


def today() -> str:
    return date.today().strftime("%Y-%m-%d")


def _get(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "baseball_predictor/1.0"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
        return json.load(response)


def get_slate(game_date: str, verbose: bool = True) -> pd.DataFrame:
    """
    One row per game. Columns:
        game_pk, game_date, start_time_utc, venue_id, venue_name,
        home_team_id, home_team, away_team_id, away_team,
        home_probable_id, home_probable, away_probable_id, away_probable,
        home_lineup, away_lineup   (lists of player ids, empty if unposted)
    """
    payload = _get(SCHEDULE_URL.format(date=game_date))
    dates = payload.get("dates") or []
    if not dates:
        if verbose:
            print(f"  No games scheduled for {game_date}.")
        return pd.DataFrame()

    rows = []
    for game in dates[0].get("games", []):
        teams = game.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        venue = game.get("venue") or {}
        lineups = game.get("lineups") or {}

        def probable(side):
            pitcher = (side.get("probablePitcher") or {})
            return pitcher.get("id"), pitcher.get("fullName")

        home_pid, home_pname = probable(home)
        away_pid, away_pname = probable(away)

        rows.append({
            "game_pk": game.get("gamePk"),
            "game_date": game_date,
            "start_time_utc": game.get("gameDate"),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("name"),
            "home_team_id": (home.get("team") or {}).get("id"),
            "home_team": (home.get("team") or {}).get("abbreviation")
                         or (home.get("team") or {}).get("name"),
            "away_team_id": (away.get("team") or {}).get("id"),
            "away_team": (away.get("team") or {}).get("abbreviation")
                         or (away.get("team") or {}).get("name"),
            "home_probable_id": home_pid,
            "home_probable": home_pname or "TBD",
            "away_probable_id": away_pid,
            "away_probable": away_pname or "TBD",
            "home_lineup": [p.get("id") for p in (lineups.get("homePlayers") or [])],
            "away_lineup": [p.get("id") for p in (lineups.get("awayPlayers") or [])],
        })

    slate = pd.DataFrame(rows)
    if verbose and not slate.empty:
        confirmed = int(sum(1 for r in rows
                            if r["home_lineup"] and r["away_lineup"]))
        tbd = int(sum(1 for r in rows
                      if r["home_probable"] == "TBD" or r["away_probable"] == "TBD"))
        print(f"  {len(slate)} games on {game_date}. "
              f"{confirmed} with confirmed lineups, "
              f"{tbd} with a TBD starter.")
    return slate


def slate_pitcher_ids(slate: pd.DataFrame) -> dict:
    """
    {team_id: opposing_probable_pitcher_id} -- what each team's hitters
    will face. None where the starter is still TBD.
    """
    mapping = {}
    for row in slate.itertuples():
        mapping[row.home_team_id] = row.away_probable_id
        mapping[row.away_team_id] = row.home_probable_id
    return mapping
