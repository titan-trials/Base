"""
Real probable starting pitcher lookup, via MLB's own Stats API
(statsapi.mlb.com, wrapped by the MLB-StatsAPI package -- pip install
MLB-StatsAPI). Different source than both Baseball Savant (pybaseball's
Statcast wrapper) and Baseball-Reference (schedule_and_record) -- this is
MLB's own official schedule data, which includes real probable pitchers,
not an approximation.

CONFIRMED against a live test call: statsapi.schedule() returns pitcher
NAMES only (e.g. 'home_probable_pitcher': 'Logan Gilbert') -- there is no
ID field for probable pitchers in this response, unlike team IDs
(home_id/away_id), which ARE present. So names need to be resolved to
MLBAM IDs via data/loader.py's get_player_id() (which already handles
accented names via its fuzzy-match fallback -- useful here too, e.g.
'Andrés Muñoz').
"""
import statsapi
from data.loader import get_player_id

TEAM_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CHW": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136,
    "SF": 137, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
}


def get_team_id(team_abbrev: str) -> int:
    """Look up MLB Stats API's numeric team ID from our project's
    abbreviation. Falls back to statsapi's own lookup_team() if the
    abbreviation isn't in the manual map above."""
    if team_abbrev in TEAM_IDS:
        return TEAM_IDS[team_abbrev]
    results = statsapi.lookup_team(team_abbrev)
    if not results:
        raise ValueError(f"No MLB Stats API team found for '{team_abbrev}'")
    return results[0]["id"]


def resolve_pitcher_id(full_name: str):
    """
    Resolve a pitcher's full name (as returned by statsapi.schedule(), e.g.
    'Logan Gilbert') to their MLBAM ID via get_player_id()'s existing
    lookup (including its accented-name fuzzy fallback).

    Caveat: splits on the FIRST space only (first_name = word 1,
    last_name = everything after). Works for typical two-part names but
    can misfire on suffixes or multi-word surnames (e.g. 'De La Cruz') --
    a known limitation, not silently guessed around.
    """
    if not full_name or " " not in full_name:
        return None
    first, last = full_name.split(" ", 1)
    try:
        return get_player_id(first, last)
    except Exception:
        return None


def get_probable_pitchers(start_date: str, end_date: str, team_abbrev: str = None) -> list:
    """
    Returns a list of dicts, one per game in the date range, each with:
      game_date, home_team_id, away_team_id,
      home_probable_pitcher_name, home_probable_pitcher_id,
      away_probable_pitcher_name, away_probable_pitcher_id
    Dates use MM/DD/YYYY per the MLB-StatsAPI package's convention.
    Pitcher IDs are resolved via resolve_pitcher_id() -- None if the name
    couldn't be matched.
    """
    kwargs = {
        "start_date": start_date,
        "end_date": end_date,
        "hydrate": "probablePitcher",
    }
    if team_abbrev is not None:
        kwargs["team"] = get_team_id(team_abbrev)

    games = statsapi.schedule(**kwargs)

    results = []
    for g in games:
        home_name = g.get("home_probable_pitcher")
        away_name = g.get("away_probable_pitcher")
        results.append({
            "game_date": g.get("game_date"),
            "home_team_id": g.get("home_id"),
            "away_team_id": g.get("away_id"),
            "home_probable_pitcher_name": home_name,
            "home_probable_pitcher_id": resolve_pitcher_id(home_name),
            "away_probable_pitcher_name": away_name,
            "away_probable_pitcher_id": resolve_pitcher_id(away_name),
        })
    return results