"""
Static HR park factor table
Values are approximate multi-year averages (100 = league-average HR rate
for that park; above 100 favors hitters, below 100 favors pitchers).

This is a simplification: real park factors shift year to year and split
by batter handedness. Good enough for now --- worth upgrading
to a season-specific, handedness-split table later if this feature turns
out to earn its keep. Baseball Savant publishes per-season, per-handedness,
per-stat factors as one CSV (statcast-park-factors leaderboard); swapping
this table for that one is the existing feature done properly, not a new
feature.

ATHLETICS (fixed 2026-09-05): the club plays at Sutter Health Park in West
Sacramento for 2025-2027, a hitter's park, and is keyed "ATH" by MLB's
feed. Until this fix predict_slate aliased ATH -> OAK and this table
answered 92 -- the Coliseum, which they no longer play in. Both keys are
carried so an old cache row spelled OAK still resolves, but OAK's number
is now only right for games before 2025.
"""

HR_PARK_FACTOR = {
    "ARI": 103, "ATL": 98,  "BAL": 106, "BOS": 94,  "CHC": 101, "CWS": 103,
    "CIN": 112, "CLE": 92,  "COL": 118, "DET": 90,  "HOU": 105, "KC": 88,
    "LAA": 97,  "LAD": 101, "MIA": 89,  "MIL": 100, "MIN": 100, "NYM": 96,
    "NYY": 110, "PHI": 104, "PIT": 88,  "SD": 92,   "SF": 85,
    "SEA": 90,  "STL": 97,  "TB": 95,   "TEX": 100, "TOR": 105, "WSH": 97,
    # Sutter Health Park, 2025-27. Savant's 2025 single-season HR factor
    # for the park ran well above average; 108 is a deliberately
    # conservative multi-year-style figure rather than one hot season.
    "ATH": 108,
    # Oakland Coliseum -- historical only (games before 2025).
    "OAK": 92,
}

# The schedule feed and Statcast spell a few clubs differently from
# Baseball-Reference. Map every spelling to the table's key.
ALIASES = {
    "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
    "TBR": "TB", "WSN": "WSH", "AZ": "ARI",
}


def get_park_factor(home_team: str) -> int:
    if not isinstance(home_team, str):
        return 100
    key = ALIASES.get(home_team.upper(), home_team.upper())
    return HR_PARK_FACTOR.get(key, 100)
