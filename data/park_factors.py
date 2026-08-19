"""
Static HR park factor table
Values are approximate multi-year averages (100 = league-average HR rate
for that park; above 100 favors hitters, below 100 favors pitchers).

This is a simplification: real park factors shift year to year and split
by batter handedness. Good enough for now --- worth upgrading
to a season-specific, handedness-split table later if this feature turns
out to earn its keep.
"""

HR_PARK_FACTOR = {
    "ARI": 103, "ATL": 98,  "BAL": 106, "BOS": 94,  "CHC": 101, "CWS": 103,
    "CIN": 112, "CLE": 92,  "COL": 118, "DET": 90,  "HOU": 105, "KC": 88,
    "LAA": 97,  "LAD": 101, "MIA": 89,  "MIL": 100, "MIN": 100, "NYM": 96,
    "NYY": 110, "OAK": 92,  "PHI": 104, "PIT": 88,  "SD": 92,   "SF": 85,
    "SEA": 90,  "STL": 97,  "TB": 95,   "TEX": 100, "TOR": 105, "WSH": 97,
}


def get_park_factor(home_team: str) -> int:
    return HR_PARK_FACTOR.get(home_team, 100)