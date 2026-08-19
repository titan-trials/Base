"""
Shared configuration for the Baseball Predictor project.

"""

# --- Date range used across all player-level Statcast pulls ---
START_DATE = "2022-01-01"
END_DATE = "2026-08-01"

# --- Single-player scripts: main.py, test_hit_prediction.py, train_v3.py ---
PLAYER_FIRST = "Aaron"
PLAYER_LAST = "Judge"

# --- Multi-player pool: train_multi_player.py, train_bridge.py,
#     train_multi_targets.py, train_pitch_matchup.py ---
PLAYERS = [
    ("Aaron", "Judge"),
    ("Shohei", "Ohtani"),
    ("Kyle", "Schwarber"),
    ("Pete", "Alonso"),
    ("Matt", "Olson"),
    ("Yordan", "Alvarez"),
    ("Juan", "Soto"),
    ("Rafael", "Devers"),
    ("Julio", "Rodriguez"),
]

# --- Team-level scripts: train_team_win.py, train_bridge.py (via
#     features/team_pipeline.py) ---
SEASONS = [2022, 2023, 2024, 2025, 2026]
ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]