"""
Opposing pitcher season stats -- computed directly from Statcast pitch-level
data via pybaseball's Baseball Savant wrapper (NOT FanGraphs).
"""
import pandas as pd
from pybaseball import statcast_pitcher

PITCHER_FEATURE_COLS = ["k_rate", "bb_rate", "hr_rate", "avg_ev_allowed"]


def load_pitcher_season_stats(pitcher_id: int, season: int):
    """Pull one pitcher's full-season Statcast data and reduce it to rate stats."""
    start_dt = f"{season}-01-01"
    end_dt = f"{season}-12-31"
    df = statcast_pitcher(start_dt, end_dt, pitcher_id)
    if df.empty:
        return None

    pa = df["events"].notna().sum()
    if pa == 0:
        return None

    strikeouts = (df["events"] == "strikeout").sum()
    walks = (df["events"] == "walk").sum()
    home_runs = (df["events"] == "home_run").sum()

    return {
        "pitcher_id": pitcher_id,
        "k_rate": strikeouts / pa,
        "bb_rate": walks / pa,
        "hr_rate": home_runs / pa,
        "avg_ev_allowed": df["launch_speed"].mean(),
    }


def build_season_pitcher_table(pitcher_ids: list, season: int) -> pd.DataFrame:
    """Build a season stats table for a list of opposing pitchers."""
    rows = []
    for pid in pitcher_ids:
        stats = load_pitcher_season_stats(pid, season)
        if stats is not None:
            rows.append(stats)
    return pd.DataFrame(rows)