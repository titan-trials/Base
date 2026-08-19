import pandas as pd
from pybaseball import schedule_and_record


def load_team_season(team_abbrev: str, season: int) -> pd.DataFrame:
    """One team's full-season schedule and results (one row per game played)."""
    df = schedule_and_record(season, team_abbrev)
    df["Tm"] = team_abbrev
    df["Season"] = season
    return df