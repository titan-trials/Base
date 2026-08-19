"""
Fatigue features -- rest days since last game, and workload over the
trailing week.
"""
import pandas as pd


def add_fatigue_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, per player:
      rest_days        -- calendar days since this player's previous game
      games_last_7_days -- how many games this player played in the
                           trailing 7 days, NOT counting today's game
    """
    df = game_log.copy()

    def _add(group):
        group = group.sort_values("game_date")
        group["rest_days"] = group["game_date"].diff().dt.days

        counts = group.rolling("7D", on="game_date")["game_pk"].count()
        group["games_last_7_days"] = counts.values - 1
        return group

    df = df.groupby("batter", group_keys=False).apply(_add)
    return df