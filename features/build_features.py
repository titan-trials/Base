"""
Feature engineering: turns pitch-level Statcast data into a game-level
dataset with a binary "hit a HR this game" target and rolling-window
form features.
"""
import pandas as pd


def build_game_log(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level Statcast rows into one row per game."""
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")

    grouped = df.groupby(["game_pk", "game_date"]).agg(
        hr=("events", lambda s: int((s == "home_run").sum())),
        hits=("events", lambda s: int(s.isin(["single", "double", "triple", "home_run"]).sum())),
        walks=("events", lambda s: int((s == "walk").sum())),
        strikeouts=("events", lambda s: int((s == "strikeout").sum())),
        pa=("events", lambda s: int(s.notna().sum())),
        avg_exit_velo=("launch_speed", "mean"),
        avg_launch_angle=("launch_angle", "mean"),
        max_exit_velo=("launch_speed", "max"),
    ).reset_index()

    grouped["hit_hr"] = (grouped["hr"] > 0).astype(int)
    return grouped.sort_values("game_date").reset_index(drop=True)


def add_rolling_features(game_log: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    """
    Add rolling-window form features. Every rolling stat is shifted by 1 game
    so a game's features never include that game's own outcome — same
    lookahead-bias guard as Quantara's Position = Signal.shift(1).
    """
    df = game_log.copy()
    for w in windows:
        df[f"hr_rate_{w}"] = df["hit_hr"].rolling(w, min_periods=3).mean().shift(1)
        df[f"avg_ev_{w}"] = df["avg_exit_velo"].rolling(w, min_periods=3).mean().shift(1)
        df[f"iso_proxy_{w}"] = (
            (df["hits"] - df["hr"]).rolling(w, min_periods=3).mean().shift(1)
        )
    df["career_hr_rate"] = df["hit_hr"].expanding(min_periods=5).mean().shift(1)
    return df.dropna().reset_index(drop=True)
