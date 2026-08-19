"""
Team-level feature engineering for win probability -- now with opponent
strength, not just a team's own recent form.

Adds real date parsing this time (needed to match a team's game to the
SAME game from its opponent's schedule page, which the earlier
game_number-based split couldn't do -- that only works within one team's
own chronological order, not across two different teams).
"""
import pandas as pd


def parse_schedule_dates(df: pd.DataFrame, season: int):
    
    raw = df["Date"].astype(str)
    dh_game = raw.str.extract(r"\((\d)\)")[0].fillna("1")
    cleaned = raw.str.replace(r"\s*\(\d\)\s*$", "", regex=True)
    cleaned = cleaned.str.split(",").str[-1].str.strip()
    parsed = pd.to_datetime(cleaned + f" {season}", format="%b %d %Y", errors="coerce")
    return parsed, dh_game


def build_team_game_log(schedule_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Reduce one team-season's schedule to a clean per-game log with a
    binary win target and a real parsed date for cross-team matching."""
    df = schedule_df.copy()

    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df = df.dropna(subset=["R", "RA", "W/L"]).reset_index(drop=True)

    df["win"] = df["W/L"].astype(str).str.startswith("W").astype(int)
    df["run_diff"] = df["R"] - df["RA"]
    df["game_number"] = df.index + 1

    df["game_date"], df["dh_game"] = parse_schedule_dates(df, season)
    return df


def add_team_rolling_features(game_log: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    """Rolling team form, shifted by 1 game so today's features never see
    today's own result -- same lookahead guard as the player-level models."""
    df = game_log.copy()
    for w in windows:
        df[f"win_rate_{w}"] = df["win"].rolling(w, min_periods=3).mean().shift(1)
        df[f"run_diff_{w}"] = df["run_diff"].rolling(w, min_periods=3).mean().shift(1)
       
        df[f"ra_{w}"] = df["RA"].rolling(w, min_periods=3).mean().shift(1)
    df["season_win_rate"] = df["win"].expanding(min_periods=5).mean().shift(1)

   
    required_cols = [
        "win", "win_rate_10", "win_rate_20",
        "run_diff_10", "run_diff_20", "ra_10", "ra_20",
        "season_win_rate", "game_date",
    ]
    return df.dropna(subset=required_cols).reset_index(drop=True)


OWN_FEATURE_COLS = [
    "win_rate_10", "win_rate_20",
    "run_diff_10", "run_diff_20",
    "ra_10", "ra_20",
    "season_win_rate",
]


def attach_opponent_features(all_teams_df: pd.DataFrame) -> pd.DataFrame:
    
    opp_lookup = all_teams_df[["Tm", "Season", "game_date", "dh_game"] + OWN_FEATURE_COLS].copy()
    opp_lookup = opp_lookup.rename(
        columns={"Tm": "Opp", **{c: f"opp_{c}" for c in OWN_FEATURE_COLS}}
    )

    merged = all_teams_df.merge(
        opp_lookup, on=["Opp", "Season", "game_date", "dh_game"], how="inner"
    )

    for c in OWN_FEATURE_COLS:
        merged[f"strength_diff_{c}"] = merged[c] - merged[f"opp_{c}"]

    return merged