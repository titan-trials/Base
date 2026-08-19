"""
Multi-player version of features/build_features.py -- same logic, but
grouped by batter so rolling windows never leak across different players.

Adds chase rate as a plate-discipline feature -- a more direct measure of
"how selective is this hitter" than rolling walk rate alone, since it's
built from every pitch's swing/take decision instead of only full
plate-appearance outcomes.
"""
import pandas as pd

# Pitches where the batter swung -- used to compute chase rate. Not an
# exhaustive Statcast taxonomy, but covers the common swing outcomes.
SWING_DESCRIPTIONS = {
    "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
}


def build_multi_game_log(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Same as build_features.build_game_log, but keyed by (batter, game_pk, game_date)
    so multiple players' data can live in one combined dataframe."""
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["batter", "game_date"])

    # Chase rate inputs: 'zone' 1-9 is inside the strike zone, 11-14 are the
    # four regions outside it (Statcast's own coding) -- >9 is a reliable
    # in/out-of-zone split. A "chase" is a swing at a pitch outside the zone.
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_out_of_zone"] = df["zone"] > 9
    df["is_chase"] = df["is_swing"] & df["is_out_of_zone"]

    grouped = df.groupby(["batter", "game_pk", "game_date"]).agg(
        hr=("events", lambda s: int((s == "home_run").sum())),
        hits=("events", lambda s: int(s.isin(["single", "double", "triple", "home_run"]).sum())),
        walks=("events", lambda s: int((s == "walk").sum())),
        strikeouts=("events", lambda s: int((s == "strikeout").sum())),
        pa=("events", lambda s: int(s.notna().sum())),
        avg_exit_velo=("launch_speed", "mean"),
        avg_launch_angle=("launch_angle", "mean"),
        max_exit_velo=("launch_speed", "max"),
        out_of_zone_pitches=("is_out_of_zone", "sum"),
        chases=("is_chase", "sum"),
    ).reset_index()

    grouped["hit_hr"] = (grouped["hr"] > 0).astype(int)
    grouped["got_walk"] = (grouped["walks"] > 0).astype(int)
    # NaN when a game had zero out-of-zone pitches (very short appearance) --
    # cleaned up by the rolling step's dropna() later.
    grouped["chase_rate"] = grouped["chases"] / grouped["out_of_zone_pitches"]
    return grouped.sort_values(["batter", "game_date"]).reset_index(drop=True)


def add_rolling_features_multi(game_log: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    """Same rolling-window logic as add_rolling_features, applied per-player
    via groupby so one player's history never bleeds into another's rolling
    average. Every rolling stat still shifted by 1 game (no lookahead)."""
    df = game_log.copy()

    def _add_player_features(group):
        for w in windows:
            group[f"hr_rate_{w}"] = group["hit_hr"].rolling(w, min_periods=3).mean().shift(1)
            group[f"walk_rate_{w}"] = group["got_walk"].rolling(w, min_periods=3).mean().shift(1)
            group[f"chase_rate_{w}"] = group["chase_rate"].rolling(w, min_periods=3).mean().shift(1)
            group[f"avg_ev_{w}"] = group["avg_exit_velo"].rolling(w, min_periods=3).mean().shift(1)
            # Rolling plate-appearances-per-game -- a proxy for batting order
            # slot (not available directly in Statcast). A leadoff/2-hole
            # hitter averages more PA/game than someone batting 7th-9th,
            # meaning more chances to walk or homer in the same game.
            group[f"pa_{w}"] = group["pa"].rolling(w, min_periods=3).mean().shift(1)
            group[f"iso_proxy_{w}"] = (
                (group["hits"] - group["hr"]).rolling(w, min_periods=3).mean().shift(1)
            )
        group["career_hr_rate"] = group["hit_hr"].expanding(min_periods=5).mean().shift(1)
        group["career_walk_rate"] = group["got_walk"].expanding(min_periods=5).mean().shift(1)
        group["career_chase_rate"] = group["chase_rate"].expanding(min_periods=5).mean().shift(1)
        group["career_pa"] = group["pa"].expanding(min_periods=5).mean().shift(1)
        return group

    df = df.groupby("batter", group_keys=False).apply(_add_player_features)
    return df.dropna().reset_index(drop=True)