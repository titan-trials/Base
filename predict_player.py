"""
Live single-prediction tool: given everything the models have learned from
history, what's the estimated probability for a SPECIFIC player's next
game (HR, walk) and their team's next game (win)?

Different purpose than every other script so far: those all report
BACKTEST PERFORMANCE (AUC/accuracy on held-out historical data) to prove
whether an approach works at all. This one assumes that's already settled
and answers the practical question instead -- trains on ALL available
history (no train/test split, since the goal here is one live estimate,
not evaluating accuracy), then predicts using the player/team's most
recent rolling-feature snapshot.

Important honesty check: these are estimates from each model's LEARNED
pattern based on recent form -- they do NOT know who the player or team is
actually playing next (no specific opponent pitcher/team plugged in here).
Given the AUC levels these models actually have (HR ~0.54-0.565, walk
~0.59, team win ~0.582), treat these as rough, real-but-modest estimates,
not confident forecasts.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data.loader import load_batter_statcast_cached, get_player_id
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.team_pipeline import get_full_team_features
from features.team_features import OWN_FEATURE_COLS
from config import PLAYERS, ALL_TEAMS, SEASONS, START_DATE, END_DATE

# --- Who to predict for ---
TARGET_FIRST = "Julio"
TARGET_LAST = "Rodriguez"
TARGET_TEAM = "SEA"

HR_FEATURES = [
    "hr_rate_10", "hr_rate_20", "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20", "career_hr_rate",
]
WALK_FEATURES = [
    "walk_rate_10", "walk_rate_20", "career_walk_rate",
    "chase_rate_10", "chase_rate_20", "career_chase_rate",
]
TEAM_FEATURES = OWN_FEATURE_COLS + [f"opp_{c}" for c in OWN_FEATURE_COLS] + \
    [f"strength_diff_{c}" for c in OWN_FEATURE_COLS]


def get_player_snapshot(target_first: str, target_last: str):
    """Train HR/walk models on the full player pool (config.PLAYERS),
    return predicted probabilities for the target player's most recent
    game state."""
    print("Loading player pool (cached where available)...")
    all_statcast = []
    for first, last in PLAYERS:
        print(f"Loading {first} {last}...")
        try:
            df = load_batter_statcast_cached(first, last, START_DATE, END_DATE)
            all_statcast.append(df)
        except Exception as e:
            print(f"  Skipping {first} {last}: {e}")

    combined = pd.concat(all_statcast, ignore_index=True)
    game_log = build_multi_game_log(combined)
    features_df = add_rolling_features_multi(game_log)

    target_id = get_player_id(target_first, target_last)
    player_rows = features_df[features_df["batter"] == target_id].sort_values("game_date")
    if player_rows.empty:
        raise ValueError(
            f"No feature rows found for {target_first} {target_last} -- "
            f"check they're in config.PLAYERS"
        )
    latest_row = player_rows.iloc[[-1]]

    results = {}
    for label, feature_cols, target_col in [
        ("HR", HR_FEATURES, "hit_hr"),
        ("Walk", WALK_FEATURES, "got_walk"),
    ]:
        model = LogisticRegression(max_iter=1000)
        model.fit(features_df[feature_cols], features_df[target_col])
        results[label] = model.predict_proba(latest_row[feature_cols])[0, 1]

    return latest_row["game_date"].iloc[0], results


def get_team_snapshot(team_abbrev: str):
    """Train the team win model on the full team dataset (cached), return
    the predicted win probability for the team's most recent game state."""
    full_df = get_full_team_features(ALL_TEAMS, SEASONS)
    team_rows = full_df[full_df["Tm"] == team_abbrev].sort_values("game_date")
    if team_rows.empty:
        raise ValueError(f"No rows found for team {team_abbrev}")
    latest_row = team_rows.iloc[[-1]]

    model = LogisticRegression(max_iter=1000)
    model.fit(full_df[TEAM_FEATURES], full_df["win"])
    prob = model.predict_proba(latest_row[TEAM_FEATURES])[0, 1]
    return latest_row["game_date"].iloc[0], prob


def main():
    print(f"=== {TARGET_FIRST} {TARGET_LAST}: current-form snapshot ===")
    snapshot_date, player_probs = get_player_snapshot(TARGET_FIRST, TARGET_LAST)
    print(f"\nBased on rolling form as of his last recorded game ({snapshot_date.date()}):")
    print(f"  Estimated HR probability, next game:   {player_probs['HR']:.1%}")
    print(f"  Estimated walk probability, next game: {player_probs['Walk']:.1%}")

    print(f"\n=== {TARGET_TEAM}: current-form snapshot ===")
    team_date, win_prob = get_team_snapshot(TARGET_TEAM)
    print(f"Based on rolling form as of their last recorded game ({team_date.date()}):")
    print(f"  Estimated win probability, next game: {win_prob:.1%}")

    print("\nReminder: these estimates reflect recent FORM only -- they don't")
    print("know the actual next opponent, pitcher, or park. Treat as rough,")
    print("modest signal (see each model's AUC in CONTEXT.md), not a forecast.")


if __name__ == "__main__":
    main()