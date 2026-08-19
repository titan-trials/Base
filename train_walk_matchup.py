"""
Tests the style-fit idea (the thing that moved HR's needle, 0.541 -> 0.565)
on walk prediction: does adding this specific opposing starter's own
control profile (zone rate, BB rate allowed) improve on the walk baseline
(AUC ~0.59-0.62, our best result so far)?

Starter identification reuses the same "first pitch of the game"
approximation as train_pitch_matchup.py (features/pitch_mix_features.py's
get_starting_pitcher_per_game) -- proven to work for historical data.
Swapping in the real MLB-StatsAPI-based starter resolver is a separate,
later upgrade, not required to test this idea.

Heads up: pulls each unique opposing starter's own season data AGAIN
under a new cache key (pitchcontrol_*) -- see walk_matchup_features.py's
docstring for why. Expect this to be slow on a first run even though
train_pitch_matchup.py already pulled overlapping pitchers.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from data.loader import load_batter_statcast_cached
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.pitch_mix_features import get_starting_pitcher_per_game
from features.walk_matchup_features import build_pitcher_control_table, attach_walk_matchup_features, CONTROL_FEATURE_COLS
from config import PLAYERS, START_DATE, END_DATE

BASELINE_WALK_FEATURES = [
    "walk_rate_10", "walk_rate_20", "career_walk_rate",
    "chase_rate_10", "chase_rate_20", "career_chase_rate",
]


def main():
    print("Loading player data (cached where available)...")
    all_statcast = []
    for first, last in PLAYERS:
        print(f"Loading {first} {last}...")
        try:
            df = load_batter_statcast_cached(first, last, START_DATE, END_DATE)
            all_statcast.append(df)
        except Exception as e:
            print(f"  Skipping {first} {last}: {e}")

    combined = pd.concat(all_statcast, ignore_index=True)

    print("Building baseline game log and rolling features...")
    game_log = build_multi_game_log(combined)
    features_df = add_rolling_features_multi(game_log)

    print("Finding opposing starting pitcher per game...")
    starters = get_starting_pitcher_per_game(combined)
    features_df = features_df.merge(starters, on="game_pk", how="left")

    features_df["mix_season"] = features_df["game_date"].dt.year - 1
    pairs = features_df[["opp_pitcher_id", "mix_season"]].dropna().drop_duplicates()
    pairs = list(pairs.itertuples(index=False, name=None))
    print(f"Pulling control profiles for {len(pairs)} unique (pitcher, season) pairs "
          f"(cached, but slow on first run)...")
    control_table = build_pitcher_control_table(pairs)

    if control_table.empty:
        print("  No pitcher control data came back -- aborting matchup test.")
        return

    features_df = attach_walk_matchup_features(features_df, control_table)

    matchup_features = CONTROL_FEATURE_COLS + ["expected_walk_exposure"]
    all_features = BASELINE_WALK_FEATURES + matchup_features

    features_df = features_df.dropna(subset=all_features + ["got_walk"]).reset_index(drop=True)
    print(f"Total games with all features available: {len(features_df)}")

    features_df = features_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(features_df) * 0.75)
    train_df, test_df = features_df.iloc[:split_idx], features_df.iloc[split_idx:]

    X_train_base, y_train = train_df[BASELINE_WALK_FEATURES], train_df["got_walk"]
    X_test_base, y_test = test_df[BASELINE_WALK_FEATURES], test_df["got_walk"]
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_base, y_train)
    baseline_probs = baseline_model.predict_proba(X_test_base)[:, 1]

    X_train_full = train_df[all_features]
    X_test_full = test_df[all_features]
    full_model = LogisticRegression(max_iter=1000)
    full_model.fit(X_train_full, y_train)
    full_probs = full_model.predict_proba(X_test_full)[:, 1]
    full_preds = full_model.predict(X_test_full)

    print(f"\nBaseline AUC (no matchup features), same split: {roc_auc_score(y_test, baseline_probs):.3f}")

    print("\n=== Walk Model + Style-Fit Pitcher Control Matchup ===")
    print(f"Train games: {len(train_df)}  |  Test games: {len(test_df)}")
    print(f"Actual walk rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, full_probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, full_preds):.3f}")
    print(classification_report(y_test, full_preds, zero_division=0))


if __name__ == "__main__":
    main()