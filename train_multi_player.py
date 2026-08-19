"""
Same target (HR) and same feature shape as V1 (rolling player form only --
no pitcher/park features this round, so this isolates the effect of more
data/more players from the effect of better features tested in V3).
No class_weight override, consistent with the V3 decision.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from config import PLAYER_FIRST, PLAYER_LAST, START_DATE, END_DATE, PLAYERS, SEASONS, ALL_TEAMS
from data.loader import get_player_id, load_batter_statcast
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi


FEATURE_COLS = [
    "hr_rate_10", "hr_rate_20",
    "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20",
    "career_hr_rate",
]


def main():
    all_statcast = []
    for first, last in PLAYERS:
        print(f"Looking up and pulling {first} {last}...")
        try:
            player_id = get_player_id(first, last)
            df = load_batter_statcast(player_id, START_DATE, END_DATE)
            all_statcast.append(df)
        except Exception as e:
            print(f"  Skipping {first} {last}: {e}")

    print(f"\nCombining data from {len(all_statcast)} players...")
    combined = pd.concat(all_statcast, ignore_index=True)

    print("Building multi-player game log...")
    game_log = build_multi_game_log(combined)

    print("Adding per-player rolling features...")
    features_df = add_rolling_features_multi(game_log)

    print(f"Total games across all players: {len(features_df)}")

    features_df = features_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(features_df) * 0.75)
    train_df, test_df = features_df.iloc[:split_idx], features_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df["hit_hr"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["hit_hr"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("\n=== Multi-Player HR Model Results ===")
    print(f"Train games: {len(train_df)}  |  Test games: {len(test_df)}")
    print(f"Actual HR rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    main()