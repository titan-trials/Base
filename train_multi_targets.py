
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from data.loader import load_batter_statcast_cached
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.prop_targets import add_forward_hr_target
from config import PLAYERS, START_DATE, END_DATE

HR_FEATURES = [
    "hr_rate_10", "hr_rate_20", "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20", "career_hr_rate",
]
WALK_FEATURES = [
    "walk_rate_10", "walk_rate_20", "career_walk_rate",
    "chase_rate_10", "chase_rate_20", "career_chase_rate",
]


def train_and_report(df: pd.DataFrame, target_col: str, feature_cols: list, label: str):
    data = df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(data) * 0.75)
    train_df, test_df = data.iloc[:split_idx], data.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(f"\n=== {label} ===")
    print(f"Train games: {len(train_df)}  |  Test games: {len(test_df)}")
    print(f"Actual positive rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))


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

    print("Building game log and rolling features...")
    game_log = build_multi_game_log(combined)
    features_df = add_rolling_features_multi(game_log)
    print(f"Total games across all players: {len(features_df)}")

    train_and_report(features_df, "hit_hr", HR_FEATURES, "HR This Game (baseline)")

    hr3_df, hr3_col = add_forward_hr_target(features_df, window=3)
    train_and_report(hr3_df, hr3_col, HR_FEATURES, "HR in Next 3 Games (loosened)")

    train_and_report(features_df, "got_walk", WALK_FEATURES, "Walk This Game (with chase rate)")


if __name__ == "__main__":
    main()