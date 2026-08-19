"""
Team win probability, V2: adds opponent strength (see features/team_features.py)
and a runs-allowed pitching-quality proxy, not just a team's own recent
form. The V1 team model (own form only) came back at AUC ~0.50 --
essentially a coin flip.

Now pulls through features/team_pipeline.py, which caches the full 30-team
result to cache/team_features_full.csv after the first run -- subsequent
runs load instantly instead of re-doing the ~90-request pull. Delete that
cache file (or pass force_refresh=True) if you want fresh data.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from config import PLAYER_FIRST, PLAYER_LAST, START_DATE, END_DATE, PLAYERS, SEASONS, ALL_TEAMS
from features.team_pipeline import get_full_team_features
from features.team_features import OWN_FEATURE_COLS


FEATURE_COLS = OWN_FEATURE_COLS + [f"opp_{c}" for c in OWN_FEATURE_COLS] + \
    [f"strength_diff_{c}" for c in OWN_FEATURE_COLS]


def main():
    full_df = get_full_team_features(ALL_TEAMS, SEASONS)

    full_df = full_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(full_df) * 0.75)
    train_df, test_df = full_df.iloc[:split_idx], full_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df["win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["win"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("\n=== Team Win Probability Model Results ===")
    print(f"Train games: {len(train_df)}  |  Test games: {len(test_df)}")
    print(f"Actual win rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    main()