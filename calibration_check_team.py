"""
Calibration check for the team win probability model. Answers: when this
model says "65% chance to win," does that team actually win about 65% of
the time across many such games? Different question than AUC (which only
measures ranking ability, not whether the probability number itself is
trustworthy at face value).
"""
from sklearn.linear_model import LogisticRegression

from features.team_pipeline import get_full_team_features
from features.team_features import OWN_FEATURE_COLS
from model.calibration import print_calibration_report
from config import ALL_TEAMS, SEASONS

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
    print_calibration_report(y_test, probs, n_bins=10, label="Team Win Probability")


if __name__ == "__main__":
    main()