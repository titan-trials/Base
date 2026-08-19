"""
V1 model: logistic regression predicting whether a player hits a home run
in a given game, from rolling-window form features.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Single source of truth for feature columns
FEATURE_COLS = [
    "hr_rate_10", "hr_rate_20",
    "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20",
    "career_hr_rate",
]


def train_test_split_by_time(df: pd.DataFrame, test_frac: float = 0.25):
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_hr_model(df: pd.DataFrame):
    train_df, test_df = train_test_split_by_time(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df["hit_hr"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["hit_hr"]

    # class_weight="balanced" matters here: HR games are a minority class
    # (most games, most players don't homer), so an unweighted model would
    # just learn to always predict "no".
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("=== V1 HR Model Results ===")
    print(f"Test games: {len(test_df)}")
    print(f"Actual HR rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))

    return model, test_df.assign(hr_probability=probs)
