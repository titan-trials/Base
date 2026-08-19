"""
Adds two new feature types to the multi-player HR baseline (AUC ~0.54):

  1. Pitch-type matchup: for each game, combines this batter's own
     rolling HR rate against fastballs/breaking/offspeed with the
     opposing starter's own season pitch mix, into one "expected HR
     exposure" score. A STYLE-fit feature, not a quality average --
     genuinely different from what V3 and the bridge script tried.
  2. Fatigue: rest days since last game, games played in the trailing
     7 days -- free, from data already pulled.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from config import PLAYER_FIRST, PLAYER_LAST, START_DATE, END_DATE, PLAYERS, SEASONS, ALL_TEAMS
from data.loader import load_batter_statcast_cached
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.fatigue_features import add_fatigue_features
from features.pitch_mix_features import (
    get_starting_pitcher_per_game,
    build_batter_bucket_log,
    add_bucket_rolling_rates,
    build_pitcher_mix_table,
    BUCKETS,
)

BASELINE_FEATURES = [
    "hr_rate_10", "hr_rate_20", "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20", "career_hr_rate",
]
FATIGUE_FEATURES = ["rest_days", "games_last_7_days"]
BUCKET_WINDOW = 20  # which rolling window to use for the matchup score


def find_best_threshold(y_true, probs) -> float:
    """Youden's J statistic: the threshold that maximizes (true positive
    rate - false positive rate) on the ROC curve. Computed on TRAINING
    predictions only -- using test data here would be leakage."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    return thresholds[best_idx]


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

    print("Adding fatigue features...")
    features_df = add_fatigue_features(features_df)

    print("Building batter pitch-type-bucket HR rates...")
    bucket_log = build_batter_bucket_log(combined)
    bucket_rates = add_bucket_rolling_rates(bucket_log)
    bucket_cols = [f"hr_rate_vs_{b}_{BUCKET_WINDOW}" for b in BUCKETS]
    features_df = features_df.merge(
        bucket_rates[["batter", "game_pk", "game_date"] + bucket_cols],
        on=["batter", "game_pk", "game_date"], how="left",
    )

    print("Finding opposing starting pitcher per game...")
    starters = get_starting_pitcher_per_game(combined)
    features_df = features_df.merge(starters, on="game_pk", how="left")

    features_df["mix_season"] = features_df["game_date"].dt.year - 1
    pairs = features_df[["opp_pitcher_id", "mix_season"]].dropna().drop_duplicates()
    pairs = list(pairs.itertuples(index=False, name=None))
    print(f"Pulling pitch mix for {len(pairs)} unique (pitcher, season) pairs "
          f"(cached, but slow on first run)...")
    mix_table = build_pitcher_mix_table(pairs)

    if not mix_table.empty:
        features_df = features_df.merge(
            mix_table, left_on=["opp_pitcher_id", "mix_season"],
            right_on=["opp_pitcher_id", "season"], how="left",
        )
        features_df["expected_hr_exposure"] = (
            features_df[f"hr_rate_vs_fastball_{BUCKET_WINDOW}"] * features_df["fastball_pct"]
            + features_df[f"hr_rate_vs_breaking_{BUCKET_WINDOW}"] * features_df["breaking_pct"]
            + features_df[f"hr_rate_vs_offspeed_{BUCKET_WINDOW}"] * features_df["offspeed_pct"]
        )
        matchup_features = ["expected_hr_exposure"]
    else:
        print("  No pitch mix data came back -- skipping matchup feature.")
        matchup_features = []

    feature_cols = BASELINE_FEATURES + FATIGUE_FEATURES + matchup_features
    features_df = features_df.dropna(subset=feature_cols + ["hit_hr"]).reset_index(drop=True)
    print(f"Total games with all features available: {len(features_df)}")

    features_df = features_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(features_df) * 0.75)
    train_df, test_df = features_df.iloc[:split_idx], features_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["hit_hr"]
    X_test, y_test = test_df[feature_cols], test_df["hit_hr"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    threshold = find_best_threshold(y_train, train_probs)

    test_probs = model.predict_proba(X_test)[:, 1]
    preds = (test_probs >= threshold).astype(int)

    print("\n=== HR Model + Pitch-Type Matchup + Fatigue ===")
    print(f"Train games: {len(train_df)}  |  Test games: {len(test_df)}")
    print(f"Actual HR rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, test_probs):.3f}  (baseline without these features was ~0.54)")
    print(f"Decision threshold (tuned on train, via Youden's J): {threshold:.3f}")
    print(f"Accuracy at tuned threshold: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    main()