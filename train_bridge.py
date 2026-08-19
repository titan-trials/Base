"""
The bridge: multi-player HR model (Track 1, AUC 0.538 baseline) + team-level
opponent strength (Track 3's winning ingredient) as additional features.

For each player-game, derives which team was the OPPONENT (Statcast doesn't
label this directly -- home_team/away_team + inning_topbot tells us which
team was batting, so the other one was pitching/fielding against our
player), then merges in that team's rolling win_rate/run_diff/ra on that
exact date from the same cached team-features table train_team_win.py uses.

Both the team pull and each player's Statcast pull are cached (see
data/cache.py) -- first run is slow, every run after is fast.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from config import PLAYER_FIRST, PLAYER_LAST, START_DATE, END_DATE, PLAYERS, SEASONS, ALL_TEAMS
from data.loader import load_batter_statcast_cached
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.team_pipeline import get_full_team_features
from features.team_features import OWN_FEATURE_COLS

PLAYER_FEATURE_COLS = [
    "hr_rate_10", "hr_rate_20",
    "avg_ev_10", "avg_ev_20",
    "iso_proxy_10", "iso_proxy_20",
    "career_hr_rate",
]
OPP_FEATURE_COLS = [f"opp_team_{c}" for c in OWN_FEATURE_COLS]


def add_opponent_team(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: which team was the OPPONENT of our batter.
    home_team bats in the bottom half of the inning, away_team in the top --
    whichever team ISN'T batting is the opponent (pitching/fielding)."""
    df = statcast_df.copy()
    df["batter_team"] = df["home_team"].where(df["inning_topbot"] == "Bot", df["away_team"])
    df["opponent_team"] = df["away_team"].where(df["batter_team"] == df["home_team"], df["home_team"])
    per_game = df.groupby("game_pk").agg(
        opponent_team=("opponent_team", "first"),
    ).reset_index()
    return per_game


def main():
    print("=== Step 1: Pull/load team features (cached) ===")
    team_features = get_full_team_features(ALL_TEAMS, SEASONS)
    team_lookup = team_features[["Tm", "game_date"] + OWN_FEATURE_COLS].copy()
    team_lookup = team_lookup.rename(
        columns={"Tm": "opponent_team", **{c: f"opp_team_{c}" for c in OWN_FEATURE_COLS}}
    )

    print("\n=== Step 2: Pull/load player Statcast data (cached) ===")
    all_statcast = []
    for first, last in PLAYERS:
        print(f"Loading {first} {last}...")
        try:
            df = load_batter_statcast_cached(first, last, START_DATE, END_DATE)
            all_statcast.append(df)
        except Exception as e:
            print(f"  Skipping {first} {last}: {e}")

    combined = pd.concat(all_statcast, ignore_index=True)

    print("\n=== Step 3: Build player features ===")
    game_log = build_multi_game_log(combined)
    features_df = add_rolling_features_multi(game_log)

    print("Deriving opponent team per game...")
    opponent_map = add_opponent_team(combined)
    features_df = features_df.merge(opponent_map, on="game_pk", how="left")

    print("Merging in opponent team strength...")
    features_df = features_df.merge(
        team_lookup, on=["opponent_team", "game_date"], how="inner"
    )
    print(f"Total player-games after team match: {len(features_df)}")

    print("\n=== Step 4: Train ===")
    feature_cols = PLAYER_FEATURE_COLS + OPP_FEATURE_COLS
    features_df = features_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(features_df) * 0.75)
    train_df, test_df = features_df.iloc[:split_idx], features_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["hit_hr"]
    X_test, y_test = test_df[feature_cols], test_df["hit_hr"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("\n=== Multi-Player HR Model + Team Opponent Strength ===")
    print(f"Train games: {len(train_df)}  |  Test games: {len(test_df)}")
    print(f"Actual HR rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}  (baseline without team features was 0.538)")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    main()