"""
V3: adds opposing starting-pitcher season stats (computed from Statcast,
not FanGraphs -- see data/pitching_data.py) and park factor to the V1
batter rolling-form features, predicting HR -- same target as the original
V1 test, so results are directly comparable.
"""
from config import PLAYER_FIRST, PLAYER_LAST, START_DATE, END_DATE, PLAYERS, SEASONS, ALL_TEAMS
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from data.loader import get_player_id, load_batter_statcast
from data.pitching_data import build_season_pitcher_table, PITCHER_FEATURE_COLS
from data.park_factors import get_park_factor
from features.build_features import build_game_log, add_rolling_features



def get_starting_pitchers_and_park(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game: the opposing starter's MLBAM id and the home team.
    'pitcher' in Statcast batter data is already the opposing pitcher (this
    data is batter-centric), and park is always wherever the home team
    plays, regardless of whether our batter's team is home or away.
    """
    df = statcast_df.copy()
    sort_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in df.columns]
    df = df.sort_values(sort_cols)
    per_game = df.groupby("game_pk").agg(
        opp_starter_id=("pitcher", "first"),
        home_team=("home_team", "first"),
    ).reset_index()
    return per_game


def attach_pitcher_and_park_features(features_df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    df = features_df.merge(game_meta, on="game_pk", how="left")
    df["park_factor"] = df["home_team"].apply(get_park_factor)

    # Prior-season pitcher stats only -- using the SAME season would leak
    # information from starts happening after our game (lookahead).
    df["stats_season"] = df["game_date"].dt.year - 1

    rows = []
    for season, group in df.groupby("stats_season"):
        pitcher_ids = group["opp_starter_id"].dropna().unique().tolist()
        print(f"  Pulling {len(pitcher_ids)} pitchers' {season} Statcast season stats "
              f"(one call per pitcher -- this can take a few minutes)...")
        stats_table = build_season_pitcher_table(pitcher_ids, season)
        if stats_table.empty:
            merged = group.copy()
            for col in PITCHER_FEATURE_COLS:
                merged[col] = None
        else:
            merged = group.merge(stats_table, on="opp_starter_id" if "opp_starter_id" in stats_table.columns else None,
                                   left_on="opp_starter_id", right_on="pitcher_id", how="left")
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    return result.dropna(subset=PITCHER_FEATURE_COLS + ["park_factor"]).reset_index(drop=True)


def main():
    print(f"Looking up player: {PLAYER_FIRST} {PLAYER_LAST}...")
    player_id = get_player_id(PLAYER_FIRST, PLAYER_LAST)

    print(f"Pulling Statcast data ({START_DATE} to {END_DATE})...")
    statcast_df = load_batter_statcast(player_id, START_DATE, END_DATE)

    print("Building game log...")
    game_log = build_game_log(statcast_df)

    print("Adding player rolling features...")
    features_df = add_rolling_features(game_log)

    print("Extracting opposing starters and park info...")
    game_meta = get_starting_pitchers_and_park(statcast_df)

    print("Attaching opposing pitcher season stats and park factor...")
    full_df = attach_pitcher_and_park_features(features_df, game_meta)

    feature_cols = [
        "hr_rate_10", "hr_rate_20", "avg_ev_10", "avg_ev_20",
        "iso_proxy_10", "iso_proxy_20", "career_hr_rate",
        "park_factor",
    ] + PITCHER_FEATURE_COLS

    split_idx = int(len(full_df) * 0.75)
    train_df, test_df = full_df.iloc[:split_idx], full_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["hit_hr"]
    X_test, y_test = test_df[feature_cols], test_df["hit_hr"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("\n=== V3 HR Model Results (player form + pitcher + park) ===")
    print(f"Test games: {len(test_df)}")
    print(f"Actual HR rate in test set: {y_test.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    main()