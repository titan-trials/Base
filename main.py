from data.loader import get_player_id, load_batter_statcast
from features.build_features import build_game_log, add_rolling_features
from model.train import train_hr_model
from config import PLAYER_FIRST, PLAYER_LAST, START_DATE, END_DATE


def main():
    print(f"Looking up player: {PLAYER_FIRST} {PLAYER_LAST}...")
    player_id = get_player_id(PLAYER_FIRST, PLAYER_LAST)

    print(f"Pulling Statcast data ({START_DATE} to {END_DATE})...")
    statcast_df = load_batter_statcast(player_id, START_DATE, END_DATE)

    print("Building game log...")
    game_log = build_game_log(statcast_df)

    print("Adding rolling features...")
    features_df = add_rolling_features(game_log)

    print(f"Training on {len(features_df)} games...")
    model, results = train_hr_model(features_df)

    print("\nMost recent test-set predictions:")
    print(results[["game_date", "hit_hr", "hr_probability"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()