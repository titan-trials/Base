"""
Builds (or loads from cache) the full 30-team feature table. This is the
expensive ~90-request pull -- caching means we pay that cost once, then
every script that needs team-level context (train_team_win.py, and the
player-to-team bridge) reuses the same cached result instead of each
re-pulling it separately.
"""
import pandas as pd
from data.team_data import load_team_season
from data.cache import load_cached, save_cache
from features.team_features import build_team_game_log, add_team_rolling_features, attach_opponent_features

CACHE_KEY = "team_features_full"

# Baseball-Reference uses different abbreviations than the common ones for
# these 6 teams. 3 of 30 team-seasons failed in earlier runs -- almost
# certainly one or more of these. Tried as a fallback when the common
# abbreviation fails, not as the primary, since the common ones work fine
# for the other 24 teams.
BR_ALTERNATES = {
    "CHW": "CWS", "KC": "KCR", "SD": "SDP",
    "SF": "SFG", "TB": "TBR", "WSH": "WSN",
}


def _pull_one_team_season(team: str, season: int):
    """Try the given abbreviation, then its Baseball-Reference-specific
    alternate if that fails."""
    try:
        return load_team_season(team, season)
    except Exception as first_error:
        alternate = BR_ALTERNATES.get(team)
        if alternate is None:
            raise first_error
        print(f"  {team} failed, retrying with BR alternate '{alternate}'...")
        return load_team_season(alternate, season)


def get_full_team_features(teams: list, seasons: list, force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh:
        cached = load_cached(CACHE_KEY)
        if cached is not None:
            print(f"Loaded {len(cached)} team-games from cache/{CACHE_KEY}.csv -- skipping the pull.")
            return cached

    all_teams = []
    for team in teams:
        for season in seasons:
            print(f"Pulling {team} {season} schedule...")
            try:
                schedule_df = _pull_one_team_season(team, season)
                game_log = build_team_game_log(schedule_df, season)
                features_df = add_team_rolling_features(game_log)
                all_teams.append(features_df)
            except Exception as e:
                print(f"  Skipping {team} {season}: {e}")

    combined = pd.concat(all_teams, ignore_index=True)
    print(f"Total team-games (own form only): {len(combined)}")

    print("Attaching opponent strength features...")
    full_df = attach_opponent_features(combined)
    print(f"Total team-games (after opponent match): {len(full_df)}")

    save_cache(CACHE_KEY, full_df)
    return full_df