"""
Style-fit matchup feature for WALK prediction -- the walk equivalent of
the pitch-type matchup that worked for HR (AUC 0.541 -> 0.565).

Different axis than HR's fastball/breaking/offspeed buckets, since walks
aren't really about pitch TYPE -- they're about whether the pitcher
throws strikes at all. So this combines:
  - This batter's own rolling walk tendency (already have this)
  - This specific opposing starter's own CONTROL profile: zone_rate
    (% of his own pitches inside the strike zone) and bb_rate_allowed
    (walks allowed per batter faced), computed from HIS OWN prior-season
    Statcast pull -- same lookahead guard as the HR pitch-mix feature
    (never the same season, which would leak future starts).

Uses its own separate cache key (pitchcontrol_*, not pitchmix_*) so this
never touches or risks breaking the already-working HR pipeline's cache
or function -- at the cost of re-pulling some of the same pitchers'
season data a second time under a different key.
"""
import pandas as pd
from pybaseball import statcast_pitcher
from data.cache import load_cached, save_cache

CONTROL_FEATURE_COLS = ["pitcher_zone_rate", "pitcher_bb_rate_allowed"]


def load_pitcher_control_cached(pitcher_id: int, season: int):
    """One pitcher's own control profile (zone rate, BB rate allowed) for
    a given season -- cached separately from the HR pitch-mix cache."""
    cache_key = f"pitchcontrol_{pitcher_id}_{season}"
    cached = load_cached(cache_key)
    if cached is not None and not cached.empty:
        return cached.iloc[0].to_dict()

    try:
        df = statcast_pitcher(f"{season}-01-01", f"{season}-12-31", pitcher_id)
    except Exception:
        return None
    if df.empty:
        return None

    total_pitches = len(df)
    in_zone = (df["zone"] <= 9).sum()
    pa = df["events"].notna().sum()
    walks_allowed = (df["events"] == "walk").sum()

    if pa == 0:
        return None

    result = {
        "opp_pitcher_id": pitcher_id,
        "season": season,
        "pitcher_zone_rate": in_zone / total_pitches,
        "pitcher_bb_rate_allowed": walks_allowed / pa,
    }
    save_cache(cache_key, pd.DataFrame([result]))
    return result


def build_pitcher_control_table(pitcher_season_pairs) -> pd.DataFrame:
    """pitcher_season_pairs: iterable of (pitcher_id, season) tuples."""
    rows = []
    for pid, season in pitcher_season_pairs:
        result = load_pitcher_control_cached(int(pid), int(season))
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def attach_walk_matchup_features(features_df: pd.DataFrame, control_table: pd.DataFrame) -> pd.DataFrame:
    """
    Merges opposing pitcher control stats onto the features table (must
    already have opp_pitcher_id and mix_season columns), then builds the
    style-fit interaction: this batter's own recent walk rate, scaled by
    how much this SPECIFIC opposing pitcher tends to walk people.
    """
    control_renamed = control_table.rename(columns={"season": "mix_season"})
    df = features_df.merge(control_renamed, on=["opp_pitcher_id", "mix_season"], how="left")
    df["expected_walk_exposure"] = df["walk_rate_20"] * df["pitcher_bb_rate_allowed"]
    return df