"""
Pitch-type matchup features -- does the opposing starter's typical pitch
mix line up with what THIS batter tends to hit HRs against?

Two pulls happen here:
  1. Batter's own pitch-level data -- already have this, no new pull.
  2. Each opposing STARTER's own full pitch mix -- a new per-pitcher
     Statcast pull, cached (data/cache.py), one call per unique starter
     faced. Expect this to be slow on first run (dozens of pitchers).
"""
import pandas as pd
from pybaseball import statcast_pitcher
from data.cache import load_cached, save_cache

FASTBALL_TYPES = {"FF", "FT", "SI", "FC"}
BREAKING_TYPES = {"SL", "CU", "KC", "SC"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "EP"}
BUCKETS = ["fastball", "breaking", "offspeed"]


def _bucket(pitch_type) -> str:
    if pd.isna(pitch_type):
        return "other"
    if pitch_type in FASTBALL_TYPES:
        return "fastball"
    if pitch_type in BREAKING_TYPES:
        return "breaking"
    if pitch_type in OFFSPEED_TYPES:
        return "offspeed"
    return "other"


def get_starting_pitcher_per_game(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the first opposing pitcher this batter faced
    (approximates the starter)."""
    df = statcast_df.copy()
    sort_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in df.columns]
    df = df.sort_values(sort_cols)
    return df.groupby("game_pk")["pitcher"].first().rename("opp_pitcher_id").reset_index()


def build_batter_bucket_log(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Per (batter, game_pk, game_date): how many pitches seen, and how
    many resulted in a HR, broken out by pitch-type bucket -- wide format,
    one column pair per bucket. Game-level counts, not pitch-level rolling,
    to keep rolling logic consistent with the rest of this project."""
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["bucket"] = df["pitch_type"].apply(_bucket)
    df["pitch_is_hr"] = (df["events"] == "home_run").astype(int)

    wide = df.pivot_table(
        index=["batter", "game_pk", "game_date"],
        columns="bucket",
        values="pitch_is_hr",
        aggfunc=["sum", "count"],
        fill_value=0,
    )
    wide.columns = [f"{stat}_{bucket}" for stat, bucket in wide.columns]
    wide = wide.reset_index().sort_values(["batter", "game_date"])
    return wide


def add_bucket_rolling_rates(bucket_log: pd.DataFrame, windows=(20, 40)) -> pd.DataFrame:
    """Rolling HR rate per pitch-type bucket, over a window of GAMES (not
    pitches) -- HR-per-pitch-type is rare enough that it needs a wider
    window than the 10/20-game windows used elsewhere. Shifted by 1 game,
    same lookahead guard as everywhere else in this project."""
    df = bucket_log.copy()

    def _add(group):
        for bucket in BUCKETS:
            hr_col, pitch_col = f"sum_{bucket}", f"count_{bucket}"
            if hr_col not in group.columns:
                group[hr_col] = 0
            if pitch_col not in group.columns:
                group[pitch_col] = 0
            for w in windows:
                rolling_hr = group[hr_col].rolling(w, min_periods=5).sum().shift(1)
                rolling_pitches = group[pitch_col].rolling(w, min_periods=5).sum().shift(1)
                group[f"hr_rate_vs_{bucket}_{w}"] = rolling_hr / rolling_pitches.replace(0, pd.NA)
        return group

    df = df.groupby("batter", group_keys=False).apply(_add)
    return df


def load_pitcher_pitch_mix_cached(pitcher_id: int, season: int):
    """One pitcher's pitch-type mix (% fastball/breaking/offspeed) for a
    given season, cached since it's slow to pull and never changes once
    the season is over."""
    cache_key = f"pitchmix_{pitcher_id}_{season}"
    cached = load_cached(cache_key)
    if cached is not None and not cached.empty:
        return cached.iloc[0].to_dict()

    try:
        df = statcast_pitcher(f"{season}-01-01", f"{season}-12-31", pitcher_id)
    except Exception:
        return None
    if df.empty:
        return None

    df["bucket"] = df["pitch_type"].apply(_bucket)
    total = len(df)
    counts = df["bucket"].value_counts()
    result = {
        "opp_pitcher_id": pitcher_id,
        "season": season,
        "fastball_pct": counts.get("fastball", 0) / total,
        "breaking_pct": counts.get("breaking", 0) / total,
        "offspeed_pct": counts.get("offspeed", 0) / total,
    }
    save_cache(cache_key, pd.DataFrame([result]))
    return result


def build_pitcher_mix_table(pitcher_season_pairs) -> pd.DataFrame:
    """pitcher_season_pairs: iterable of (pitcher_id, season) tuples."""
    rows = []
    for pid, season in pitcher_season_pairs:
        result = load_pitcher_pitch_mix_cached(int(pid), int(season))
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)