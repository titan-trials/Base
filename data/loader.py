"""
Data loading for the Baseball Player Prop Predictor.
Uses pybaseball (free, scrapes Baseball Savant) to pull Statcast data.
V1 scope: one player at a time, matching Quantara's "one stock" starting point.
"""
import unicodedata
import pandas as pd
from pybaseball import statcast_batter, playerid_lookup
from data.cache import load_cached, save_cache


def _normalize(s: str) -> str:
    """Strip accents for name matching, e.g. 'Alvarez' should match
    'Álvarez' -- the Chadwick ID registry stores accented names, but
    people naturally type the unaccented version."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def get_player_id(first_name: str, last_name: str) -> int:
    """Look up a player's MLBAM ID from their name."""
    lookup = playerid_lookup(last_name, first_name)

    if lookup.empty:
        # Exact match failed. This isn't just a first-name accent issue --
        # the registry can store the LAST name itself accented too (e.g.
        # Alvarez -> "álvarez"), so an exact last-name search misses the
        # player entirely, not just a first-name variant. fuzzy=True
        # handles accent/spelling variation on both fields at once.
        try:
            fuzzy_result = playerid_lookup(last_name, first_name, fuzzy=True)
        except TypeError:
            fuzzy_result = pd.DataFrame()

        if not fuzzy_result.empty:
            target = _normalize(first_name)
            exact_first = fuzzy_result[fuzzy_result["name_first"].apply(_normalize) == target]
            lookup = exact_first if not exact_first.empty else fuzzy_result.head(1)

    if lookup.empty:
        raise ValueError(f"No player found for {first_name} {last_name}")

    # If multiple matches (e.g. a retired player with the same name), take the
    # one who played most recently.
    lookup = lookup.sort_values("mlb_played_last", ascending=False)
    return int(lookup.iloc[0]["key_mlbam"])


def load_batter_statcast(player_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pull raw pitch-by-pitch Statcast data for one batter over a date range."""
    df = statcast_batter(start_dt, end_dt, player_id)
    if df.empty:
        raise ValueError(
            "No Statcast data returned for this player/date range. "
            "Check the player ID and dates (Statcast data starts 2015, "
            "full data quality from 2020 on)."
        )
    return df


def load_batter_statcast_cached(first: str, last: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    """Same as load_batter_statcast, but checks a local cache first --
    a completed season's pitch data never changes, so there's no reason
    to re-pull it every run."""
    cache_key = f"statcast_{first}_{last}_{start_dt}_{end_dt}".replace(" ", "_")
    cached = load_cached(cache_key)
    if cached is not None:
        print(f"  Loaded {first} {last} from cache/{cache_key}.csv -- skipping the pull.")
        return cached

    player_id = get_player_id(first, last)
    df = load_batter_statcast(player_id, start_dt, end_dt)
    save_cache(cache_key, df)
    return df