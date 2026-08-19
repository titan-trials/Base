import os
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.csv")


def load_cached(key: str):
    """Returns the cached DataFrame if it exists, else None."""
    path = cache_path(key)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def save_cache(key: str, df: pd.DataFrame):
    df.to_csv(cache_path(key), index=False)
    print(f"  Cached {len(df)} rows to cache/{key}.csv")