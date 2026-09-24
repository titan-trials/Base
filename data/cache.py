"""
Simple CSV disk cache.

TWO THINGS HERE ARE LOAD-BEARING, both learned on 2026-09-24 when
score_slate died with:

    ValueError: time data " Yordan"" doesn't match format "%Y-%m-%d"

`" Yordan"` was the back half of `"Alvarez, Yordan"` -- a chunk of bytes
had gone missing from the middle of one row in one player cache, so the
fields after it shifted and a NAME landed in the game_date column. One bad
row out of 10,898 took down the whole scoring run.

1. READING MUST NOT RAISE ON ONE BAD ROW. refresh.load_player_cache
   already coerces and drops unparseable dates -- it was written knowing
   these files can carry junk. But it calls load_cached() first, and
   load_cached parsed dates strictly, so it raised one frame before the
   protection could run. The defensive code existed and was unreachable.

   It also must not drop rows SILENTLY. Five files had been damaged for
   two days and nothing said so; the only reason anyone found out is that
   the sixth one happened to break hard. A dropped row prints.

2. WRITING MUST BE ALL-OR-NOTHING. save_cache wrote straight over the
   real path, so anything that interrupted or interfered with the write
   left a file that was neither the old version nor the new one. Writing
   to a temp file and then os.replace() makes the swap atomic: readers see
   the complete old file or the complete new one, never a half-formed mix.
"""
import os

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.csv")


def load_cached(key: str):
    """
    Returns the cached DataFrame if it exists, else None.

    Rows whose game_date cannot be parsed are dropped and REPORTED. They
    mean the file is damaged; `python diagnose_cache.py --gaps` says how.
    """
    path = cache_path(key)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, low_memory=False)

    if "game_date" in df.columns:
        parsed = pd.to_datetime(df["game_date"], errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            sample = df.loc[parsed.isna(), "game_date"].astype(str).head(3)
            print(f"  WARNING cache/{key}.csv is damaged: {bad} of "
                  f"{len(df):,} rows have an unreadable game_date and were "
                  f"dropped.")
            print(f"    e.g. {', '.join(repr(v) for v in sample)}")
            print(f"    Run: python diagnose_cache.py --gaps")
            df = df.loc[parsed.notna()].copy()
            parsed = parsed.loc[parsed.notna()]
        df["game_date"] = parsed

    return df


def save_cache(key: str, df: pd.DataFrame):
    """Write atomically: a reader sees the whole old file or the whole new one."""
    path = cache_path(key)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)          # atomic on Windows and POSIX, same volume
    except BaseException:
        # Leave the existing cache untouched rather than half-replaced.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    print(f"  Cached {len(df)} rows to cache/{key}.csv")
