"""
Per-game environmental context from MLB's own Stats API.

WHY THIS FILE EXISTS
--------------------
Statcast (via pybaseball) gives pitch-level physics but carries NO weather
and NO game start time. Those live in MLB's Stats API, under
`gameData` on the live-feed endpoint. That endpoint also happens to be the
exact source behind the "83deg 11 mph, R To L" string you see on public
HR-prop sites -- so this is the primary source, not a reconstruction.

Endpoint used (field-filtered so the response is a few hundred bytes
instead of a few megabytes):

    https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
        ?fields=gameData,datetime,dayNight,officialDate,
                venue,id,name,weather,condition,temp,wind

Free, no API key, no rate limit published (a small polite delay is used
anyway).

CACHING
-------
One combined CSV (`cache/game_context.csv`), appended incrementally and
re-saved every CHECKPOINT_EVERY games. A season's weather never changes
once played, and the first build across ~4-5k games takes a while, so an
interrupted run must be able to resume -- hence checkpointing rather than
one save at the end.

WIND, AND WHY IT NEEDS PARSING
------------------------------
MLB reports wind as a human string: "11 mph, R To L" / "8 mph, Out To CF"
/ "5 mph, In From LF" / "Calm" / "12 mph, Varies". The direction is
already expressed RELATIVE TO THE FIELD, which is exactly what we want --
it means we don't need park compass orientation to know whether the wind
is helping a fly ball.

Two components get extracted:

  wind_out_mph   signed along the home-plate -> center-field axis.
                 Positive = blowing out (carries fly balls), negative =
                 blowing in (knocks them down), 0 = pure crosswind.

  wind_pull_mph  signed toward the batter's PULL field. Requires batter
                 handedness, so it's computed later in
                 features/context_features.py, not here -- this file only
                 stores the raw left/right component (`wind_toward_lf`:
                 +1 if wind blows toward left field, -1 toward right).

Note "R To L" means FROM right field TOWARD left field, i.e. toward LF.
That helps a right-handed pull hitter and hurts a lefty. Getting this
sign backwards would silently invert the feature, so it is spelled out
here rather than left implicit.

TEMPERATURE
-----------
Real and physical, not folklore: warmer air is less dense, so a batted
ball carries further -- roughly a few feet per 10 degrees F, which is
enough to matter at the wall. Stored raw in Fahrenheit; the model decides
how much it's worth.
"""
import os
import re
import time
import json
import urllib.request
import urllib.error

import pandas as pd

from data.cache import cache_path

CACHE_KEY = "game_context"
CHECKPOINT_EVERY = 100
REQUEST_DELAY_SEC = 0.12
TIMEOUT_SEC = 20

FIELDS = (
    "gameData,datetime,dayNight,officialDate,"
    "venue,id,name,weather,condition,temp,wind"
)
URL = (
    "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    "?fields=" + FIELDS
)

CONTEXT_COLS = [
    "game_pk", "venue_id", "venue_name", "day_night",
    "temp_f", "sky_condition", "wind_mph", "wind_dir_raw",
    "wind_out_mph", "wind_toward_lf",
]

# Wind direction phrasing -> (out/in axis sign, toward-LF sign).
#   out/in:  +1 blowing out to the outfield, -1 blowing in, 0 neither
#   towardLF:+1 blowing toward left field, -1 toward right field, 0 neither
_WIND_DIRECTIONS = {
    "out to cf": (1.0, 0.0),
    "out to lf": (0.7, 0.7),
    "out to rf": (0.7, -0.7),
    "in from cf": (-1.0, 0.0),
    "in from lf": (-0.7, -0.7),
    "in from rf": (-0.7, 0.7),
    "l to r": (0.0, -1.0),   # from left field toward right field
    "r to l": (0.0, 1.0),    # from right field toward left field
    "calm": (0.0, 0.0),
    "none": (0.0, 0.0),
    "varies": (0.0, 0.0),
}


def parse_wind(wind_str) -> tuple:
    """
    '11 mph, R To L' -> (11.0, 'r to l', 0.0, 11.0)

    Returns (speed_mph, direction_key, out_component_mph, toward_lf_mph).
    Unknown or missing wind -> (0.0, '', 0.0, 0.0) rather than NaN, since
    'no wind reported' behaves like calm for modelling purposes and NaN
    would drop the whole game row later.
    """
    if wind_str is None or (isinstance(wind_str, float) and pd.isna(wind_str)):
        return 0.0, "", 0.0, 0.0

    text = str(wind_str).strip().lower()
    if not text:
        return 0.0, "", 0.0, 0.0

    speed_match = re.search(r"(\d+(?:\.\d+)?)\s*mph", text)
    speed = float(speed_match.group(1)) if speed_match else 0.0

    direction_key = ""
    out_sign, lf_sign = 0.0, 0.0
    for key, (o_sign, l_sign) in _WIND_DIRECTIONS.items():
        if key in text:
            direction_key, out_sign, lf_sign = key, o_sign, l_sign
            break

    return speed, direction_key, speed * out_sign, speed * lf_sign


def _fetch_one(game_pk: int) -> dict:
    """One field-filtered live-feed call. Returns a context dict, or None
    if the game isn't retrievable (spring training, cancelled, bad pk)."""
    try:
        request = urllib.request.Request(
            URL.format(game_pk=int(game_pk)),
            headers={"User-Agent": "baseball_predictor/1.0"},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None

    game_data = payload.get("gameData") or {}
    weather = game_data.get("weather") or {}
    venue = game_data.get("venue") or {}
    datetime_block = game_data.get("datetime") or {}

    temp_raw = weather.get("temp")
    try:
        temp_f = float(temp_raw)
    except (TypeError, ValueError):
        temp_f = None

    speed, direction_key, out_mph, toward_lf = parse_wind(weather.get("wind"))

    return {
        "game_pk": int(game_pk),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "day_night": datetime_block.get("dayNight"),
        "temp_f": temp_f,
        "sky_condition": weather.get("condition"),
        "wind_mph": speed,
        "wind_dir_raw": direction_key,
        "wind_out_mph": out_mph,
        "wind_toward_lf": toward_lf,
    }


def _load_existing() -> pd.DataFrame:
    path = cache_path(CACHE_KEY)
    if not os.path.exists(path):
        return pd.DataFrame(columns=CONTEXT_COLS)
    existing = pd.read_csv(path)
    for col in CONTEXT_COLS:
        if col not in existing.columns:
            existing[col] = None
    return existing[CONTEXT_COLS]


def _save(df: pd.DataFrame):
    df.to_csv(cache_path(CACHE_KEY), index=False)


def get_game_context(game_pks, verbose: bool = True) -> pd.DataFrame:
    """
    One row per game_pk with venue, day/night and weather.

    Only games not already in cache/game_context.csv are fetched, so the
    slow first build happens once and every later run is instant. Games
    that fail to fetch are recorded as a row of NULLs so they aren't
    retried forever on every subsequent run -- delete the cache file to
    force a full refetch.
    """
    wanted = sorted({int(pk) for pk in pd.Series(list(game_pks)).dropna().unique()})
    cached = _load_existing()
    have = set(cached["game_pk"].dropna().astype(int)) if not cached.empty else set()
    missing = [pk for pk in wanted if pk not in have]

    if verbose:
        print(f"  Game context: {len(wanted)} games wanted, "
              f"{len(have & set(wanted))} cached, {len(missing)} to fetch.")

    if not missing:
        return cached[cached["game_pk"].isin(wanted)].reset_index(drop=True)

    new_rows, network_failures = [], 0
    for i, game_pk in enumerate(missing, start=1):
        row = _fetch_one(game_pk)
        if row is None:
            # A failed REQUEST is transient -- leave it uncached so a later
            # run retries it. Writing a permanent "no data" row here would
            # mean one network hiccup silently blanks the weather for those
            # games forever, with no way to tell that from a genuine dome.
            network_failures += 1
            if network_failures >= 20 and network_failures == i:
                print(f"    Aborting: the first {i} requests all failed. "
                      f"Check network access to statsapi.mlb.com.")
                break
            continue
        new_rows.append(row)
        time.sleep(REQUEST_DELAY_SEC)

        if i % CHECKPOINT_EVERY == 0 and new_rows:
            _save(pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True))
            if verbose:
                print(f"    ...{i}/{len(missing)} fetched (checkpointed)")

    if network_failures and verbose:
        print(f"    {network_failures} request(s) failed and were NOT cached "
              f"-- rerun to retry just those.")
    if not new_rows:
        return cached[cached["game_pk"].isin(wanted)].reset_index(drop=True)

    combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset="game_pk", keep="last")
    _save(combined)
    if verbose:
        print(f"  Game context cached to cache/{CACHE_KEY}.csv "
              f"({len(combined)} games total).")

    return combined[combined["game_pk"].isin(wanted)].reset_index(drop=True)


def attach_game_context(features_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Left-join game context onto a per-(batter, game) feature frame."""
    context = get_game_context(features_df["game_pk"].unique(), verbose=verbose)
    if context.empty:
        for col in CONTEXT_COLS:
            if col != "game_pk":
                features_df[col] = None
        return features_df
    context["game_pk"] = context["game_pk"].astype("int64")
    merged = features_df.copy()
    merged["game_pk"] = merged["game_pk"].astype("int64")
    return merged.merge(context, on="game_pk", how="left")
