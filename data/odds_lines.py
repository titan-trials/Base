"""
Closing lines from the sportsbooks -- the benchmark the model is missing.

WHY THIS MATTERS MORE THAN ANY FEATURE
---------------------------------------
Every skill number in this project is measured against "quote the league
base rate for everyone." That is a low bar. A sportsbook clears it
comfortably, so +1.5% Brier skill answers *is the model better than
nothing* and says nothing at all about *is the model any good*.

Those are different questions and only the second one matters. A model can
be beautifully calibrated and still have no edge, because the market is
also beautifully calibrated -- and better. Until these numbers sit beside
a closing line, there is no way to tell whether this is an edge or an
expensive way to reproduce public information.

It also doubles as a curriculum. The five biggest disagreements between
the model and the closing line, every night, are a list of things the
market knows that the model does not, delivered in the exact places where
the difference costs money.

THE CREDIT BUDGET IS THE BINDING CONSTRAINT
--------------------------------------------
The Odds API free tier gives 500 credits a month, and

    cost = (unique markets in the response) x (number of regions)

charged per event. The events endpoint is free.

So on a fifteen-game slate:

    1 market  x 1 region  =  15 credits/night  ->  33 nights   (fits)
    2 markets x 1 region  =  30 credits/night  ->  16 nights
    4 markets x 1 region  =  60 credits/night  ->   8 nights   (does not fit)

One market per night is the only thing that covers a month. DEFAULT_MARKET
is pitcher strikeouts because that is where the model's measured edge is
largest (+0.064 Brier skill out of sample, against +0.018 on the hitter
side) and because the market is liquid. Change it if you would rather
benchmark home runs; do not quietly turn on four at once.

FETCH LATE. The value is in the CLOSING line -- the price just before
first pitch, after the market has absorbed lineups, weather and money. An
opening line is a much weaker benchmark. Predict early, fetch odds late,
compare afterwards.

THE KEY NEVER GOES IN THE REPO
-------------------------------
Read from the ODDS_API_KEY environment variable, or from a file named
`.odds_api_key` in the project root. That filename is gitignored. If you
ever paste a key into a tracked file, rotate it -- git history is
forever and the repo is public.

    setx ODDS_API_KEY "your-key-here"        (Windows, new shell after)
    echo your-key-here > .odds_api_key       (simpler)

USAGE
-----
    python -m data.odds_lines 2026-08-31
"""
import os
import json
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

from data.cache import cache_path

SPORT = "baseball_mlb"
BASE = "https://api.the-odds-api.com/v4"

# See the credit arithmetic above before adding to this.
DEFAULT_MARKET = "pitcher_strikeouts"
KNOWN_MARKETS = (
    "pitcher_strikeouts",
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
)
DEFAULT_REGIONS = "us"
KEY_FILE = ".odds_api_key"
TIMEOUT_SEC = 20
REQUEST_DELAY_SEC = 0.15


def _clean_key(raw: bytes) -> str:
    """
    Decode a key file written by any of the obvious means.

    PowerShell's `echo key > file` writes UTF-16 LE WITH A BOM, so the
    bytes are FF FE 39 00 30 00 ... -- every character followed by a null.
    Read as UTF-8 that becomes a string full of \\x00, which urllib then
    refuses with "URL can't contain control characters", from a traceback
    forty lines deep in http.client that says nothing about encodings.

    Trying UTF-8-with-BOM first and UTF-16 second covers `echo`,
    `Set-Content`, Notepad and any sane editor. Nulls and whitespace are
    stripped regardless, because the cost of being wrong here is an
    inscrutable error at the bottom of the stack.
    """
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        cleaned = text.replace("\x00", "").strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return ""


def _validate_key(key: str) -> str:
    """
    Fail here, with a readable message, rather than inside urllib.

    The keys are hex-ish and about 32 characters. Anything with spaces or
    control characters in it is a file-encoding problem, not a key
    problem, and saying so saves the twenty minutes it otherwise costs.
    """
    if not key:
        raise SystemExit(
            "The key file exists but is empty after decoding.\n"
            f"  Rewrite it: Set-Content -Path {KEY_FILE} -Value 'yourkey' "
            f"-Encoding ascii -NoNewline")
    if not key.isalnum():
        bad = [c for c in key if not c.isalnum()][:5]
        raise SystemExit(
            f"The key contains characters that are not letters or digits "
            f"({bad!r}), which almost always means the file was saved in "
            f"the wrong encoding.\n"
            f"  In PowerShell, `echo k > file` writes UTF-16. Use:\n"
            f"    Set-Content -Path {KEY_FILE} -Value 'yourkey' "
            f"-Encoding ascii -NoNewline\n"
            f"  Or skip the file entirely:  $env:ODDS_API_KEY='yourkey'")
    return key


def load_api_key() -> str:
    """Environment first, then the gitignored file. Never a literal."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return _validate_key(key)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, KEY_FILE)
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return _validate_key(_clean_key(handle.read()))
    raise SystemExit(
        "No API key. Set ODDS_API_KEY, or put the key in a file called "
        f"{KEY_FILE} in the project root (it is gitignored).\n"
        f"  PowerShell: Set-Content -Path {KEY_FILE} -Value 'yourkey' "
        f"-Encoding ascii -NoNewline\n"
        "  Free tier: https://the-odds-api.com/"
    )


def _get(url: str):
    """Returns (payload, headers) or (None, {}) on failure. Never raises."""
    try:
        with urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "baseball_predictor/1.0"}),
                timeout=TIMEOUT_SEC) as response:
            return json.load(response), dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        print(f"    HTTP {exc.code}: {body}")
        return None, {}
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError,
            OSError) as exc:
        print(f"    Request failed: {exc}")
        return None, {}


def list_events(api_key: str, game_date: str) -> list:
    """
    Tonight's games. FREE -- this endpoint does not count against quota,
    which is why the event ids are looked up rather than guessed.
    """
    url = f"{BASE}/sports/{SPORT}/events?apiKey={api_key}"
    events, _ = _get(url)
    if not events:
        return []
    target = pd.Timestamp(game_date).date()
    out = []
    for event in events:
        start = pd.to_datetime(event.get("commence_time"), utc=True)
        # Compare in US Eastern: a 10pm Eastern game is already "tomorrow"
        # in UTC, and matching on the UTC date silently drops every late
        # game on the slate.
        if start.tz_convert("America/New_York").date() == target:
            out.append(event)
    return out


def american_to_prob(price) -> float:
    """American odds to implied probability, vig included."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return float("nan")
    if price == 0:
        return float("nan")
    return (100.0 / (price + 100.0) if price > 0
            else (-price) / ((-price) + 100.0))


def devig_two_way(over_prob: float, under_prob: float) -> float:
    """
    Remove the bookmaker's margin from a two-sided market.

    Raw implied probabilities sum to more than 1 -- that excess IS the
    book's margin, and comparing a model probability against a vigged one
    would make the model look better than it is by roughly half the
    margin on every single line. Normalising both sides to sum to 1 is
    the standard first-order fix.

    It is not perfect: real margin is not split evenly between the two
    sides (favourite-longshot bias), so a proportional de-vig slightly
    over-prices favourites. Better methods exist. This one is honest,
    simple, and enormously better than not de-vigging.
    """
    if not np.isfinite(over_prob) or not np.isfinite(under_prob):
        return float("nan")
    total = over_prob + under_prob
    return over_prob / total if total > 0 else float("nan")


def fetch_event_props(api_key: str, event_id: str, markets: str,
                      regions: str = DEFAULT_REGIONS):
    """One event's player props. COSTS (unique markets) x (regions)."""
    url = (f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
           f"?apiKey={api_key}&regions={regions}&markets={markets}"
           f"&oddsFormat=american")
    return _get(url)


def _rows_from_event(event: dict, payload: dict) -> list:
    """
    Flatten one event's response into (player, line, over, under) rows.

    The API nests bookmaker -> market -> outcome. Each outcome carries a
    `description` (the player) and a `point` (the line), with Over and
    Under as separate entries that have to be paired back up before the
    vig can be removed.
    """
    rows = []
    for book in payload.get("bookmakers") or []:
        for market in book.get("markets") or []:
            pairs = {}
            for outcome in market.get("outcomes") or []:
                player = outcome.get("description")
                point = outcome.get("point")
                side = (outcome.get("name") or "").lower()
                if player is None or point is None or side not in ("over", "under"):
                    continue
                pairs.setdefault((player, float(point)), {})[side] = \
                    outcome.get("price")
            for (player, point), sides in pairs.items():
                if "over" not in sides or "under" not in sides:
                    # A one-sided quote cannot be de-vigged, so it is
                    # dropped rather than compared on a vigged number.
                    continue
                over_raw = american_to_prob(sides["over"])
                under_raw = american_to_prob(sides["under"])
                rows.append({
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": book.get("key"),
                    "market": market.get("key"),
                    "player": player,
                    "line": float(point),
                    "price_over": sides["over"],
                    "price_under": sides["under"],
                    "prob_over_raw": over_raw,
                    "hold": over_raw + under_raw - 1.0,
                    "prob_over": devig_two_way(over_raw, under_raw),
                })
    return rows


def consensus(rows: pd.DataFrame) -> pd.DataFrame:
    """
    One de-vigged number per (player, market, line), across books.

    Median rather than mean: a single book with a stale or mistaken price
    should not drag the benchmark, and with a handful of books the median
    is the robust choice. `n_books` is kept so a consensus built from one
    quote is visibly different from one built from eight.
    """
    if rows.empty:
        return rows
    grouped = rows.groupby(["market", "player", "line"], as_index=False).agg(
        prob_over=("prob_over", "median"),
        hold=("hold", "median"),
        n_books=("bookmaker", "nunique"),
        commence_time=("commence_time", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
    )
    return grouped.sort_values(["market", "line", "prob_over"],
                              ascending=[True, True, False])


def fetch_slate_odds(game_date: str, markets: str = DEFAULT_MARKET,
                     regions: str = DEFAULT_REGIONS,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Fetch, de-vig and cache one slate's closing lines.

    Writes cache/odds_{date}.csv. Re-running overwrites, so fetching again
    closer to first pitch replaces an earlier, weaker line -- which is
    usually what you want, at the cost of more credits.
    """
    api_key = load_api_key()
    events = list_events(api_key, game_date)
    if not events:
        print(f"  No events found for {game_date}. Nothing fetched, "
              f"no credits spent.")
        return pd.DataFrame()

    n_markets = len([m for m in markets.split(",") if m.strip()])
    n_regions = len([r for r in regions.split(",") if r.strip()])
    if verbose:
        print(f"  {len(events)} events for {game_date}.")
        print(f"  Markets: {markets} | regions: {regions}")
        print(f"  Estimated cost: {len(events)} x {n_markets} x {n_regions} "
              f"= up to {len(events) * n_markets * n_regions} credits "
              f"(you are charged only for markets that come back with data).")

    all_rows, remaining = [], None
    for i, event in enumerate(events, start=1):
        payload, headers = fetch_event_props(api_key, event["id"], markets,
                                             regions)
        remaining = headers.get("x-requests-remaining", remaining)
        if payload:
            all_rows.extend(_rows_from_event(event, payload))
        time.sleep(REQUEST_DELAY_SEC)
        if verbose and i % 5 == 0:
            print(f"    {i}/{len(events)} events, "
                  f"{len(all_rows)} quotes, {remaining} credits left")

    if not all_rows:
        print("  No two-sided quotes came back. Either the market is not "
              "posted yet, or the plan does not include it.")
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    lines = consensus(raw)
    lines["game_date"] = game_date
    lines["fetched_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.to_csv(cache_path(f"odds_{game_date}"), index=False)

    if verbose:
        print(f"\n  {len(lines)} lines from {raw['bookmaker'].nunique()} "
              f"bookmakers. Median hold {raw['hold'].median():.3f}.")
        print(f"  Credits remaining: {remaining}")
        print(f"  Saved to cache/odds_{game_date}.csv")
    return lines


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        raise SystemExit("Usage: python -m data.odds_lines YYYY-MM-DD "
                         "[market]")
    market = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MARKET
    if market not in KNOWN_MARKETS:
        print(f"  Note: '{market}' is not one of {KNOWN_MARKETS}. "
              f"Sending it anyway.")
    fetch_slate_odds(date, markets=market)
