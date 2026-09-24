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
import socket
import random
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


# Retries, and the one rule that decides which failures get one.
#
# Player props are charged PER EVENT. So a retry is only safe when the
# request CANNOT have reached the server -- DNS failure, connection
# refused, connection reset, or a 5xx the server itself says it did not
# process. Those cost nothing and are worth repeating.
#
# A TIMEOUT is the dangerous one and is deliberately NOT retried. Twenty
# seconds without a reply does not mean the request never arrived; the
# Odds API may have served and BILLED it while the response was lost. On a
# 500-credit month at ~25 credits a slate, paying twice for one event to
# save a re-run is a bad trade -- especially because the re-run is nearly
# free: `already_captured()` reads odds_{date}.csv and skips every game
# already in it, so running the odds step again fetches only what is
# missing. A truncated body (JSONDecodeError) is the same case -- the
# server answered, so it billed.
#
# 4xx is never retried either: 401 is a bad key, 422 bad parameters, and
# 429 means the quota is gone and asking again makes it worse.
RETRY_ATTEMPTS = 3
RETRY_BASE_SEC = 1.5


def _get(url: str, attempts: int = RETRY_ATTEMPTS):
    """Returns (payload, headers) or (None, {}) on failure. Never raises."""
    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(
                        url, headers={"User-Agent": "baseball_predictor/1.0"}),
                    timeout=TIMEOUT_SEC) as response:
                return json.load(response), dict(response.headers)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            print(f"    HTTP {exc.code}: {body}")
            # Only the server's own "I failed" is worth repeating.
            if exc.code >= 500 and not last:
                _backoff(attempt, f"HTTP {exc.code}")
                continue
            return None, {}
        except json.JSONDecodeError as exc:
            print(f"    Truncated response ({exc}). NOT retried -- the "
                  f"server answered, so it billed. Re-run the odds step; "
                  f"captured games are skipped.")
            return None, {}
        except (TimeoutError, socket.timeout) as exc:
            print(f"    Timed out after {TIMEOUT_SEC}s ({exc}). NOT retried "
                  f"-- this may have been served and billed. Re-run the odds "
                  f"step; captured games are skipped.")
            return None, {}
        except (urllib.error.URLError, OSError) as exc:
            # URLError wraps a socket timeout too, and that one must follow
            # the timeout rule rather than the connection-error rule.
            if isinstance(getattr(exc, "reason", None),
                          (TimeoutError, socket.timeout)):
                print(f"    Timed out after {TIMEOUT_SEC}s. NOT retried -- "
                      f"this may have been served and billed. Re-run the "
                      f"odds step; captured games are skipped.")
                return None, {}
            print(f"    Request failed: {exc}")
            if not last:
                _backoff(attempt, str(exc)[:60])
                continue
            return None, {}
    return None, {}


def _backoff(attempt: int, why: str):
    """Wait before the next try. Jittered so parallel runs do not sync up."""
    delay = RETRY_BASE_SEC * (2 ** (attempt - 1))
    delay += random.uniform(0, delay * 0.25)
    print(f"    retrying in {delay:.1f}s ({why})")
    time.sleep(delay)


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


def _merge_capture(fresh: pd.DataFrame, path: str, keys: list,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Add a capture to whatever is already on disk instead of replacing it.

    Overwriting was safe only while every run fetched the whole slate.
    Now that started games are skipped to save credits, a late run returns
    ONLY the games still to come -- and a plain to_csv would replace a full
    slate with those few. That is exactly backwards: the rows it deletes
    are pre-game prices already paid for, and they cannot be re-fetched at
    any price.

    So: rows for a game not in this capture are kept as they were, and a
    game present in both takes the NEW row, because a later pre-game price
    is a better closing line than an earlier one. Each row carries its own
    fetched_at_utc, so a file can legitimately hold several capture times
    and anything reading it can tell which price was taken when.
    """
    if not os.path.exists(path):
        return fresh
    try:
        previous = pd.read_csv(path)
    except Exception:
        return fresh
    if previous.empty or not set(keys).issubset(previous.columns):
        return fresh
    combined = pd.concat([previous, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    kept = len(combined) - len(fresh)
    if verbose and kept > 0:
        print(f"  Kept {kept} earlier row(s) for games already underway; "
              f"{len(fresh)} refreshed. File now holds {len(combined)}.")
    return combined


CREDIT_LOG_KEY = "credit_log"


def log_credits(game_date, call, fetched, skipped, remaining, verbose=True):
    """
    One row per odds call, so the month's spend is a number rather than a
    memory.

    The free tier is 500 credits a month and player props are charged PER
    EVENT, so a fifteen-game slate is fifteen credits and a second capture
    the same night is fifteen more. That arithmetic decides how many
    nights of the season get benchmarked at all, and until now it existed
    only in a docstring and in whatever the last run happened to print.

    `remaining` comes from the API's own x-requests-remaining header, so
    it is authoritative rather than something this file counts up and
    slowly gets wrong.
    """
    path = cache_path(CREDIT_LOG_KEY)
    row = {
        "fetched_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_date": game_date, "call": call,
        "events_fetched": int(fetched),
        # Skipped for either reason -- already underway, or already
        # priced. Both are credits not spent, which is what the CREDITS
        # block in run_slate reports.
        "events_skipped": int(skipped),
        "credits_charged": int(fetched) if call == "props" else GAMELINE_CREDITS,
        "credits_saved": int(skipped) if call == "props" else 0,
        "credits_remaining": (int(remaining)
                              if remaining not in (None, "") else None),
    }
    try:
        prior = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        pd.concat([prior, pd.DataFrame([row])],
                  ignore_index=True).to_csv(path, index=False)
    except Exception as exc:          # never fail a capture over bookkeeping
        if verbose:
            print(f"  (credit log not written: {exc})")
    return row


# Game lines are charged per market for the WHOLE slate, not per event:
# h2h + totals, one region = 2 credits however many games there are.
GAMELINE_CREDITS = 2


def already_captured(game_date: str) -> set:
    """
    {(home_team, away_team)} already in cache/odds_{date}.csv.

    The output file has no event_id -- it is keyed by market/player/line --
    so the game is identified by its two team names, which the API returns
    in the same spelling it writes here ("Minnesota Twins").
    """
    path = cache_path(f"odds_{game_date}")
    if not os.path.exists(path):
        return set()
    try:
        prior = pd.read_csv(path, usecols=["home_team", "away_team"])
    except Exception:
        return set()
    return set(zip(prior["home_team"].astype(str),
                   prior["away_team"].astype(str)))


def fetch_slate_odds(game_date: str, markets: str = DEFAULT_MARKET,
                     regions: str = DEFAULT_REGIONS,
                     verbose: bool = True,
                     refresh: bool = False) -> pd.DataFrame:
    """
    Fetch, de-vig and cache one slate's prices.

    INCREMENTAL BY DEFAULT. A game already present in cache/odds_{date}.csv
    is skipped, because the player-props endpoint is charged PER EVENT and
    has no idea it already sold you that game -- a second pull on a
    fifteen-game slate used to cost fifteen more credits and re-buy nine
    games whose price had barely moved.

    On the free tier that is the difference between ~33 nights of season
    and ~16, which is the difference between benchmarking the model and
    not.

    `refresh=True` re-buys everything at full price. That is the right
    call when you deliberately want a later, stronger closing line on
    games you priced early -- the earlier capture really is the weaker
    number. It is just no longer what happens by accident.
    """
    api_key = load_api_key()
    events = list_events(api_key, game_date)
    if not events:
        print(f"  No events found for {game_date}. Nothing fetched, "
              f"no credits spent.")
        return pd.DataFrame()

    # Games already underway are dropped BEFORE any request is made.
    #
    # The player-props endpoint is charged per event, so a started game
    # costs a credit to fetch and the line it returns is unusable -- an
    # in-play or pulled price that cannot be a closing line. Skipping it
    # is the same exclusion, and it keeps the credit for a night when the
    # game has not started. Everything left in odds_{date}.csv is
    # pre-game by construction.
    #
    # This is the one input that cannot be re-fetched. Predictions are
    # protected after the fact by preserve_committed_rows; odds have no
    # equivalent, so the guard has to be here.
    now = pd.Timestamp.utcnow()
    started = [e for e in events
               if pd.to_datetime(e.get("commence_time"), utc=True) <= now]
    events = [e for e in events if e not in started]
    if started and verbose:
        print(f"  Skipping {len(started)} game(s) already underway -- their "
              f"prices are no longer pre-game. {len(started)} credit(s) not "
              f"spent.")
    if not events:
        print(f"  Every game on {game_date} has started. Nothing fetched, "
              f"no credits spent.")
        return pd.DataFrame()

    # Games already priced. See the docstring -- this is the guard that
    # decides how many nights of the season get benchmarked at all.
    cached = set() if refresh else already_captured(game_date)
    seen = [e for e in events
            if (str(e.get("home_team")), str(e.get("away_team"))) in cached]
    events = [e for e in events if e not in seen]
    if seen and verbose:
        print(f"  Skipping {len(seen)} game(s) already in "
              f"cache/odds_{game_date}.csv -- {len(seen)} credit(s) not "
              f"spent. Use --refresh to re-buy them at a later line.")
    if not events:
        print(f"  Every game on {game_date} is either underway or already "
              f"priced. Nothing fetched, no credits spent.")
        return pd.DataFrame()

    n_markets = len([m for m in markets.split(",") if m.strip()])
    n_regions = len([r for r in regions.split(",") if r.strip()])
    if verbose:
        print(f"  {len(events)} event(s) still to start for {game_date}.")
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
    out_path = cache_path(f"odds_{game_date}")
    lines = _merge_capture(lines, out_path, ["market", "player", "line"],
                           verbose)
    lines.to_csv(out_path, index=False)

    log_credits(game_date, "props", len(events), len(started) + len(seen),
                remaining, verbose)
    if verbose:
        print(f"\n  {len(lines)} lines from {raw['bookmaker'].nunique()} "
              f"bookmakers. Median hold {raw['hold'].median():.3f}.")
        print(f"  Credits remaining: {remaining}")
        print(f"  Saved to cache/odds_{game_date}.csv")
    return lines


# ---------------------------------------------------------------------
# Game lines: moneyline and total, for the whole slate at once
# ---------------------------------------------------------------------
#
# The featured-markets endpoint returns EVERY game in one response and
# is charged per market per region, not per event:
#
#     h2h + totals, one region  =  2 credits for the whole slate
#
# against ~30 for one night of pitcher strikeouts. This is what makes
# the team model gradeable against the market for nearly nothing. Cached
# to cache/gamelines_{date}.csv with one row per game: de-vigged home win
# probability, the consensus total, and the de-vigged over probability at
# that total.
GAME_LINE_MARKETS = "h2h,totals"


def fetch_game_lines(game_date: str, regions: str = DEFAULT_REGIONS,
                     verbose: bool = True) -> pd.DataFrame:
    """Moneylines and totals for one date. Costs 2 credits."""
    api_key = load_api_key()
    url = (f"{BASE}/sports/{SPORT}/odds?apiKey={api_key}&regions={regions}"
           f"&markets={GAME_LINE_MARKETS}&oddsFormat=american")
    payload, headers = _get(url)
    if not payload:
        print("  No game lines came back.")
        return pd.DataFrame()

    target = pd.Timestamp(game_date).date()
    now = pd.Timestamp.utcnow()
    rows, in_play = [], 0
    for event in payload:
        start = pd.to_datetime(event.get("commence_time"), utc=True)
        if start.tz_convert("America/New_York").date() != target:
            continue
        # Same pre-game rule the props path enforces, applied here too.
        #
        # It could not be applied the same WAY: this endpoint is one call
        # for the whole slate, so a started game cannot be skipped to save
        # a credit -- the two are already paid for. But its price still has
        # to be thrown away, because the feed keeps returning a game after
        # first pitch with LIVE odds attached, and a live moneyline is not
        # a forecast. On 2026-09-07 that produced a 0.967 and a 0.029:
        # teams four runs up and four runs down, priced as though somebody
        # had predicted it.
        #
        # Left in, those rows overwrote the pre-game capture on merge and
        # gave the market a Brier of 0.137 on 2026-09-06 -- better than any
        # book has ever priced baseball, and only because the games were
        # already decided. The model was then graded against that.
        if start <= now:
            in_play += 1
            continue
        home, away = event.get("home_team"), event.get("away_team")
        for book in event.get("bookmakers") or []:
            h2h, totals = {}, {}
            for market in book.get("markets") or []:
                for outcome in market.get("outcomes") or []:
                    name = outcome.get("name")
                    if market.get("key") == "h2h":
                        h2h[name] = american_to_prob(outcome.get("price"))
                    elif market.get("key") == "totals":
                        totals.setdefault(outcome.get("point"), {})[
                            (name or "").lower()] = american_to_prob(outcome.get("price"))
            if home in h2h and away in h2h:
                rows.append({"event_id": event.get("id"),
                             "commence_time": event.get("commence_time"),
                             "home_team": home, "away_team": away,
                             "bookmaker": book.get("key"), "market": "h2h",
                             "line": np.nan,
                             "prob": devig_two_way(h2h[home], h2h[away]),
                             "hold": h2h[home] + h2h[away] - 1.0})
            for point, sides in totals.items():
                if point is None or "over" not in sides or "under" not in sides:
                    continue
                rows.append({"event_id": event.get("id"),
                             "commence_time": event.get("commence_time"),
                             "home_team": home, "away_team": away,
                             "bookmaker": book.get("key"), "market": "totals",
                             "line": float(point),
                             "prob": devig_two_way(sides["over"], sides["under"]),
                             "hold": sides["over"] + sides["under"] - 1.0})
    if not rows:
        print(f"  No two-sided game lines for {game_date}.")
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    ml = (raw[raw["market"] == "h2h"]
          .groupby("event_id", as_index=False)
          .agg(home_team=("home_team", "first"), away_team=("away_team", "first"),
               commence_time=("commence_time", "first"),
               home_win_prob_market=("prob", "median"),
               ml_books=("bookmaker", "nunique"), ml_hold=("hold", "median")))
    tot = raw[raw["market"] == "totals"]
    if not tot.empty:
        # The consensus total is the most-quoted line; the over probability
        # is the median across books AT that line, so a book hanging 8.5
        # while the rest are at 9 does not pull the number sideways.
        main_line = (tot.groupby("event_id")["line"]
                        .agg(lambda s: s.value_counts().idxmax()).rename("total_line"))
        tot = tot.join(main_line, on="event_id")
        tot = tot[tot["line"] == tot["total_line"]]
        tot = (tot.groupby("event_id", as_index=False)
                  .agg(total_line=("line", "first"),
                       over_prob_market=("prob", "median"),
                       total_books=("bookmaker", "nunique")))
        ml = ml.merge(tot, on="event_id", how="left")
    ml["game_date"] = game_date
    ml["fetched_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if in_play and verbose:
        print(f"  Discarded {in_play} game(s) already underway -- those "
              f"prices are live, not pre-game. Any earlier capture of them "
              f"is kept below.")
    gl_path = cache_path(f"gamelines_{game_date}")
    ml = _merge_capture(ml, gl_path, ["event_id"], verbose)
    ml.to_csv(gl_path, index=False)
    log_credits(game_date, "gamelines", len(ml), in_play,
                headers.get("x-requests-remaining"), verbose)
    if verbose:
        print(f"  {len(ml)} games. Home favourites: "
              f"{(ml['home_win_prob_market'] > 0.5).mean():.0%}. "
              f"Credits remaining: {headers.get('x-requests-remaining')}")
        print(f"  Saved to cache/gamelines_{game_date}.csv")
    return ml


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        raise SystemExit("Usage: python -m data.odds_lines YYYY-MM-DD "
                         "[market | gamelines]")
    argv = [a for a in sys.argv[2:] if a != "--refresh"]
    refresh = "--refresh" in sys.argv
    market = argv[0] if argv else DEFAULT_MARKET
    if market == "gamelines":
        # Game lines are charged per MARKET for the whole slate, not per
        # event, so there is no per-game saving to make here -- the two
        # credits buy every game whether you want them all or not.
        fetch_game_lines(date)
        raise SystemExit(0)
    if market not in KNOWN_MARKETS:
        print(f"  Note: '{market}' is not one of {KNOWN_MARKETS}. "
              f"Sending it anyway.")
    fetch_slate_odds(date, markets=market, refresh=refresh)
