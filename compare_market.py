"""
The model against the closing line.

    python compare_market.py 2026-08-31

WHY THIS IS THE MEASUREMENT THAT MATTERS
-----------------------------------------
Every skill number in this project is scored against "quote the league
base rate for everyone." A sportsbook clears that bar easily, so +1.5%
Brier skill answers *is the model better than nothing* and says nothing
about *is the model any good*.

This asks the second question. Two things come out of it:

  BEFORE the games -- where the model and the market disagree most. That
  list is a curriculum. The market has the injuries, the bullpen usage,
  the weather at gametime and every professional's opinion priced in, so
  the places it disagrees with the model are the places the model is
  missing something, delivered in exactly the spots where it costs money.

  AFTER the games -- whose numbers were better, scored head to head on
  the same events. That is the honest answer, and it can be brutal. A
  model that loses to the closing line is not worthless; it is a model
  with no edge, which is a different and more useful thing to know than
  "+1.5% versus the base rate".

THE JOIN IS THE DANGEROUS PART
-------------------------------
The odds feed says "Tampa Bay Rays" and this project says "TB". It says
"George Kirby" and the slate says "George Kirby" -- usually. Accented
names have already broken this project once (Yordan Alvarez cached as
"alvarez"), and a name join that silently matches nothing produces a
clean, empty, entirely wrong comparison.

So the match rate is reported loudly and a low one is treated as a
failure, not as a result. Every unmatched name is printed. If the tables
disagree about who is pitching, that is worth seeing.

THE MARKET'S LINE IS NOT YOUR LINE
-----------------------------------
Books set a line per pitcher -- 3.5 for a spot starter, 6.5 for an ace.
Comparing the model's 5.5 number against a 4.5 market line is not a
comparison. predict_slate.py therefore stores the full strikeout
distribution as `k_dist`, and the model probability is read off at
whatever line the market actually posted.
"""
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

from data.cache import cache_path

# The odds feed uses full club names; everything else here uses the
# abbreviations Statcast reports. Thirty entries, written out rather than
# fuzzy-matched, because "Athletics" versus "Oakland Athletics" versus
# "ATH" is exactly the kind of thing a clever matcher gets wrong once a
# season and never announces.
TEAM_ABBREV = {
    "Arizona Diamondbacks": "AZ", "Athletics": "ATH",
    "Oakland Athletics": "ATH", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# A disagreement below this is not interesting. Books and models will
# always differ by a point or two on noise alone.
NOTABLE_EDGE = 0.05


def normalise_name(name) -> str:
    """
    A name reduced to something two sources can agree on.

    Strips accents, punctuation and generational suffixes. "Yordan
    Álvarez", "Yordan Alvarez" and "Yordan Alvarez Jr." all become
    "yordan alvarez". This is deliberately blunt: the alternative is
    fuzzy matching, which succeeds on the wrong player often enough to be
    worse than failing loudly.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace(".", "").replace("'", "").replace("-", " ")
    parts = [p for p in text.split() if p not in ("jr", "sr", "ii", "iii", "iv")]
    return " ".join(parts)


def prob_over_from_dist(dist_string, line: float) -> float:
    """Model probability at an arbitrary line, from the stored distribution."""
    if not isinstance(dist_string, str) or not dist_string:
        return float("nan")
    dist = np.fromstring(dist_string, sep=",")
    if dist.size == 0:
        return float("nan")
    first = int(np.floor(line)) + 1
    return float(dist[first:].sum()) if first < dist.size else 0.0


def load_market(game_date: str) -> pd.DataFrame:
    path = cache_path(f"odds_{game_date}")
    if not os.path.exists(path):
        raise SystemExit(
            f"No market lines at cache/odds_{game_date}.csv.\n"
            f"  Capture them BEFORE first pitch: "
            f"python -m data.odds_lines {game_date}\n"
            f"  They cannot be fetched afterwards on the free tier.")
    market = pd.read_csv(path)
    market["key"] = market["player"].map(normalise_name)
    return market


def compare_pitchers(game_date: str, verbose: bool = True) -> pd.DataFrame:
    """Model against market on starter strikeouts."""
    props_path = cache_path(f"pitchers_{game_date}")
    if not os.path.exists(props_path):
        raise SystemExit(
            f"No model predictions at cache/pitchers_{game_date}.csv -- "
            f"run predict_slate.py {game_date} first.")
    props = pd.read_csv(props_path)
    market = load_market(game_date)
    market = market[market["market"] == "pitcher_strikeouts"]
    if market.empty:
        print("  No pitcher strikeout lines in the market file.")
        return pd.DataFrame()

    props["key"] = props["pitcher"].map(normalise_name)
    merged = market.merge(
        props[["key", "pitcher", "team", "opponent", "expected_k",
               "expected_bf", "starts_seen", "k_dist"]],
        on="key", how="left", suffixes=("_mkt", ""))

    matched = merged["k_dist"].notna()
    if verbose:
        print(f"\n  Market lines: {len(merged)}   matched to a model "
              f"prediction: {int(matched.sum())}")
        missing = merged.loc[~matched, "player"].tolist()
        if missing:
            # Printed, not counted. A pitcher the model never predicted is
            # usually a late scratch or a name spelled differently, and
            # both are worth seeing rather than silently dropping.
            print(f"  Not matched ({len(missing)}): {', '.join(missing[:12])}"
                  f"{' ...' if len(missing) > 12 else ''}")
        if matched.sum() == 0:
            print("\n  NOTHING MATCHED. That is a join failure, not a "
                  "result -- check the name formats in both files before "
                  "reading anything into this.")
            return pd.DataFrame()
        if matched.mean() < 0.5:
            print(f"  WARNING: under half the market matched. Treat the "
                  f"comparison below as a sample, not as the slate.")

    out = merged[matched].copy()
    out["model_prob"] = [prob_over_from_dist(d, l)
                         for d, l in zip(out["k_dist"], out["line"])]
    out["market_prob"] = out["prob_over"]
    out["edge"] = out["model_prob"] - out["market_prob"]
    out["side"] = np.where(out["edge"] > 0, "OVER", "UNDER")
    return out.drop(columns=["k_dist"]).sort_values(
        "edge", key=abs, ascending=False)


def report(game_date: str, top_n: int = 5):
    out = compare_pitchers(game_date)
    if out.empty:
        return

    print("\n" + "=" * 74)
    print(f"MODEL vs MARKET -- starter strikeouts, {game_date}")
    print("=" * 74)
    view = out[["player", "team", "line", "model_prob", "market_prob",
                "edge", "n_books", "starts_seen"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n  Mean absolute disagreement: {out['edge'].abs().mean():.3f}")
    print(f"  Mean signed disagreement:   {out['edge'].mean():+.3f}", end="")
    # A signed average far from zero means the model is systematically
    # higher or lower than the market on this prop -- a bias, not an edge.
    if abs(out["edge"].mean()) > 0.03:
        print("   <- systematically "
              f"{'HIGH' if out['edge'].mean() > 0 else 'LOW'} versus the "
              f"market. That is a calibration problem, not an edge.")
    else:
        print()

    big = out[out["edge"].abs() >= NOTABLE_EDGE].head(top_n)
    print("\n" + "-" * 74)
    print(f"THE {len(big)} BIGGEST DISAGREEMENTS -- the curriculum")
    print("-" * 74)
    if big.empty:
        print("  Nothing over "
              f"{NOTABLE_EDGE:.0%}. The model and the market agree tonight.")
    else:
        for r in big.to_dict("records"):
            print(f"\n  {r['player']} ({r['team']}) over {r['line']}")
            print(f"    model {r['model_prob']:.1%}  vs  market "
                  f"{r['market_prob']:.1%}   -> model likes the "
                  f"{r['side']} by {abs(r['edge']):.1%}")
            print(f"    expected {r['expected_k']:.1f} K over "
                  f"{r['expected_bf']:.1f} batters, {int(r['starts_seen'])} "
                  f"starts of history, {int(r['n_books'])} book(s) quoting")
        print("\n  For each of these, ask what the market knows that the")
        print("  model does not. Injury, bullpen plan, weather, a call-up.")
        print("  That question is the fastest way to learn what is missing.")

    path = cache_path(f"market_compare_{game_date}")
    out.to_csv(path, index=False)
    print(f"\n  Saved to cache/market_compare_{game_date}.csv")
    print("\n  This is the PRE-GAME view. Whose numbers were actually")
    print("  better is answered by score_slate.py after the games.")


MARKET_LOG_KEY = "market_log"


def score_against_market(game_date: str, actuals: pd.DataFrame,
                         verbose: bool = True):
    """
    Head to head, after the games: model Brier versus market Brier.

    `actuals` needs pitcher_id or pitcher, plus strikeouts. score_slate.py
    has exactly that after joining the official pitching lines.

    THIS IS THE NUMBER. Everything else in the project is scored against
    the league base rate, which a sportsbook beats without trying. Scored
    against the closing line, on the same events, at the same lines, there
    is nowhere to hide:

        model Brier LOWER than market   ->  a real edge, on this sample
        roughly equal                   ->  reproducing public information
        model Brier HIGHER              ->  the market knows more

    None of those is a bad outcome to learn. The third one is only bad if
    you keep believing the first.

    A slate is a dozen pitchers, so one night decides nothing and the
    running total in cache/market_log.csv is what to read.
    """
    props_path = cache_path(f"pitchers_{game_date}")
    odds_path = cache_path(f"odds_{game_date}")
    if not (os.path.exists(props_path) and os.path.exists(odds_path)):
        return None

    props = pd.read_csv(props_path)
    if "k_dist" not in props.columns:
        if verbose:
            print("\n  Market comparison needs `k_dist`, which this slate "
                  "file predates. Re-run predict_slate to enable it.")
        return None

    market = load_market(game_date)
    market = market[market["market"] == "pitcher_strikeouts"]
    if market.empty:
        return None

    props["key"] = props["pitcher"].map(normalise_name)
    truth = actuals.copy()
    truth["key"] = truth["pitcher"].map(normalise_name)

    df = (market.merge(props[["key", "k_dist"]], on="key", how="inner")
                .merge(truth[["key", "strikeouts"]], on="key", how="inner"))
    if df.empty:
        if verbose:
            print("\n  No pitcher had a market line, a model prediction and "
                  "an official line all three. Nothing to compare.")
        return None

    df["model_prob"] = [prob_over_from_dist(d, l)
                        for d, l in zip(df["k_dist"], df["line"])]
    df["hit"] = (df["strikeouts"] > df["line"]).astype(int)
    df["brier_model"] = (df["model_prob"] - df["hit"]) ** 2
    df["brier_market"] = (df["prob_over"] - df["hit"]) ** 2
    df = df.dropna(subset=["model_prob", "prob_over"])
    if df.empty:
        return None

    entry = pd.DataFrame([{
        "game_date": game_date, "n": len(df),
        "brier_model": float(df["brier_model"].mean()),
        "brier_market": float(df["brier_market"].mean()),
        "mean_model": float(df["model_prob"].mean()),
        "mean_market": float(df["prob_over"].mean()),
        "base_rate": float(df["hit"].mean()),
    }])

    log_path = cache_path(MARKET_LOG_KEY)
    if os.path.exists(log_path):
        previous = pd.read_csv(log_path)
        previous = previous[previous["game_date"] != game_date]
        log = pd.concat([previous, entry], ignore_index=True)
    else:
        log = entry
    log.to_csv(log_path, index=False)

    if verbose:
        w = log["n"].to_numpy(dtype=float)
        bm = float(np.average(log["brier_model"], weights=w))
        bk = float(np.average(log["brier_market"], weights=w))
        print("\n" + "=" * 72)
        print("MODEL vs MARKET -- the benchmark that counts")
        print("=" * 72)
        print(f"  Tonight ({len(df)} pitchers):  model "
              f"{entry['brier_model'].iloc[0]:.4f}   market "
              f"{entry['brier_market'].iloc[0]:.4f}")
        print(f"  Running ({int(w.sum())} pitchers over "
              f"{log['game_date'].nunique()} slates): "
              f"model {bm:.4f}   market {bk:.4f}")
        gap = bm - bk
        # Lower Brier is better, so a NEGATIVE gap is the model winning.
        if gap < -0.005:
            print(f"\n  Model is AHEAD by {-gap:.4f}. On this sample the "
                  f"model beats the closing line.")
        elif gap > 0.005:
            print(f"\n  Market is ahead by {gap:.4f}. No edge here yet -- "
                  f"which is the normal result and worth knowing.")
        else:
            print(f"\n  Within {abs(gap):.4f} of the market. That is "
                  f"reproducing public information, not beating it.")
        print(f"\n  {int(w.sum())} pitchers is a small sample; a dozen "
              f"slates before this means much.")
        print(f"  Log: cache/{MARKET_LOG_KEY}.csv")
    return entry


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python compare_market.py YYYY-MM-DD")
    report(sys.argv[1])
