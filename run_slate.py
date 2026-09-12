"""
One command for a slate, in the order the steps actually depend on.

    python run_slate.py                 # today
    python run_slate.py 2026-09-06
    python run_slate.py --no-odds       # free: lineups + predictions only
    python run_slate.py --dry-run       # say what would run, spend nothing

WHY THIS EXISTS
---------------
The five commands were being run in an order that cannot work:

    odds_lines
    compare_market        <- reads cache/pitchers_{date}.csv
    predict_slate         <- WRITES cache/pitchers_{date}.csv
    check_lineups

compare_market ran against whichever pitcher file happened to be on disk,
which on 2026-09-06 was the previous night's. It produced no output and no
market_log row, and nothing said so -- the failure looked like a quiet
no-op rather than an error. Ordering that a person has to remember is
ordering that will be got wrong on the morning it matters.

THE CREDIT BUDGET IS WHY THIS IS ONE SHOT
-----------------------------------------
The Odds API free tier is 500 credits a month. Player props are charged
PER EVENT (15 a night), game lines are charged per market for the whole
slate (2 a night). Seventeen a night is 29 nights of a 500 budget, so
there is no second capture to fall back on.

Everything else is free. check_lineups makes one schedule call.
predict_slate is Statcast and models, no odds quota at all. compare_market
reads local files. So the pattern that fits the budget is:

    poll check_lineups (free)  ->  predict_slate as often as you like
    (free)  ->  ONE odds capture, as late as you can before the earliest
    first pitch  ->  compare_market (free)

FIRST PITCH IS THE DEADLINE, NOT THE AVERAGE START
--------------------------------------------------
Predictions are protected after the fact -- preserve_committed_rows keeps
the number committed to before a game began. Odds have no equivalent and
cannot be re-fetched, so a game already underway is a game whose price is
lost. data/odds_lines skips those before spending the credit; this script
prints how many there are before anything runs, so the cost of being late
is visible while it can still be acted on.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Game lines are no longer bought.
#
# Two credits a night for the moneyline and total, against a team model
# that graded out at a coin flip on winners (home-win Brier 0.237-0.265
# versus 0.25) and half a run worse than the market on totals. Paying to
# benchmark a model nobody reads is the wrong two credits. The fetcher
# stays in data/odds_lines -- `python -m data.odds_lines DATE gamelines`
# still works by hand if the team model ever earns the check.
CREDITS_GAMELINES = 0


def _run(label: str, args: list, dry: bool) -> bool:
    """One step. Returns True if it succeeded (or was a dry run)."""
    printable = " ".join([os.path.basename(sys.executable)] + args)
    print(f"\n{'=' * 72}\n{label}\n  $ {printable}\n{'=' * 72}")
    if dry:
        print("  (dry run -- not executed)")
        return True
    result = subprocess.run([sys.executable] + args)
    if result.returncode != 0:
        print(f"\n  !! {label} exited {result.returncode}.")
    return result.returncode == 0


def slate_clock(game_date: str):
    """
    (games, started, earliest) from the schedule this project already has.

    Read from the existing slate file rather than a fresh API call: this
    runs before predict_slate on purpose, so the numbers describe the
    slate as last written. When there is no file yet it returns None and
    the caller simply does not print the warning.
    """
    path = os.path.join(CACHE, f"slate_{game_date}.csv")
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path, usecols=["game_pk", "start_time_utc"])
    except Exception:
        return None
    starts = (frame.drop_duplicates("game_pk")["start_time_utc"]
              .pipe(pd.to_datetime, utc=True, errors="coerce").dropna())
    if starts.empty:
        return None
    now = pd.Timestamp.utcnow()
    return len(starts), int((starts <= now).sum()), starts.min()


def main():
    parser = argparse.ArgumentParser(
        description="Run a slate end to end, in dependency order.")
    parser.add_argument("date", nargs="?", default=None,
                        help="YYYY-MM-DD (default: today, local)")
    parser.add_argument("--no-odds", action="store_true",
                        help="skip both odds calls and compare_market; "
                             "spends no credits")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without running anything")
    args = parser.parse_args()
    game_date = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"SLATE {game_date}")

    clock = slate_clock(game_date)
    if clock:
        games, started, earliest = clock
        local = earliest.tz_convert("America/Los_Angeles")
        print(f"  {games} games, earliest first pitch "
              f"{local.strftime('%I:%M %p')} PT "
              f"({earliest.strftime('%H:%M')} UTC).")
        if started:
            # Said before anything runs, because the only thing that can be
            # done about it is done by starting earlier tomorrow.
            print(f"  {started} of {games} already underway. Their odds are "
                  f"gone -- odds_lines will skip them and keep the credits. "
                  f"The other {games - started} are unaffected.")
        else:
            print(f"  Nothing has started. Full slate available.")

    if not args.no_odds:
        remaining = (clock[0] - clock[1]) if clock else "?"
        print(f"\n  Odds cost this run: ~{remaining} credit(s), all player "
              f"props. Free tier is 500 a month.")

    # 1. Lineups. Free, one second, and the thing worth knowing before
    #    committing a prediction: projected slots and confirmed slots give
    #    materially different plate appearances.
    _run("1/4  Lineups", ["check_lineups.py", game_date], args.dry_run)

    # 2. Predictions. Free. MUST come before compare_market, which reads
    #    the pitcher file this writes.
    if not _run("2/4  Predictions", ["predict_slate.py", game_date],
                args.dry_run):
        print("\n  Stopping: predictions failed, so there is nothing for the "
              "market comparison to compare against. No credits were spent.")
        return 1

    if args.no_odds:
        print(f"\n{'=' * 72}\n  --no-odds: skipped both odds calls and "
              f"compare_market. Nothing spent.\n{'=' * 72}")
        return 0

    # 3 and 4. The only steps that cost anything. Game lines first: two
    #    credits for the whole slate, and if the key or the plan is wrong
    #    it fails here having spent two rather than fifteen.
    _run("3/4  Strikeout props (1 credit per game not yet started)",
         ["-m", "data.odds_lines", game_date], args.dry_run)

    # 4. Free, local, and the step that was running too early.
    _run("4/4  Model vs market", ["compare_market.py", game_date],
         args.dry_run)

    if not args.dry_run:
        print(f"\n{'=' * 72}\nWROTE\n{'=' * 72}")
        for name in (f"slate_{game_date}", f"pitchers_{game_date}",
                     f"teams_{game_date}", f"odds_{game_date}",
                     f"market_compare_{game_date}"):
            path = os.path.join(CACHE, f"{name}.csv")
            if os.path.exists(path):
                stamp = pd.Timestamp(os.path.getmtime(path), unit="s",
                                     tz="UTC").strftime("%H:%M")
                print(f"  cache/{name}.csv  ({stamp} UTC)")
            else:
                print(f"  cache/{name}.csv  -- not written")
        print("\n  After the games: python score_slate.py " + game_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
