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


def credit_report(game_date: str):
    """
    What tonight cost, what the month has cost, and how long that lasts.

    The free tier is 500 credits a month and player props are charged PER
    EVENT, so a fifteen-game slate is fifteen credits and a second capture
    the same night is fifteen MORE -- the endpoint has no idea it already
    sold you that game. Two captures a night is therefore roughly 16
    nights of season, not 33, and that arithmetic used to live only in a
    docstring where it could not warn anyone.
    """
    path = os.path.join(CACHE, "credit_log.csv")
    if not os.path.exists(path):
        return
    try:
        log = pd.read_csv(path)
    except Exception:
        return
    if log.empty:
        return
    log["fetched_at_utc"] = pd.to_datetime(log["fetched_at_utc"],
                                           errors="coerce", utc=True)

    print(f"\n{'=' * 72}\nCREDITS\n{'=' * 72}")

    tonight = log[log["game_date"].astype(str) == game_date]
    if len(tonight):
        spent = int(tonight["credits_charged"].sum())
        saved = int(tonight["credits_saved"].sum())
        calls = len(tonight)
        # "saved" now covers two reasons -- already underway, and already
        # priced -- and this row cannot tell them apart, so it does not
        # claim to. odds_lines prints the breakdown as it happens.
        print(f"  This slate     {spent:>4} spent"
              + (f" · {saved} saved (already underway or already priced)"
                 if saved else "")
              + (f" · across {calls} captures" if calls > 1 else ""))
        if calls > 1:
            # The endpoint charges per event every time and has no idea it
            # already sold you a game, so odds_lines skips games already
            # in odds_{date}.csv. Worth saying, because the saving is
            # invisible otherwise.
            print(f"                  later pulls skipped games already "
                  f"priced — pass --refresh to re-buy them at a later line")

    now = pd.Timestamp.utcnow()
    month = log[log["fetched_at_utc"].dt.strftime("%Y-%m")
                == now.strftime("%Y-%m")]
    if len(month):
        used = int(month["credits_charged"].sum())
        print(f"  This month     {used:>4} spent over "
              f"{month['game_date'].nunique()} slate(s)")

    # The API's own header beats anything counted up here, and it is the
    # only figure that survives a cache wipe.
    left = log["credits_remaining"].dropna()
    if len(left):
        remaining = int(left.iloc[-1])
        print(f"  Plan says      {remaining:>4} left")
        if len(month) and month["game_date"].nunique():
            per_night = used / month["game_date"].nunique()
            if per_night > 0:
                print(f"  At {per_night:.0f} a night  "
                      f"{remaining / per_night:>4.0f} more nights")


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
    _run("1/6  Lineups", ["check_lineups.py", game_date], args.dry_run)

    # 2. Predictions. Free. MUST come before compare_market, which reads
    #    the pitcher file this writes.
    if not _run("2/6  Predictions", ["predict_slate.py", game_date],
                args.dry_run):
        print("\n  Stopping: predictions failed, so there is nothing for the "
              "market comparison to compare against. No credits were spent.")
        return 1

    # 3. Recent form for tonight's starters. Free, local, and it has to
    #    run HERE rather than in the dashboard: it reads the statcast
    #    pitcher caches, which are 1.2 GB and permanently untracked, so
    #    the deployed app can never compute it. ~25 KB a night.
    #
    #    After predict_slate because it reads pitchers_{date}.csv to know
    #    who is starting. A failure is not fatal -- the Pitchers tab says
    #    so and shows everything else.
    _run("3/6  Recent form", ["pitcher_form.py", game_date], args.dry_run)

    if args.no_odds:
        # Slips still get committed. They lose only their pitcher leg --
        # an arm needs a captured line to have a measured edge against --
        # and a slate with no committed slips cannot be graded at all,
        # which is a worse outcome than one graded without arms.
        _run("6/6  Commit slips (no arm legs without odds)",
             ["slips.py", game_date], args.dry_run)
        print(f"\n{'=' * 72}\n  --no-odds: skipped both odds calls and "
              f"compare_market. Nothing spent.\n{'=' * 72}")
        return 0

    # 4. The only step that costs anything.
    #    Incremental: a game already in odds_{date}.csv is skipped, so a
    #    second run tonight buys only what is new. `--refresh` on
    #    data.odds_lines re-buys everything at a later line.
    _run("4/6  Strikeout props (1 credit per game not yet priced)",
         ["-m", "data.odds_lines", game_date], args.dry_run)

    # 5. Free, local, and the step that was running too early.
    _run("5/6  Model vs market", ["compare_market.py", game_date],
         args.dry_run)

    # 6. Commit tonight's slips. LAST because it reads the odds: a
    #    pitcher leg needs a captured line to have a measured edge
    #    against, and the arm is the only leg on the board with one.
    #
    #    This is the step that makes "which slips hit" answerable. The
    #    Bet ready tab used to build slips at render time, so they
    #    existed only as pixels and could never be graded -- rebuilding
    #    them the next morning would score today's code against last
    #    night's games. Same rule as predictions and odds: commit before
    #    first pitch or it is not evidence.
    _run("6/6  Commit slips", ["slips.py", game_date], args.dry_run)

    if not args.dry_run:
        print(f"\n{'=' * 72}\nWROTE\n{'=' * 72}")
        for name in (f"slate_{game_date}", f"pitchers_{game_date}",
                     f"teams_{game_date}", f"pitcher_form_{game_date}",
                     f"odds_{game_date}", f"market_compare_{game_date}",
                     f"slips_{game_date}"):
            path = os.path.join(CACHE, f"{name}.csv")
            if os.path.exists(path):
                stamp = pd.Timestamp(os.path.getmtime(path), unit="s",
                                     tz="UTC").strftime("%H:%M")
                print(f"  cache/{name}.csv  ({stamp} UTC)")
            else:
                print(f"  cache/{name}.csv  -- not written")
        credit_report(game_date)
        # One command. score_slate grades the slips as steps 6-10 now,
        # reusing the Statcast refresh it already did rather than making
        # a second identical one.
        print("\n  After the games: python score_slate.py " + game_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
