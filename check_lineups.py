"""
Are tonight's lineups posted yet?

    python check_lineups.py              # tomorrow
    python check_lineups.py 2026-08-18   # a specific date

Runs in about a second. One schedule call, no Statcast, no models, no
cache writes -- so you can poll it every 20 minutes without cost while
waiting to fire the real slate run.

WHY THIS EXISTS
---------------
`predict_slate.py` produces materially different numbers depending on
whether lineups are confirmed or projected, because batting-order slot
drives plate appearances and plate appearances drive everything else --
4.51 expected PA at leadoff versus 3.14 at ninth on a real slate.

MLB has no league-mandated posting time. In practice teams submit roughly
three to four hours before first pitch, but it varies by club and by
manager, and a team scratching someone late can push it. So rather than
guessing from a rule of thumb, this asks the same feed predict_slate.py
reads and reports what is actually there.

The useful workflow is: poll this until the games you care about say
CONFIRMED, then run predict_slate.py once. That produces the version worth
scoring, written with real slots instead of projected ones.
"""
import sys
from datetime import datetime, timezone

import pandas as pd

from data.schedule import get_slate, tomorrow


def fmt_clock(ts) -> str:
    """Local-style 12-hour clock without strftime's platform-specific
    zero-stripping flag (%-I on POSIX, %#I on Windows -- each errors on
    the other)."""
    if ts is None or pd.isna(ts):
        return "  --  "
    hour = ts.hour % 12 or 12
    return f"{hour}:{ts.minute:02d} {'AM' if ts.hour < 12 else 'PM'}"


def main(game_date: str = None):
    game_date = game_date or tomorrow()
    slate = get_slate(game_date, verbose=False)
    if slate.empty:
        print(f"No games scheduled for {game_date}.")
        return

    slate = slate.copy()
    slate["start"] = pd.to_datetime(slate["start_time_utc"], utc=True,
                                    errors="coerce").dt.tz_convert("America/New_York")
    slate["home_posted"] = slate["home_lineup"].apply(lambda x: bool(x))
    slate["away_posted"] = slate["away_lineup"].apply(lambda x: bool(x))
    slate = slate.sort_values("start")

    now = datetime.now(timezone.utc).astimezone(
        pd.Timestamp.now(tz="America/New_York").tzinfo)

    print(f"\n  {game_date} -- checked {fmt_clock(pd.Timestamp(now))} ET\n")
    print(f"  {'first pitch':>12}  {'matchup':<14} {'away':<10} {'home':<10} {'starters'}")
    print(f"  {'-'*12}  {'-'*14} {'-'*10} {'-'*10} {'-'*8}")

    both = 0
    for game in slate.itertuples():
        matchup = f"{game.away_team} @ {game.home_team}"
        away = "CONFIRMED" if game.away_posted else "waiting"
        home = "CONFIRMED" if game.home_posted else "waiting"
        tbd = sum(1 for p in (game.home_probable, game.away_probable) if p == "TBD")
        starters = "both named" if tbd == 0 else f"{tbd} TBD"
        if game.away_posted and game.home_posted:
            both += 1
        print(f"  {fmt_clock(game.start):>12}  {matchup:<14} {away:<10} {home:<10} {starters}")

    print(f"\n  {both} of {len(slate)} games have BOTH lineups posted.")

    if both == 0:
        earliest = slate["start"].min()
        hours = (earliest - pd.Timestamp(now)).total_seconds() / 3600.0
        print(f"  Nothing posted yet. First pitch is in {hours:.1f} hours.")
        print("  Teams typically submit 3-4 hours out, so check back around")
        print(f"  {fmt_clock(earliest - pd.Timedelta(hours=3.5))} ET.")
    elif both < len(slate):
        print("  Partial. Running now gives confirmed slots for those games and")
        print("  projected slots for the rest -- the dashboard labels which is")
        print("  which, so this is a perfectly reasonable time to run.")
    else:
        print("  All posted. Run it:")
        print(f"      python predict_slate.py {game_date}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
