"""
A slate that covers fewer games than the schedule must say so.

WHY
---
cache/slate_2026-09-21.csv held 54 hitters across 3 games where a September
slate is ~270 across 15. Every component behaved correctly -- get_slate
returned all games, preserve_committed_rows only adds, predict_slate wrote
what it could see -- and the night still graded 11 hitters. The failure was
that nothing SAID so, and it stayed invisible for three weeks.

So the assertion is on the warning firing, not on any number changing.
"""
import io
import os
import sys
import contextlib

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_slate as P

ok = True


def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok  ' if good else 'FAIL'}] {label:56} "
          f"got {got!r}, want {want!r}")


def slate_of(pks):
    return pd.DataFrame({
        "game_pk": pks,
        "home_team": [f"H{i}" for i in range(len(pks))],
        "away_team": [f"A{i}" for i in range(len(pks))],
    })


def out_of(pks, per_game=18):
    return pd.DataFrame({"game_pk": [p for p in pks for _ in range(per_game)]})


def run(out, slate, game_date):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.report_coverage(out, slate, game_date)
    return buf.getvalue()


FUTURE = (pd.Timestamp.today().normalize() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
TODAY = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

print("=" * 72)
print("PART 1 -- the real 09-21 shape")
print("=" * 72)
text = run(out_of(range(3)), slate_of(range(15)), FUTURE)
check("warns", "THIN SLATE" in text, True)
check("counts both sides", "3 of 15 scheduled games" in text, True)
check("names the missing games", "A3 @ H3" in text, True)
check("says re-running is free", "loses nothing" in text, True)
check("blames the early run when it IS early", "day(s) BEFORE" in text, True)

print()
print("=" * 72)
print("PART 2 -- it must stay quiet when there is nothing to say")
print("=" * 72)
check("full slate is silent", run(out_of(range(15)), slate_of(range(15)), TODAY), "")
check("a genuinely 3-game day is silent",
      run(out_of(range(3)), slate_of(range(3)), TODAY), "")
check("more covered than scheduled is silent",
      run(out_of(range(15)), slate_of(range(3)), TODAY), "")
check("empty predictions are silent",
      run(pd.DataFrame({"game_pk": []}), slate_of(range(15)), TODAY), "")
check("empty schedule is silent",
      run(out_of(range(3)), pd.DataFrame(), TODAY), "")
check("no schedule at all is silent", run(out_of(range(3)), None, TODAY), "")

print()
print("=" * 72)
print("PART 3 -- same-day thinness gets the other advice")
print("=" * 72)
text = run(out_of(range(9)), slate_of(range(15)), TODAY)
check("still warns", "THIN SLATE" in text, True)
check("does NOT blame an early run", "day(s) BEFORE" in text, False)
check("tells you to re-run when lineups post",
      "when the remaining lineups post" in text, True)

print()
print("=" * 72)
print("PART 4 -- a warning must never be able to stop the run")
print("=" * 72)
broken = pd.DataFrame({"game_pk": [1, 2]})          # slate with no team names
text = run(out_of(range(1)), broken, TODAY)
check("survives a schedule missing team columns", "THIN SLATE" in text, True)
try:
    run(out_of(range(1)), slate_of(range(5)), "not-a-date")
    survived = True
except Exception:
    survived = False
check("survives an unparseable date", survived, True)

print()
print("=" * 72)
print("ALL PASS" if ok else "FAILURES ABOVE")
print("=" * 72)
sys.exit(0 if ok else 1)
