"""
A damaged cache must not take down a run, and a write must not leave one.

WHY
---
On 2026-09-24 score_slate died on a single bad row in one of 539 player
caches. The protection against exactly that already existed in
refresh.load_player_cache (`errors="coerce"` then `dropna`) but sat one
frame BELOW the strict parse in cache.load_cached, so it could never run.

Both properties below are the kind that fail silently once someone
"simplifies" them, so each is asserted directly rather than inferred from
a run completing.

Nothing here touches the real cache/ directory.
"""
import os
import sys
import shutil
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ok = True


def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok  ' if good else 'FAIL'}] {label:56} "
          f"got {got!r}, want {want!r}")


# Point the cache module at a scratch directory before anything else
# imports it, so no test can touch the real 1.2 GB of caches.
SCRATCH = tempfile.mkdtemp(prefix="cachetest_")
import data.cache as C
C.CACHE_DIR = SCRATCH

GOOD = ("pitch_type,game_date,player_name,batter,game_pk,at_bat_number,"
        "pitch_number\n")
ROW = 'FF,2026-09-{d:02d},"Alvarez, Yordan",514888,70000{d},5,{p}\n'


def write(key, text):
    with open(C.cache_path(key), "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


print("=" * 72)
print("PART 1 -- one damaged row must not raise")
print("=" * 72)

# The real shape: a chunk of bytes vanished from the start of one row, so
# fields shifted left and something that is not a date landed in the
# game_date column. This is 672640's damage -- the surviving fragment is a
# number, so the bad row costs exactly itself.
write("damaged",
      GOOD
      + ROW.format(d=20, p=1)
      + '0.35,1.08,0.94,514888,700021,5,2\n'      # shifted row
      + ROW.format(d=22, p=1)
      + ROW.format(d=23, p=1))

df = C.load_cached("damaged")
check("returns a frame instead of raising", df is not None, True)
check("keeps every good row", len(df), 3)
check("drops the damaged one", int(df["game_date"].isna().sum()), 0)
check("dates are real datetimes",
      str(df["game_date"].dtype).startswith("datetime64"), True)

# 670541's damage is worse and the test found out why: its bad row begins
# with a lone `"`, which opens a quoted field that runs on until the next
# quote character -- swallowing the FOLLOWING row as well. So one lost
# chunk can cost two rows, and a gap scan is the only thing that sees the
# second one. All load_cached owes us here is that it survives and says so.
write("quote_damaged",
      GOOD
      + ROW.format(d=20, p=1)
      + '",514888,700020,5,2\n'          # <- the row that killed score_slate
      + ROW.format(d=22, p=1))

df = C.load_cached("quote_damaged")
check("a quote-leading break does not raise either", df is not None, True)
check("and leaves no unparseable dates behind",
      int(df["game_date"].isna().sum()), 0)
check("but it costs MORE than its own row", len(df) < 3, True)

# A clean file must be untouched and must not print a warning.
write("clean", GOOD + ROW.format(d=20, p=1) + ROW.format(d=22, p=1))
df = C.load_cached("clean")
check("a clean file keeps every row", len(df), 2)

check("a missing file is still None", C.load_cached("nope") is None, True)

# A file with no game_date column at all must still load.
write("nodate", "a,b\n1,2\n")
check("no game_date column is fine", len(C.load_cached("nodate")), 1)

print()
print("=" * 72)
print("PART 2 -- a write is all-or-nothing")
print("=" * 72)

frame = pd.DataFrame({"game_date": ["2026-09-20", "2026-09-22"],
                      "player_name": ["Alvarez, Yordan", "Diaz, Yainer"]})
C.save_cache("written", frame)
back = C.load_cached("written")
check("round-trips every row", len(back), 2)
check("a comma inside a name survives",
      str(back["player_name"].iloc[0]), "Alvarez, Yordan")

# The failure that matters: if writing blows up partway, the file already
# on disk must still be the old complete one, not a truncated new one.
C.save_cache("existing", frame)
before = open(C.cache_path("existing"), encoding="utf-8").read()


class Exploding(pd.DataFrame):
    """A frame that raises while pandas is serialising it."""
    @property
    def _constructor(self):
        return Exploding

    def to_csv(self, *a, **k):
        raise RuntimeError("disk died mid-write")


try:
    C.save_cache("existing", Exploding(frame))
    raised = False
except RuntimeError:
    raised = True

check("a failed write re-raises", raised, True)
after = open(C.cache_path("existing"), encoding="utf-8").read()
check("the old file is intact, not half-replaced", after, before)
check("no .tmp litter left behind",
      [f for f in os.listdir(SCRATCH) if ".tmp." in f], [])

shutil.rmtree(SCRATCH, ignore_errors=True)

print()
print("=" * 72)
print("ALL PASS" if ok else "FAILURES ABOVE")
print("=" * 72)
sys.exit(0 if ok else 1)
