"""
The hitter row log, and the two ways an append-only file goes wrong.

    python test_hitter_log.py

score_slate is safe to re-run by design -- odds arrive late, a boxscore
gets corrected, a slate gets graded twice. A log that GREW on every re-run
would silently weight those nights two or three times, and every
calibration number computed from it afterwards would be wrong in a way
nothing would flag. That is the bug this file exists to catch.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(label)


tmp = tempfile.mkdtemp()
os.environ["BP_CACHE_DIR"] = tmp
import data.cache as dc
dc.CACHE_DIR = tmp                      # redirect cache_path at the source
import score_slate as S


def frame(n=40, seed=0, date="2026-09-12"):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "player_id": range(n),
        "team": "NYY", "opponent": "BOS",
        "lineup_slot": rng.integers(1, 10, n),
        "prob_hr": rng.uniform(0.02, 0.40, n),
        "prob_hit": rng.uniform(0.3, 0.85, n),
        "got_hr": rng.integers(0, 2, n),
        "got_hit": rng.integers(0, 2, n),
        "hits": rng.integers(0, 4, n),
        "total_bases": rng.integers(0, 6, n),
    })


print("writing")
S._log_hitter_rows(frame(), "2026-09-12")
p = dc.cache_path(S.HITTER_ROW_LOG)
check("the log is created", os.path.exists(p))
d = pd.read_csv(p)
check("one row per hitter", len(d) == 40, f"({len(d)})")
check("game_date is first", list(d.columns)[0] == "game_date")
check("predictions kept", "prob_hr" in d.columns)
check("outcomes kept", "got_hr" in d.columns)

print("\nre-scoring the SAME slate")
S._log_hitter_rows(frame(seed=1), "2026-09-12")
d = pd.read_csv(p)
# The one that matters. Re-running score_slate on a night must replace that
# night, not append it again -- otherwise a twice-graded slate counts twice
# in every calibration number drawn from this file.
check("rows are replaced, not doubled", len(d) == 40, f"({len(d)})")
check("only that date present", set(d.game_date.astype(str)) == {"2026-09-12"})
check("the NEW values won", abs(d.prob_hr.iloc[0] - frame(seed=1).prob_hr.iloc[0]) < 1e-9)

print("\nadding a second slate")
S._log_hitter_rows(frame(n=25, seed=2), "2026-09-13")
d = pd.read_csv(p)
check("both nights present", len(d) == 65, f"({len(d)})")
check("two dates", sorted(set(d.game_date.astype(str))) ==
      ["2026-09-12", "2026-09-13"])

print("\nedge cases")
before = len(pd.read_csv(p))
S._log_hitter_rows(pd.DataFrame(), "2026-09-14")
check("an empty frame writes nothing", len(pd.read_csv(p)) == before)
S._log_hitter_rows(None, "2026-09-14")
check("None writes nothing", len(pd.read_csv(p)) == before)
# A slate predicted before a column existed must still contribute the
# columns it DOES have, exactly like the pitcher log.
thin = frame(n=31, seed=3).drop(columns=["prob_hit", "hits", "total_bases"])
S._log_hitter_rows(thin, "2026-09-15")
d = pd.read_csv(p)
check("a slate missing columns still logs", (d.game_date == "2026-09-15").sum() == 31)

print()
if FAILED:
    print("FAILURES: " + ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
