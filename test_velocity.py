"""
The velocity adjustment, and the two ways it could quietly go wrong.

    python test_velocity.py

It is a small function, but it sits on the projection path for every
starter on the board, and two of its failure modes are silent: an Unknown
pitcher getting an adjustment built from NaN, and the clamp not actually
clamping so a thin velocity history swings a projection on nothing.
"""
import sys

import numpy as np

# pitcher_form, not predict_slate: the adjustment lives beside the feature
# it scales, so this test needs neither the slate pipeline nor a network.
import pitcher_form as P
from pitcher_form import velocity_for

FAILED = []


def check(label, got, want=None, ok=None):
    passed = ok if ok is not None else (got == want)
    print(f"  {'ok  ' if passed else 'FAIL'}  {label}")
    if not passed:
        FAILED.append(f"{label}: got {got!r}, wanted {want!r}")


print("apply_velocity")
BASE = 0.22

# A pitcher with nothing to say about his arm must come through untouched.
# This is the one that matters most: NaN arithmetic does not raise, it
# propagates, and a NaN k_rate turns into a NaN distribution and a blank
# row that looks like missing data rather than a bug.
check("Unknown (NaN) passes the rate through unchanged",
      P.apply_velocity(BASE, float("nan")), BASE)
check("None passes the rate through unchanged",
      P.apply_velocity(BASE, None), BASE)
check("zero z is a no-op", P.apply_velocity(BASE, 0.0), BASE)

up = P.apply_velocity(BASE, 1.0)
down = P.apply_velocity(BASE, -1.0)
check("a live arm raises the rate", up > BASE, ok=up > BASE)
check("a dead arm lowers it", down < BASE, ok=down < BASE)
check("one sd is worth the measured slope",
      round(up - BASE, 6), round(P.VELO_K_SLOPE, 6))
check("symmetric about zero",
      round((up - BASE) + (down - BASE), 9), 0.0)

# The clamp. Without it a pitcher with three erratic starts and a tiny
# baseline standard deviation produces a huge z on almost no evidence.
far = P.apply_velocity(BASE, 12.0)
cap = P.apply_velocity(BASE, P.VELO_Z_CAP)
check("z beyond the cap is clamped, not extrapolated", far, cap)
check("the clamp binds below the cap too",
      P.apply_velocity(BASE, -12.0), P.apply_velocity(BASE, -P.VELO_Z_CAP))
check("at the cap the move is about 0.4 K over 23 batters",
      0.35 < (cap - BASE) * 23 < 0.50,
      ok=0.35 < (cap - BASE) * 23 < 0.50)

# A rate is a probability. Nothing downstream checks this.
check("never leaves [0.01, 0.60] on a silly input",
      0.01 <= P.apply_velocity(0.59, 99.0) <= 0.60,
      ok=0.01 <= P.apply_velocity(0.59, 99.0) <= 0.60)
check("a near-zero rate cannot be pushed negative",
      P.apply_velocity(0.011, -99.0) >= 0.01,
      ok=P.apply_velocity(0.011, -99.0) >= 0.01)

print("\nvelocity_for")
# An id with no cache must come back PRESENT and Unknown. If it came back
# missing instead, a join in predict_slate would drop the pitcher and he
# would vanish from the board rather than project without the adjustment.
tbl = velocity_for([1, 2, 3])
check("an unknown pitcher id is present, not dropped", len(tbl), 3)
check("and is marked Unknown",
      set(tbl["velo_state"]), {"Unknown"})
check("with a NaN z that apply_velocity tolerates",
      P.apply_velocity(BASE, tbl.iloc[0]["velo_z"]), BASE)
check("duplicate ids collapse to one row", len(velocity_for([1, 1, 1])), 1)

print()
if FAILED:
    print("FAILURES")
    for f in FAILED:
        print("  " + f)
    sys.exit(1)
print("ALL PASS")
