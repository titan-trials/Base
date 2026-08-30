"""
Synthetic tests for features/pitcher_workload.py.

The backtest already showed the model calibrates on real starts. These
check the things a backtest cannot: that the exact compounding really is
exact, that the pieces do what their names say, and that the failure modes
fail loudly.

    python test_workload.py
"""
import sys

import numpy as np
import pandas as pd

from features.pitcher_workload import (
    identify_starts, regular_season_only, WorkloadModel,
    k_count_distribution, prob_over, _tilt_pmf,
    OPENER_MAX_BF, OPENER_MAX_INNING, K_DISPERSION_SD,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail and not cond else ''}")


def make_pa(pitcher, game_pk, n_bf, n_k, date="2026-05-01",
            first_inning=1, last_inning=6, game_type="R"):
    ev = ["strikeout"] * n_k + ["field_out"] * (n_bf - n_k)
    return [{"pitcher": pitcher, "game_pk": game_pk, "batter": 1000 + i,
             "events": ev[i], "inning": first_inning if i == 0 else last_inning,
             "at_bat_number": i + 1, "game_date": date, "game_type": game_type}
            for i in range(n_bf)]


print("=" * 68)
print("PART 1 -- the compounding is exact, not approximate")
print("=" * 68)

# With a fixed number of batters faced and one constant probability, the
# answer is the binomial distribution. Anything else is a bug.
p, n = 0.25, 20
sup, bfp = np.array([n]), np.array([1.0])
got = k_count_distribution([p] * n, sup, bfp, dispersion_sd=0.0)
from math import comb
want = np.array([comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n + 1)])
check("matches the binomial exactly when BF is fixed and p constant",
      np.allclose(got[:n + 1], want, atol=1e-12),
      f"max diff {np.abs(got[:n+1] - want).max():.2e}")

check("the distribution sums to 1",
      abs(got.sum() - 1.0) < 1e-12, f"sums to {got.sum():.12f}")

# Varying p per batter: mean must equal the sum of the probabilities.
probs = np.linspace(0.10, 0.40, 25)
got = k_count_distribution(probs, np.array([25]), np.array([1.0]),
                           dispersion_sd=0.0)
mean = float((np.arange(len(got)) * got).sum())
check("mean equals the sum of per-batter probabilities",
      abs(mean - probs.sum()) < 1e-10,
      f"{mean:.10f} vs {probs.sum():.10f}")

# Mixing over BF must equal the weighted mix of the fixed-BF answers.
sup, bfp = np.array([18, 22, 26]), np.array([0.3, 0.5, 0.2])
mixed = k_count_distribution([0.22] * 30, sup, bfp, dispersion_sd=0.0)
parts = [k_count_distribution([0.22] * 30, np.array([n]), np.array([1.0]),
                              dispersion_sd=0.0) for n in sup]
# Each fixed-BF call returns an array sized to its own support, so pad to
# the longest before mixing.
manual = np.zeros(len(mixed))
for w, part in zip(bfp, parts):
    manual[:len(part)] += w * part
check("marginalising over BF equals the weighted mix of fixed-BF answers",
      np.allclose(mixed, manual, atol=1e-12))


print()
print("=" * 68)
print("PART 2 -- dispersion widens without moving the mean")
print("=" * 68)

sup, bfp = np.array([22]), np.array([1.0])
tight = k_count_distribution([0.23] * 25, sup, bfp, dispersion_sd=0.0)
wide = k_count_distribution([0.23] * 25, sup, bfp, dispersion_sd=K_DISPERSION_SD)


def moments(d):
    n = np.arange(len(d))
    mu = float((n * d).sum())
    return mu, float(np.sqrt((n**2 * d).sum() - mu**2))


m1, s1 = moments(tight)
m2, s2 = moments(wide)
check("dispersion leaves the mean alone", abs(m1 - m2) < 1e-9,
      f"{m1:.6f} -> {m2:.6f}")
check("dispersion widens the spread", s2 > s1, f"{s1:.4f} -> {s2:.4f}")

# The bug found during the build: a module constant used as a default
# argument is bound at import and cannot be overridden afterwards.
import features.pitcher_workload as pwmod
original = pwmod.K_DISPERSION_SD
pwmod.K_DISPERSION_SD = 0.40
bumped = k_count_distribution([0.23] * 25, sup, bfp)
pwmod.K_DISPERSION_SD = original
check("the module constant is read at CALL time, not bound at import",
      moments(bumped)[1] > s2,
      "reassigning K_DISPERSION_SD had no effect -- it is a bound default")


print()
print("=" * 68)
print("PART 3 -- openers and spring training are excluded")
print("=" * 68)

rows = []
rows += make_pa(1, 100, 24, 7)                              # normal start
rows += make_pa(2, 101, 6, 2, first_inning=1, last_inning=2)  # opener
rows += make_pa(3, 102, 9, 3, first_inning=1, last_inning=7)  # 9 BF into the 7th
rows += make_pa(4, 103, 22, 6, game_type="S")               # spring training
starts = identify_starts(pd.DataFrame(rows))

check("the opener is flagged",
      bool(starts.set_index("pitcher").loc[2, "is_opener"]))
check("a short outing that goes deep is NOT an opener",
      not bool(starts.set_index("pitcher").loc[3, "is_opener"]),
      "9 BF into the 7th is a gem, not an opener")
check("the normal start is not flagged",
      not bool(starts.set_index("pitcher").loc[1, "is_opener"]))
check("the spring training game never reaches the start table",
      4 not in set(starts["pitcher"]),
      f"pitchers present: {sorted(starts['pitcher'])}")

raw = pd.DataFrame(rows)
check("regular_season_only drops exactly the non-R rows",
      len(regular_season_only(raw)) == len(raw[raw.game_type == "R"]))
check("regular_season_only is a no-op when game_type is absent",
      len(regular_season_only(raw.drop(columns=["game_type"]))) == len(raw))


print()
print("=" * 68)
print("PART 4 -- tilting hits the target mean")
print("=" * 68)

support = np.arange(10, 31, dtype=float)
base = np.exp(-0.5 * ((support - 22) / 4.0) ** 2)
base /= base.sum()
for target in (19.0, 22.0, 25.0):
    tilted = _tilt_pmf(support, base, target)
    got = float((tilted * support).sum())
    check(f"tilted mean lands on {target}", abs(got - target) < 1e-6,
          f"got {got:.6f}")
    check(f"  and stays a probability distribution ({target})",
          abs(tilted.sum() - 1) < 1e-12 and (tilted >= 0).all())


print()
print("=" * 68)
print("PART 5 -- the model shrinks, and prob_over reads the right tail")
print("=" * 68)

rows = []
for gp in range(200, 230):     # a deep-going pitcher, 30 starts
    rows += make_pa(10, gp, 27, 8, date=f"2026-0{1 + gp % 8}-15")
for gp in range(300, 302):     # a pitcher with only 2 starts
    rows += make_pa(11, gp, 27, 8, date=f"2026-0{1 + gp % 8}-15")
for gp in range(400, 460):     # the league, going ~20
    rows += make_pa(12 + gp % 20, gp, 20, 4, date=f"2026-0{1 + gp % 8}-15")
wm = WorkloadModel.fit(identify_starts(pd.DataFrame(rows)),
                       as_of=None, verbose=False)
deep, thin = wm.expected_bf(10), wm.expected_bf(11)
check("a pitcher with 30 starts keeps most of his own mean",
      deep > 25.0, f"{deep:.2f} against a raw 27")
check("a pitcher with 2 starts is pulled hard toward the league",
      thin < deep - 1.0, f"2 starts -> {thin:.2f}, 30 starts -> {deep:.2f}")
check("an unknown pitcher gets the league mean exactly",
      abs(wm.expected_bf(999999) - wm.league_mean) < 1e-12)

d = np.zeros(12)
d[4] = d[5] = d[6] = d[7] = 0.25
check("prob_over(5.5) counts 6 and up, not 5",
      abs(prob_over(d, 5.5) - 0.5) < 1e-12, f"got {prob_over(d, 5.5)}")
check("prob_over(6.5) counts 7 and up",
      abs(prob_over(d, 6.5) - 0.25) < 1e-12, f"got {prob_over(d, 6.5)}")
check("prob_over past the support is 0, not an error",
      prob_over(d, 99.5) == 0.0)


print()
print("=" * 68)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
print("=" * 68)
sys.exit(1 if FAIL else 0)
