"""
The four tiers of expected_bf, on synthetic data with known answers.

WHY SYNTHETIC
-------------
The bug this tests for is not an exception, it is a WRONG NUMBER -- the
model classified Riley Cornelio as a reliever, printed that into the slate
file, and projected him for 22.65 batters faced anyway. Nothing raised.
A test that only checks the code runs would have passed on the bug.

So each pitcher here is built with a workload that makes the right answer
obvious by construction, and the assertions are on ranges the correct
model must land in and the broken one cannot.

THE FOUR TIERS
--------------
  ACE        many real starts        -> his own mean, ~25
  OPENER     only opener-length      -> his own opener mean, ~7
  RELIEVER   never started at all    -> relief work shrunk to opener, <12
  DEBUT      no history whatsoever   -> the new-pitcher prior, ~20

The third is the one that was missing. The fourth is the control: it must
NOT move, because a genuine debut starter really does face about twenty and
the fix must not drag him down with the relievers.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.pitcher_workload import WorkloadModel, RELIEF_PRIOR_APPEARANCES

AS_OF = pd.Timestamp("2026-09-13")
RNG = np.random.default_rng(3)


def build():
    """starts + appearances frames with four deliberately distinct arms."""
    starts, apps, pk = [], [], 900000

    def add(pitcher, date, bf, k, last_inning, is_start, outs=None):
        nonlocal pk
        pk += 1
        row = {"pitcher": pitcher, "game_pk": pk, "game_date": date,
               "bf": bf, "k": k, "last_inning": last_inning,
               "is_start": is_start,
               "outs": outs if outs is not None else bf * 0.7}
        apps.append(row)
        if is_start:
            starts.append(row)

    d0 = AS_OF - pd.Timedelta(days=300)
    # --- a population of ordinary starters, so the league mean is real ---
    for p in range(1, 61):
        for i in range(18):
            bf = int(RNG.normal(23, 3))
            add(f"STARTER{p}", d0 + pd.Timedelta(days=5 * i),
                bf, int(bf * 0.23), 6, True)
    # --- a population of openers, so opener_mean_bf is real --------------
    for p in range(1, 21):
        for i in range(6):
            bf = int(RNG.integers(5, 10))
            add(f"OPEN{p}", d0 + pd.Timedelta(days=9 * i),
                bf, int(bf * 0.22), 2, True, outs=bf * 0.6)
    # --- debut starters, so new_pitcher_mean is real ---------------------
    for p in range(1, 41):
        add(f"DEBUT{p}", d0 + pd.Timedelta(days=200),
            int(RNG.normal(20, 3)), 4, 5, True)

    # --- the four pitchers under test ------------------------------------
    for i in range(20):                                  # ACE
        add("ACE", d0 + pd.Timedelta(days=4 * i), 25, 8, 7, True)
    for i in range(5):                                   # OPENER
        add("OPENER", d0 + pd.Timedelta(days=14 * i), 7, 2, 2, True, outs=4)
    for i in range(40):                                  # RELIEVER, never started
        add("RELIEVER", d0 + pd.Timedelta(days=6 * i), 4, 1, 8, False, outs=3)
    # DEBUT_ARM appears nowhere at all -- the unseen case.

    cols = ["pitcher", "game_pk", "game_date", "bf", "k", "last_inning",
            "is_start", "outs"]
    s = pd.DataFrame(starts)[cols]
    s["is_opener"] = (s["bf"] < 10) & (s["last_inning"] < 4)
    return s, pd.DataFrame(apps)[cols]


def main():
    starts, apps = build()
    m = WorkloadModel.fit(starts, as_of=AS_OF, appearances=apps, verbose=True)
    print()

    league = m.league_mean
    checks = [
        ("ACE",       "starter",  22.0, 27.0, "his own 25-batter history"),
        ("OPENER",    "opener",    5.0, 10.0, "his own opener starts"),
        ("RELIEVER",  "reliever",  3.0, 12.0, "relief work, shrunk to opener"),
        ("DEBUT_ARM", "unknown",  16.0, 23.0, "the new-pitcher prior"),
    ]
    print(f"{'pitcher':11} {'role':9} {'exp_bf':>7} {'k_rate':>7} "
          f"{'expected':>12}   basis")
    ok = True
    for pid, want_role, lo, hi, basis in checks:
        bf = m.expected_bf(pid)
        role = m.role(pid)
        good = lo <= bf <= hi and role == want_role
        ok &= good
        print(f"{pid:11} {role:9} {bf:7.2f} {m.k_rate(pid):7.4f} "
              f"{f'{lo:.0f}-{hi:.0f}':>12}   {basis}"
              f"{'' if good else '   <-- FAIL'}")

    print(f"\nleague starter mean {league:.2f} | opener mean "
          f"{m.opener_mean_bf:.2f} | new-pitcher {m.new_pitcher_mean:.2f}")

    # The regression this exists to catch: before the fix, RELIEVER
    # returned the starter prior. Assert the gap explicitly so a future
    # refactor that drops the tier fails loudly rather than quietly.
    gap = league - m.expected_bf("RELIEVER")
    print(f"RELIEVER is {gap:.1f} batters below the starter prior "
          f"(was 0.0 before the fix)")
    if gap < 8:
        print("  <-- FAIL: the relief tier is not being applied")
        ok = False

    # And the control: the fix must not have moved genuine debut starters.
    if not 16 <= m.new_pitcher_mean <= 23:
        print(f"  <-- FAIL: new_pitcher_mean {m.new_pitcher_mean:.1f} "
              f"moved; the fix leaked into debut starters")
        ok = False

    # The PMF has to follow expected_bf for EVERY tier, not just the new
    # one. k_count_distribution reads bf_pmf, not expected_bf, so a gap
    # here means the strikeout probabilities on the slate are computed
    # off a different workload than the one printed next to them.
    #
    # This is how the opener tier's own bug was found: OPENER projected
    # 7.01 batters and its distribution averaged 13.01, because tilting
    # cannot pull a pmf below its own support and the starter support
    # bottoms out at 13. That had been live since the opener tier landed.
    print()
    for pid, *_ in checks:
        support, probs = m.bf_pmf(pid)
        mean_pmf = float((support * probs).sum())
        gap = abs(mean_pmf - m.expected_bf(pid))
        bad = gap > 1.0
        ok &= not bad
        print(f"  bf_pmf mean {pid:11} {mean_pmf:6.2f}  vs expected_bf "
              f"{m.expected_bf(pid):6.2f}{'   <-- FAIL' if bad else ''}")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
