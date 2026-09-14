"""
Slip building and slip grading, on data with answers known by construction.

WHY SYNTHETIC
-------------
Grading is all off-by-one risk. "Over 1.5 hits" means TWO hits, not one;
"Under 5.5 K" means five or fewer. A scorer that is one out on either
convention still runs, still produces plausible percentages, and would be
wrong on every count prop forever. Nothing raises.

So each case here is a leg whose correct answer is obvious, and the
assertion is on that answer rather than on the code path.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slips as S
import score_slips as G

ok = True


def check(label, got, want):
    global ok
    good = (got == want) or (pd.isna(got) and pd.isna(want))
    ok &= good
    print(f"  [{'ok  ' if good else 'FAIL'}] {label:52} "
          f"got {got!r}, want {want!r}")


print("=" * 72)
print("PART 1 -- a leg is graded on the right side of the line")
print("=" * 72)

# Count props: strictly greater than the line.
row = pd.Series({"hits": 2, "total_bases": 3, "hrr": 1, "got_hr": 0,
                 "got_hit": 1, "got_walk": 0, "strikeouts": 6})
check("2 hits clears Hits 1.5", G.grade_leg("prob_hits_over_1.5", np.nan, row), 1.0)
check("2 hits does NOT clear Hits 2.5", G.grade_leg("prob_hits_over_2.5", np.nan, row), 0.0)
check("3 TB clears TB 2.5", G.grade_leg("prob_tb_over_2.5", np.nan, row), 1.0)
check("3 TB does NOT clear TB 3.5", G.grade_leg("prob_tb_over_3.5", np.nan, row), 0.0)
check("1 H+R+RBI clears 0.5", G.grade_leg("prob_hrr_over_0.5", np.nan, row), 1.0)
check("1 H+R+RBI does NOT clear 1.5", G.grade_leg("prob_hrr_over_1.5", np.nan, row), 0.0)

# Binary props.
check("got_hit 1 -> Hit lands", G.grade_leg("prob_hit", np.nan, row), 1.0)
check("got_hr 0 -> HR misses", G.grade_leg("prob_hr", np.nan, row), 0.0)
check("got_walk 0 -> Walk misses", G.grade_leg("prob_walk", np.nan, row), 0.0)

# Pitcher legs, both sides.
check("6 K clears Over 5.5", G.grade_leg("pitcher_k_over", 5.5, row), 1.0)
check("6 K does NOT clear Over 6.5", G.grade_leg("pitcher_k_over", 6.5, row), 0.0)
check("6 K means Under 5.5 MISSES", G.grade_leg("pitcher_k_under", 5.5, row), 0.0)
check("6 K means Under 6.5 LANDS", G.grade_leg("pitcher_k_under", 6.5, row), 1.0)

# Ungradeable is NaN, never a miss.
check("absent player -> NaN, not 0", G.grade_leg("prob_hit", np.nan, None), float("nan"))
check("missing column -> NaN", G.grade_leg("prob_hits_over_1.5", np.nan,
                                           pd.Series({"got_hit": 1})), float("nan"))

print()
print("=" * 72)
print("PART 2 -- a slip lands only when EVERY leg does")
print("=" * 72)

built = pd.DataFrame([
    # window 0, SAFE: three legs, all land
    dict(window=0, tier="SAFE", n_games=3, solo=False, clash=False,
         player_id=1, name="A", prop_key="prob_hit", prop="Hit", line=np.nan,
         kind="bat", p=0.7, game_pk=10, slip_p=0.343),
    dict(window=0, tier="SAFE", n_games=3, solo=False, clash=False,
         player_id=2, name="B", prop_key="prob_hit", prop="Hit", line=np.nan,
         kind="bat", p=0.7, game_pk=11, slip_p=0.343),
    dict(window=0, tier="SAFE", n_games=3, solo=False, clash=False,
         player_id=3, name="C", prop_key="prob_hit", prop="Hit", line=np.nan,
         kind="bat", p=0.7, game_pk=12, slip_p=0.343),
    # window 0, LOTTO: two land, one misses -> slip loses
    dict(window=0, tier="LOTTO", n_games=3, solo=False, clash=False,
         player_id=1, name="A", prop_key="prob_hr", prop="HR", line=np.nan,
         kind="bat", p=0.2, game_pk=10, slip_p=0.008),
    dict(window=0, tier="LOTTO", n_games=3, solo=False, clash=False,
         player_id=2, name="B", prop_key="prob_hr", prop="HR", line=np.nan,
         kind="bat", p=0.2, game_pk=11, slip_p=0.008),
    # window 1, MEDIUM: one leg cannot be graded -> whole slip dropped
    dict(window=1, tier="MEDIUM", n_games=1, solo=True, clash=False,
         player_id=9, name="Z", prop_key="prob_hit", prop="Hit", line=np.nan,
         kind="bat", p=0.5, game_pk=20, slip_p=0.25),
    dict(window=1, tier="MEDIUM", n_games=1, solo=True, clash=False,
         player_id=1, name="A", prop_key="prob_hit", prop="Hit", line=np.nan,
         kind="bat", p=0.5, game_pk=10, slip_p=0.25),
])
hitters = {
    (1, 10): pd.Series({"got_hit": 1, "got_hr": 1}),
    (2, 11): pd.Series({"got_hit": 1, "got_hr": 0}),
    (3, 12): pd.Series({"got_hit": 1, "got_hr": 0}),
    # player 9 never appeared
}
scored, legs = G.grade(built, hitters, {})
safe = scored[scored.tier == "SAFE"].iloc[0]
lot = scored[scored.tier == "LOTTO"].iloc[0]
med = scored[scored.tier == "MEDIUM"].iloc[0]
check("SAFE 3/3 legs -> slip hit", safe["hit"], 1.0)
check("SAFE legs_hit counted", safe["legs_hit"], 3.0)
check("LOTTO 1/2 legs -> slip lost", lot["hit"], 0.0)
check("LOTTO legs_hit counted", lot["legs_hit"], 1.0)
check("MEDIUM with an ungraded leg -> NaN", med["hit"], float("nan"))
check("  and it is reported as partly graded", med["n_graded"], 1)
check("solo window marked correlated", bool(med["correlated"]), True)

print()
print("=" * 72)
print("PART 3 -- build_slips on the real slate")
print("=" * 72)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
frame = pd.read_csv(os.path.join(CACHE, "slate_2026-09-12.csv"))
frame["start"] = pd.to_datetime(frame["start_time_utc"], utc=True,
                                errors="coerce")
pit = pd.read_csv(os.path.join(CACHE, "pitchers_2026-09-12.csv"))
out = S.build_slips(frame, pit, S.market_lines("2026-09-12", CACHE), legs=3)

per = out.drop_duplicates(["window", "tier"])
check("four tiers per window", sorted(out["tier"].unique()),
      sorted(S.TIER_ORDER))
check("every slip has >= 2 legs", int(out.groupby(["window", "tier"]).size().min()) >= 2, True)

dupes = out.groupby(["window", "tier"])["name"].apply(
    lambda s: len(s) != s.nunique()).sum()
check("no slip repeats a player", int(dupes), 0)

arms = out[out["kind"] == "arm"]
check("at most one arm per slip",
      int(arms.groupby(["window", "tier"]).size().max()) if len(arms) else 1, 1)
check("no arm leg has a negative edge",
      int((arms["edge"].astype(float) < 0).sum()) if len(arms) else 0, 0)
check("no suspect arm (Over at a low line)",
      int((arms["prop_key"].eq("pitcher_k_over")
           & (arms["line"] <= S.LOW_LINE)).sum()) if len(arms) else 0, 0)

# slip_p must be the product of its own legs.
worst = 0.0
for (_w, _t), g in out.groupby(["window", "tier"]):
    worst = max(worst, abs(float(np.prod(g["p"])) - float(g["slip_p"].iloc[0])))
check("slip_p equals the product of its legs", worst < 1e-9, True)

# Tier membership must match the cut points.
bad_tier = 0
for _t, key in (("SAFE", "p"), ("MEDIUM", None), ("LOTTO", None)):
    g = out[out["tier"] == _t]
    if _t == "SAFE":
        bad_tier += int((g["p"] < S.SAFE_MIN).sum())
    elif _t == "LOTTO":
        bad_tier += int((g["p"] >= S.LOTTO_MAX).sum())
    else:
        bad_tier += int(((g["p"] < S.LOTTO_MAX) | (g["p"] >= S.SAFE_MIN)).sum())
check("every pure-tier leg is inside its cut points", bad_tier, 0)

print(f"\n  {len(per)} slips, {len(out)} legs, "
      f"{out['window'].nunique()} windows")
for t in S.TIER_ORDER:
    s = per[per["tier"] == t]
    print(f"    {t:8} {len(s):>2} slips, mean {s['slip_p'].mean():6.2%}, "
          f"expected hits {s['slip_p'].sum():.2f}")

print()
print("=" * 72)
print("ALL PASS" if ok else "FAILURES ABOVE")
print("=" * 72)
sys.exit(0 if ok else 1)
