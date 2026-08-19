"""
Checks on the hits+runs+RBI swap.

    python test_hrr.py

Two parts. The first is synthetic and runs anywhere: it verifies the
arithmetic identities the new code depends on, using data whose right
answer is known by construction. The second only runs if the real caches
are present, and checks that the estimated pieces reproduce the numbers
actually observed in 306k boxscore lines.

WHY BOTH
--------
Every bug in this project so far has been a fallback path that looked like
a success path -- no error, plausible output, quietly wrong. Synthetic
tests catch the arithmetic. Only real data catches "this ran fine and
returned the league average for everybody".
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.pa_table import (
    build_pa_table, build_game_totals, per_pa_contribution_distribution,
)
from features.run_features import (
    RunScoringModel, add_lineup_obp_behind, build_run_training_frame,
    LEAGUE_SCORE_PROB,
)
from data.cache import cache_path

PASS, FAIL = "  PASS", "  FAIL"
failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def synthetic_pa(n=20000, seed=0):
    """A fake PA table with known event frequencies."""
    rng = np.random.default_rng(seed)
    events = rng.choice(
        ["single", "double", "triple", "home_run", "walk", "hit_by_pitch",
         "field_error", "strikeout", "field_out"],
        size=n,
        p=[0.145, 0.045, 0.005, 0.035, 0.085, 0.010, 0.010, 0.225, 0.440],
    )
    rbi = np.where(events == "home_run", rng.integers(1, 5, n), 0)
    rbi = np.where(np.isin(events, ["single", "double", "triple"]),
                   rng.binomial(1, 0.35, n), rbi)
    return pd.DataFrame({
        "batter": rng.integers(1, 40, n),
        "pitcher": rng.integers(100, 130, n),
        "game_pk": rng.integers(1, 2500, n),
        "game_date": pd.Timestamp("2025-04-01") + pd.to_timedelta(
            rng.integers(0, 150, n), unit="D"),
        "at_bat_number": rng.integers(1, 40, n),
        "events": events,
        "inning": rng.integers(1, 10, n),
        "bat_score": 0,
        "post_bat_score": rbi,
        "inning_topbot": rng.choice(["Top", "Bot"], n),
        "stand": rng.choice(["L", "R"], n),
        "p_throws": rng.choice(["L", "R"], n),
        "home_team": rng.choice(["nyy", "lad", "hou"], n),
        "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan,
        "outs_when_up": rng.integers(0, 3, n),
    })


# ---------------------------------------------------------------- part 1
print("=" * 72)
print("PART 1 -- ARITHMETIC (synthetic data, known answers)")
print("=" * 72)

raw = synthetic_pa()
pa = build_pa_table(raw)

check("build_pa_table emits the new columns",
      {"reached", "reached_nonhr", "hrr_certain"}.issubset(pa.columns))
check("hrbw is gone", "hrbw" not in pa.columns)

# A home run is a hit, is reaching base, and is NOT a non-HR time on base.
hr = pa[pa["is_hr"] == 1]
check("home runs count as reaching base", bool((hr["reached"] == 1).all()))
check("home runs are excluded from reached_nonhr",
      bool((hr["reached_nonhr"] == 0).all()))
check("a home run's certain value includes its own run",
      bool((hr["hrr_certain"] == 2 + hr["rbi"]).all()),
      f"(1 hit + RBI + his own run; sample {hr['hrr_certain'].head(3).tolist()} "
      f"on RBI {hr['rbi'].head(3).tolist()})")
check("a solo home run is worth exactly 3",
      bool((hr.loc[hr["rbi"] == 1, "hrr_certain"] == 3).all()),
      "(1 hit + 1 RBI + 1 run)")
check("reached_nonhr is never negative", bool((pa["reached_nonhr"] >= 0).all()))
check("outs contribute nothing",
      float(pa.loc[pa["reached"] == 0, "hrr_certain"].abs().sum()) ==
      float(pa.loc[pa["reached"] == 0, "rbi"].sum()),
      "(an RBI groundout still counts)")

# The whole point of folding the run in analytically: the mean must move by
# exactly q per non-home-run time on base, with no sampling noise.
q = 0.3084
plain = per_pa_contribution_distribution(pa, "hrr_certain")
with_runs = per_pa_contribution_distribution(pa, "hrr_certain", score_prob=q)
values = np.arange(len(plain))
mean_plain = float((plain * values).sum())
mean_runs = float((with_runs * values).sum())
expected_shift = q * float(pa["reached_nonhr"].mean())

check("both distributions are proper probability vectors",
      abs(plain.sum() - 1) < 1e-12 and abs(with_runs.sum() - 1) < 1e-12)
check("folding the run in shifts the mean by exactly q x times-on-base",
      abs((mean_runs - mean_plain) - expected_shift) < 1e-9,
      f"(shift {mean_runs - mean_plain:.6f} vs expected {expected_shift:.6f})")
check("q = 0 leaves the distribution untouched",
      np.allclose(per_pa_contribution_distribution(pa, "hrr_certain",
                                                   score_prob=0.0), plain))
check("the fold is deterministic (no hidden RNG)",
      np.array_equal(with_runs,
                     per_pa_contribution_distribution(pa, "hrr_certain",
                                                      score_prob=q)))
check("mass moves UP, never down",
      float(with_runs[0]) <= float(plain[0]) + 1e-12)

totals = build_game_totals(pa)
check("game totals carry the new columns",
      {"reached_nonhr", "hrr_certain"}.issubset(totals.columns))
check("game-level hrr_certain equals the sum of its PAs",
      abs(float(totals["hrr_certain"].sum()) - float(pa["hrr_certain"].sum())) < 1e-9)

# --- the run model's own arithmetic ---
model = RunScoringModel()
frame = pd.DataFrame({"batter": [1, 2, 3], "game_pk": [1, 1, 1],
                      "lineup_slot": [1, 4, 9]})
probs = model.score_prob(frame)
check("fallback slot factors reproduce the measured U-shape",
      probs[0] > probs[1] and probs[2] > probs[1],
      f"(slot1 {probs[0]:.4f} > slot4 {probs[1]:.4f} < slot9 {probs[2]:.4f})")

runs = model.runs_per_pa(frame, p_hr=np.array([0.05, 0.05, 0.05]),
                         p_reach=np.array([0.40, 0.40, 0.40]))
manual = 0.05 + (0.40 - 0.05) * probs
check("runs_per_pa = P(HR) + P(reached otherwise) x q",
      np.allclose(runs, manual))

# A hitter who never reaches base scores only his home runs.
never = model.runs_per_pa(frame, p_hr=np.zeros(3), p_reach=np.zeros(3))
check("a hitter who never reaches base never scores", bool((never == 0).all()))
always_hr = model.runs_per_pa(frame, p_hr=np.ones(3), p_reach=np.ones(3))
check("a hitter who always homers always scores", bool(np.allclose(always_hr, 1.0)))

# Matchup responsiveness -- the reason runs are routed through reaching
# base rather than estimated as a per-batter constant.
tough = model.runs_per_pa(frame, p_hr=np.full(3, 0.03), p_reach=np.full(3, 0.28))
easy = model.runs_per_pa(frame, p_hr=np.full(3, 0.06), p_reach=np.full(3, 0.42))
check("a tougher pitcher lowers expected runs", bool((easy > tough).all()),
      f"({tough.mean():.4f} vs {easy.mean():.4f} per PA)")

# obp_behind must wrap around the order.
lineup = pd.DataFrame({"batter": range(1, 10), "game_pk": [7] * 9,
                       "lineup_slot": range(1, 10)})
obp = pd.Series({i: 0.250 for i in range(1, 10)})
obp.loc[1] = obp.loc[2] = obp.loc[3] = 0.400
out = add_lineup_obp_behind(lineup, obp, 0.320)
behind9 = float(out.loc[out["lineup_slot"] == 9, "obp_behind"].iloc[0])
behind5 = float(out.loc[out["lineup_slot"] == 5, "obp_behind"].iloc[0])
check("obp_behind wraps: slot 9 sees slots 1-2-3",
      abs(behind9 - 0.400) < 1e-9, f"(got {behind9:.4f})")
check("obp_behind is low where the bottom of the order follows",
      abs(behind5 - 0.250) < 1e-9, f"(slot 5 got {behind5:.4f})")

# Guard rails.
wild = RunScoringModel(batter_factors=pd.Series({1: 99.0, 2: 0.0, 3: 1.0}))
capped = wild.score_prob(frame)
check("q is clamped to a sane range",
      bool(((capped >= 0.10) & (capped <= 0.60)).all()),
      f"({capped.round(4).tolist()})")

# fit() must refuse to invent a model from nothing.
thin = RunScoringModel.fit(pd.DataFrame({"batter": [1], "lineup_slot": [1]}),
                           pd.Series(dtype=float), 0.32, verbose=False)
check("fit() falls back cleanly when columns are missing",
      abs(thin.league_prob - LEAGUE_SCORE_PROB) < 1e-12)


# ---------------------------------------------------------------- part 2
print("\n" + "=" * 72)
print("PART 2 -- REAL DATA (skipped if the caches aren't here)")
print("=" * 72)

lines_path = cache_path("batting_lines")
slots_path = cache_path("lineup_slots")
if not (os.path.exists(lines_path) and os.path.exists(slots_path)):
    print("  cache/batting_lines.csv or cache/lineup_slots.csv not found -- "
          "skipping.")
else:
    lines = pd.read_csv(lines_path)
    lines = lines[lines["player_id"] > 0]
    slots = pd.read_csv(slots_path)
    slots = slots[slots["is_starter"] == 1][["game_pk", "batter", "lineup_slot"]]

    # Stand in for game_totals -- INCLUDING the Statcast-named columns that
    # build_game_totals really emits. An earlier version of this test built
    # a bare frame with only slots, which meant the merge had nothing to
    # collide with and a name collision on `walks` sailed through here and
    # blew up in the real pipeline instead. A test fixture that is tidier
    # than the real input tests the fixture.
    totals = slots.copy()
    totals["game_date"] = pd.Timestamp("2025-01-01")
    for column in ("pa", "hr", "hits", "walks", "strikeouts", "rbi",
                   "reached", "reached_nonhr", "hrr_certain"):
        totals[column] = 0
    real = build_run_training_frame(totals, lines)
    check("official columns survive the merge un-suffixed",
          not any(c.endswith(("_x", "_y")) for c in real.columns))

    print(f"  {len(real):,} player-games matched.")
    check("build_run_training_frame produces hrr",
          real["hrr"].notna().mean() > 0.9,
          f"({real['hrr'].notna().mean():.1%} matched)")
    check("mean H+R+RBI for starters is in the right neighbourhood",
          1.6 < real["hrr"].mean() < 1.9, f"({real['hrr'].mean():.3f})")

    obp_real = ((real.groupby("batter")["hits_official"].sum()
                 + real.groupby("batter")["walks_official"].sum()
                 + real.groupby("batter")["hbp_official"].sum())
                / real.groupby("batter")["pa_official"].sum())
    league_obp = float(obp_real.median())
    real = add_lineup_obp_behind(real, obp_real, league_obp)

    fitted = RunScoringModel.fit(real, obp_real, league_obp, verbose=True)

    check("league scoring rate matches the 306k-line measurement",
          abs(fitted.league_prob - 0.3084) < 0.02,
          f"({fitted.league_prob:.4f} vs 0.3084)")
    check("slot 1 scores more often than slot 4",
          fitted.slot_factors[1] > fitted.slot_factors[4],
          f"({fitted.slot_factors[1]:.3f} vs {fitted.slot_factors[4]:.3f})")
    check("slot 9 scores more often than slot 4 (top of the order follows)",
          fitted.slot_factors[9] > fitted.slot_factors[4],
          f"({fitted.slot_factors[9]:.3f} vs {fitted.slot_factors[4]:.3f})")

    spread = fitted.batter_factors
    check("the per-batter factor actually varies", float(spread.std()) > 0.01,
          f"(sd {float(spread.std()):.4f}, "
          f"{spread.min():.3f} to {spread.max():.3f})")
    check("the per-batter factor is centred near 1",
          abs(float(spread.mean()) - 1.0) < 0.05,
          f"(mean {float(spread.mean()):.4f})")

    # THE HEADLINE CHECK. Predicted runs, summed over every player-game,
    # against the runs that were actually scored. This is the one that
    # catches a plausible-looking model that is quietly biased.
    real["p_reach"] = ((real["hits_official"] + real["walks_official"]
                        + real["hbp_official"])
                       / real["pa_official"].replace(0, np.nan))
    real["p_hr"] = real["hr_official"] / real["pa_official"].replace(0, np.nan)
    usable = real.dropna(subset=["p_reach", "p_hr", "lineup_slot"]).copy()
    predicted = fitted.runs_per_pa(usable, usable["p_hr"], usable["p_reach"])
    predicted_runs = float((predicted * usable["pa_official"]).sum())
    actual_runs = float(usable["runs_official"].sum())
    bias = predicted_runs / actual_runs - 1

    print(f"\n  Predicted {predicted_runs:,.0f} runs against {actual_runs:,.0f} "
          f"actual ({bias:+.2%}).")
    check("total predicted runs are within 3% of actual", abs(bias) < 0.03,
          f"({bias:+.2%})")

    # And by slot, where a single league constant would visibly fail.
    usable["pred_runs"] = predicted * usable["pa_official"]
    print(f"\n  {'slot':>4} {'predicted':>10} {'actual':>10} {'error':>8}")
    worst = 0.0
    for slot, g in usable.groupby("lineup_slot"):
        p, a = g["pred_runs"].sum(), g["runs_official"].sum()
        err = p / a - 1
        worst = max(worst, abs(err))
        print(f"  {int(slot):>4} {p:>10,.0f} {a:>10,.0f} {err:>+8.2%}")
    check("no lineup slot is off by more than 5%", worst < 0.05,
          f"(worst {worst:.2%})")




# ---------------------------------------------------------------- part 3
print("\n" + "=" * 72)
print("PART 3 -- STARTER vs BULLPEN IDENTIFICATION")
print("=" * 72)

from data.pitcher_data import measure_bullpen_rates, STARTER_PA_SHARE

# A synthetic slate where the truth is known: 2 teams per game, one starter
# each, who throws the first 60% of his side's plate appearances.
rng = np.random.default_rng(7)
rows = []
for game in range(1, 401):
    for side in (0, 1):
        starter = 1000 + game * 2 + side
        for i in range(40):
            vs_starter = i < 24            # 60% by construction
            rows.append({
                "game_pk": game, "is_home": side,
                "inning": 1 + i // 5,
                "at_bat_number": i * 2 + side,
                "pitcher": starter if vs_starter else 9000 + rng.integers(0, 50),
                "batter": rng.integers(1, 30),
                "is_hr": 0, "is_hit": 0, "is_walk": 0, "is_k": 0,
            })
fake = pd.DataFrame(rows)

share, _ = measure_bullpen_rates(fake)
check("starter share recovered from a known-truth slate",
      abs(share - 0.60) < 0.02, f"(got {share:.1%}, truth 60.0%)")

# The exact failure mode that shipped: group by game only, and the home
# side's starter is invisible. Simulated by hiding is_home.
no_side = fake.drop(columns=["is_home"])
share_broken, _ = measure_bullpen_rates(no_side)
check("a PA table without is_home falls back instead of guessing",
      abs(share_broken - STARTER_PA_SHARE) < 1e-9,
      f"(got {share_broken:.1%})")

# An implausible share must be rejected, not passed downstream.
scrambled = fake.copy()
scrambled["pitcher"] = rng.integers(9000, 9050, len(scrambled))
share_bad, _ = measure_bullpen_rates(scrambled)
check("an impossible starter share is rejected",
      abs(share_bad - STARTER_PA_SHARE) < 1e-9, f"(got {share_bad:.1%})")

check("PA_COLUMNS carries what starter identification needs",
      {"inning", "at_bat_number"}.issubset(set(pa.columns)),
      f"(pa_table columns include inning/at_bat_number)")


# ---------------------------------------------------------------- part 4
print("\n" + "=" * 72)
print("PART 4 -- TOTAL BASES")
print("=" * 72)

from features.bases_features import (
    measure_bases_per_hit, expected_total_bases_per_pa, LEAGUE_BASES_PER_HIT,
    BASE_VALUES,
)
from model.compound import (
    scale_contribution_distribution, compound_count_distribution_overdispersed,
    prob_over_line, prob_at_least_one,
)

check("build_pa_table emits total_bases", "total_bases" in pa.columns)

# The map is definitional, so a typo here would be a silently wrong prop
# rather than a crash -- worth asserting rather than trusting.
check("single=1 double=2 triple=3 homer=4",
      BASE_VALUES == {"single": 1, "double": 2, "triple": 3, "home_run": 4})

walks = pa[pa["is_walk"] == 1]
check("a walk is worth ZERO total bases",
      bool((walks["total_bases"] == 0).all()),
      f"({len(walks)} walks checked)")
check("every hit is worth at least one base",
      bool((pa.loc[pa["is_hit"] == 1, "total_bases"] >= 1).all()))
check("every non-hit is worth zero",
      bool((pa.loc[pa["is_hit"] == 0, "total_bases"] == 0).all()))
check("every home run is worth exactly four",
      bool((pa.loc[pa["is_hr"] == 1, "total_bases"] == 4).all()))

league, per_batter, prior = measure_bases_per_hit(pa, verbose=False)
check("bases per non-HR hit lands between a single and a triple",
      1.0 < league < 2.0, f"({league:.4f})")

# The double-counting trap: is_hit includes home runs.
tb = expected_total_bases_per_pa(p_hr=np.array([0.04]), p_hit=np.array([0.24]),
                                 bases_per_hit=np.array([1.27]))
manual = 0.04 * 4 + (0.24 - 0.04) * 1.27
check("home runs are not counted twice",
      abs(float(tb[0]) - manual) < 1e-12,
      f"(got {float(tb[0]):.5f}, hand-computed {manual:.5f})")

# A hitter who only ever homers: every hit is a homer, so p_hit == p_hr
# and the non-HR term must vanish entirely.
only_hr = expected_total_bases_per_pa(np.array([0.10]), np.array([0.10]),
                                      np.array([1.27]))
check("when every hit is a homer the non-HR term vanishes",
      abs(float(only_hr[0]) - 0.40) < 1e-12, f"({float(only_hr[0]):.4f})")

check("a doubles hitter projects more bases than a singles hitter",
      float(expected_total_bases_per_pa([0.04], [0.24], [1.45])[0]) >
      float(expected_total_bases_per_pa([0.04], [0.24], [1.10])[0]))

# THE CROSS-CHECK. P(hits over 0.5) by convolution must equal
# P(at least one hit) by 1-(1-p)^n. Two unrelated code paths, one answer.
hit_dist = per_pa_contribution_distribution(pa, "is_hit")
worst = 0.0
for p_hit in (0.15, 0.22, 0.30):
    for n in (3, 4, 5):
        shaped = scale_contribution_distribution(hit_dist, p_hit,
                                                 method="frequency")
        by_convolution = prob_over_line(
            compound_count_distribution_overdispersed(shaped, n, 0.0,
                                                      method="frequency"), 0.5)
        by_formula = float(prob_at_least_one(p_hit, n))
        worst = max(worst, abs(by_convolution - by_formula))
check("convolution and 1-(1-p)^n agree on P(at least one hit)",
      worst < 1e-6, f"(largest gap {worst:.2e} across 9 combinations)")

# Total bases must be monotone in the hitter's rates.
better = expected_total_bases_per_pa([0.06], [0.28], [1.30])[0]
worse = expected_total_bases_per_pa([0.03], [0.20], [1.25])[0]
check("a better hitter projects more total bases per PA",
      better > worse, f"({worse:.4f} -> {better:.4f})")


print("\n" + "=" * 72)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
print("All checks passed.")
