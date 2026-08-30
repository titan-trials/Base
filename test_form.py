"""
Synthetic tests for features/form_features.py.

Synthetic rather than real data on purpose: with a constructed history the
right answer is known exactly, so a wrong one is unmistakable. Run against
real Statcast the numbers would all look plausible and nothing would be
proved -- which is how the walks/hbp column collision and the 22.8%
starter share both survived their first tests in this project.

    python test_form.py
"""
import sys

import numpy as np
import pandas as pd

from features.form_features import (
    add_form_deviation, latest_form_by_batter,
    FORM_MIN_PA, HOT_Z, COLD_Z,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  -- ' + detail) if detail and not condition else ''}")


def make_pa(batter, outcomes, start_game=1):
    """A batter's career as a list of dicts, one per plate appearance."""
    rows = []
    for i, o in enumerate(outcomes):
        rows.append({
            "batter": batter,
            "game_pk": start_game + i // 4,
            "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i // 4),
            "is_k": int(o.get("k", 0)),
            "is_hit": int(o.get("h", 0)),
            "reached": int(o.get("r", o.get("h", 0))),
        })
    return rows


def steady(batter, n, k_rate=0.25, h_rate=0.25, seed=0):
    """A player with stable, known rates."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({"k": rng.random() < k_rate, "h": rng.random() < h_rate})
    return make_pa(batter, out)


print("=" * 68)
print("PART 1 -- the lookahead guard")
print("=" * 68)

# A batter who is utterly cold for 700 PA and then hits in every one of
# his last 40. The form score ON the first hot PA must not know about it.
cold = [{"k": 1, "h": 0} for _ in range(700)]
hot = [{"k": 0, "h": 1} for _ in range(40)]
df = pd.DataFrame(make_pa(1, cold + hot))
out = add_form_deviation(df)

switch = 700  # index of the first hot plate appearance
z_at_switch = out["form_z"].iloc[switch]
z_later = out["form_z"].iloc[switch + 35]

check("z at the moment form flips is still cold or neutral",
      pd.isna(z_at_switch) or z_at_switch <= 0.5,
      f"got {z_at_switch:.3f} -- the PA is seeing its own outcome")
check("z has gone clearly hot 35 PA later",
      z_later > HOT_Z, f"got {z_later:.3f}")

# The hard version: shift by exactly one. The value at index i must equal
# what you would compute from rows [0, i), never [0, i].
sub = out.iloc[switch:switch + 6]["form_is_hit_z_short"].to_numpy()
check("z rises monotonically through the streak, not in one jump",
      np.all(np.diff(sub) > 0),
      f"got {np.round(sub, 3)}")


print()
print("=" * 68)
print("PART 2 -- sign conventions")
print("=" * 68)

# Strikeouts up should read COLD, because is_k carries sign -1.
base = [{"k": 1, "h": 0} if i % 4 == 0 else {"k": 0, "h": 0}
        for i in range(700)]                       # 25% K rate
k_spike = [{"k": 1, "h": 0} for _ in range(60)]    # 100% K rate
df = pd.DataFrame(make_pa(2, base + k_spike))
out = add_form_deviation(df)
z_k = out["form_is_k_z_short"].iloc[-1]
check("a strikeout spike produces a NEGATIVE z (cold)",
      z_k < COLD_Z, f"got {z_k:.3f}")

# Hits up should read HOT.
base = [{"k": 0, "h": 1} if i % 4 == 0 else {"k": 0, "h": 0}
        for i in range(700)]                       # 25% hit rate
h_spike = [{"k": 0, "h": 1} for _ in range(60)]
df = pd.DataFrame(make_pa(3, base + h_spike))
out = add_form_deviation(df)
z_h = out["form_is_hit_z_short"].iloc[-1]
check("a hit spike produces a POSITIVE z (hot)",
      z_h > HOT_Z, f"got {z_h:.3f}")


print()
print("=" * 68)
print("PART 3 -- a steady player should sit near zero")
print("=" * 68)

rows = []
for b in range(10, 20):
    rows.extend(steady(b, 800, seed=b))
df = pd.DataFrame(rows)
out = add_form_deviation(df)
tail = out.dropna(subset=["form_z"])
mean_z = tail["form_z"].mean()
sd_z = tail["form_z"].std()
check("mean z across steady players is near 0",
      abs(mean_z) < 0.25, f"got {mean_z:.3f}")
# After standardisation this must be ~1.0 by construction. If it is not,
# the scaling step silently did not run.
check("sd of the standardised composite is ~1",
      0.9 < sd_z < 1.1, f"got {sd_z:.3f} -- standardisation did not apply")
check("the scale factor is recorded rather than hidden",
      "form_scale" in out.attrs and 0.2 < out.attrs["form_scale"] < 2.0,
      f"got {out.attrs.get('form_scale')}")

frac_flagged = ((tail["form_z"] >= HOT_Z) | (tail["form_z"] <= COLD_Z)).mean()
check("false-positive rate at the chosen threshold is a few percent",
      0.005 < frac_flagged < 0.10, f"got {frac_flagged:.3f}")
print(f"       (steady players flagged Hot or Cold {frac_flagged:.1%} of the "
      f"time -- this is the false-positive floor)")


print()
print("=" * 68)
print("PART 4 -- insufficient evidence is Unknown, not Normal")
print("=" * 68)

df = pd.DataFrame(make_pa(30, [{"k": 0, "h": 1}] * (FORM_MIN_PA - 2)))
out = add_form_deviation(df)
check("a batter below the PA floor has NaN z",
      out["form_z"].isna().all())
check("and is labelled Unknown rather than Normal",
      (out["form_state"] == "Unknown").all(),
      f"got {out['form_state'].unique()}")

# The very first PA of any career must be NaN, never 0.
rows = []
for b in range(40, 45):
    rows.extend(steady(b, 300, seed=b))
df = pd.DataFrame(rows)
out = add_form_deviation(df)
firsts = out.groupby("batter").head(1)
check("the first PA of every career is NaN, not a confident 0",
      firsts["form_z"].isna().all())


print()
print("=" * 68)
print("PART 5 -- no cross-player leakage")
print("=" * 68)

# One ice-cold player interleaved with one red-hot one. If the rolling
# window ever crosses the batter boundary, the cold player's z will be
# dragged upward.
cold_rows = make_pa(50, [{"k": 1, "h": 0}] * 400)
hot_rows = make_pa(51, [{"k": 0, "h": 1}] * 400)
mixed = pd.DataFrame(cold_rows + hot_rows).sample(frac=1.0, random_state=7)
# Restore the chronological-by-batter order the real pipeline guarantees.
mixed = mixed.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
out = add_form_deviation(mixed)
z_cold = out[out.batter == 50]["form_is_hit_z_short"].dropna()
check("the all-outs player never reads hot",
      (z_cold <= 0.5).all(),
      f"max {z_cold.max() if len(z_cold) else float('nan'):.3f}")

latest = latest_form_by_batter(out)
check("latest_form_by_batter returns exactly one row per batter",
      len(latest) == mixed["batter"].nunique() and
      latest["batter"].is_unique,
      f"got {len(latest)} rows for {mixed['batter'].nunique()} batters")


print()
print("=" * 68)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
print("=" * 68)
sys.exit(1 if FAIL else 0)
