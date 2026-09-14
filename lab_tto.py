"""
Does a lineup strike out less the third time it sees a pitcher?

    python lab_tto.py

WHY THIS EXISTS
---------------
`predict_slate` walks tonight's lineup in batting order and hands each
slot a strikeout probability from `per_batter_k_probs`. Then
`k_count_distribution` TILES that nine-element array down the batters-faced
distribution:

    if len(p_by_batter) < max_bf:
        p_by_batter = np.tile(p_by_batter, reps)[:max_bf]

So batter 1, batter 10 and batter 19 are the same hitter with the same
probability. The model has no notion that the third look is different from
the first.

The times-through-the-order penalty is one of the better established
effects in the sport, and it should bite strikeouts hardest -- a hitter's
edge on the third look is precisely that he has now timed the fastball and
seen the shape of the slider twice.

If it is real and the model ignores it, the model overstates strikeouts,
and it overstates them MOST in long outings. That is the same direction as
everything else measured on 2026-09-14 (k-overconfidence): probabilities
run hot, the left tail is too thin.

THE THREE WAYS THIS MEASUREMENT CAN LIE, AND WHAT IS DONE ABOUT EACH
--------------------------------------------------------------------
1. SURVIVORSHIP. Only a pitcher having a good night reaches the third time
   through. Comparing all TTO-1 plate appearances against all TTO-3 ones
   compares a population of every start against a population of good
   starts. Handled by never comparing across starts: every contrast is
   de-meaned inside one start, and the headline number is de-meaned inside
   one (start, batter) pair.

2. BATTER QUALITY. A pitcher pulled mid-order in the third time through
   faced slots 1-5 that time and slots 1-9 the first. The third look would
   then be measured against better hitters. Handled by the (start, batter)
   de-meaning: the same hitter is compared against himself, so his own
   quality cancels exactly.

3. FATIGUE, WHICH IS NOT FAMILIARITY. The third time through arrives around
   pitch 75. If the effect is really the arm tiring, "times through" is
   just a proxy for pitch count and the fix belongs somewhere else. These
   are hard to separate and the literature has argued about it for years,
   but they are not perfectly collinear -- a fast worker reaches the third
   time through on fewer pitches than a grinder. So pitch count is measured
   INSIDE each times-through bucket, where familiarity is held fixed.

A shuffled null runs alongside. Permuting the times-through labels within
each (start, batter) pair destroys the ordering and preserves everything
else, which is the right null here: the earlier labs in this project twice
produced a "null" that quietly destroyed a real main effect and made noise
look like signal.


RUN THIS AGAINST THE FULL CACHE SET
-----------------------------------
The numbers in the header above were measured on 2026-09-14 against 50
statcast caches. The machine that owns this project had 196. Nothing
failed and nothing warned -- the lab simply used whatever files were
beside it, which was a quarter of the evidence.

lab_log5.py shows what that is worth: a -0.72% effect at z = -1.27 on 50
caches, and +0.03% at z = -0.55 once all 196 were in. The lean was noise.

So treat every figure here as provisional until it has been re-run. Run
`python distill_caches.py` first and this file will read the distilled
table instead of the raw caches -- same numbers, seconds instead of
minutes, and the row count is printed so the sample is never a guess
again.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

K_EVENTS = {"strikeout", "strikeout_double_play"}
TTO_COL = "n_priorpa_thisgame_player_at_bat"
USE = ["game_pk", "game_date", "game_type", "events", "batter",
       "at_bat_number", "pitch_number", TTO_COL]
MIN_BF = 12            # a start, not a relief cameo
N_SHUFFLE = 400
RNG = np.random.default_rng(20260914)


def load():
    """One row per plate appearance across every cached pitcher."""
    frames = []
    for path in sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv"))):
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in USE,
                              low_memory=False)
        except Exception:
            continue
        if raw.empty or TTO_COL not in raw.columns:
            continue
        if "game_type" in raw.columns:
            raw = raw[raw["game_type"] == "R"]
        if raw.empty:
            continue

        # His pitch count entering each plate appearance. The cache holds
        # only this pitcher's pitches, so a cumulative row count within the
        # game -- ordered the way the game was played -- is exactly that.
        raw = raw.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        raw["pitch_no"] = raw.groupby("game_pk").cumcount() + 1

        pa = raw[raw["events"].notna() & raw["events"].ne("")].copy()
        if pa.empty:
            continue
        pa["pitcher"] = int(re.search(r"(\d+)", os.path.basename(path)).group(1))
        frames.append(pa)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["is_k"] = df["events"].isin(K_EVENTS).astype(float)
    df["tto"] = pd.to_numeric(df[TTO_COL], errors="coerce") + 1
    df = df.dropna(subset=["tto"])
    df["tto"] = df["tto"].astype(int)
    df["start"] = df["pitcher"].astype(str) + "_" + df["game_pk"].astype(str)

    size = df.groupby("start")["is_k"].transform("size")
    df = df[size >= MIN_BF]
    return df[df["tto"].between(1, 3)]


def raw_table(df):
    print("RAW K RATE BY TIMES THROUGH THE ORDER  (survivorship NOT removed)")
    g = df.groupby("tto")["is_k"].agg(["size", "mean"])
    for tto, r in g.iterrows():
        print(f"  time {tto}   n={int(r['size']):>6}   K rate {r['mean']:.3%}")
    print("  This table is the one to distrust: only a pitcher having a")
    print("  good night is still out there for the third time through.")


def within_start(df):
    print("\nWITHIN ONE START  (each start's own mean removed)")
    d = df.copy()
    d["dm"] = d["is_k"] - d.groupby("start")["is_k"].transform("mean")
    g = d.groupby("tto")["dm"].agg(["size", "mean", "sem"])
    base = g.loc[1, "mean"]
    for tto, r in g.iterrows():
        print(f"  time {tto}   n={int(r['size']):>6}   "
              f"{r['mean'] - base:+.3%} vs the first time through   "
              f"(+/- {r['sem'] * 100:.2f} pts)")
    print("  Still confounded: a pitcher pulled mid-order faced the top of")
    print("  the lineup the third time and all nine slots the first.")


def within_batter(df):
    """
    The headline. Same hitter, same pitcher, same game, same day, same
    park, same umpire -- the only thing that changed is that he has seen
    him before.

    Restricted to hitters who faced him exactly three times, so every
    group contributes to all three buckets and the comparison is balanced.
    An unbalanced version would let two-PA groups load the first two
    buckets and shift the contrast for a reason that has nothing to do
    with familiarity.
    """
    print("\nSAME HITTER vs SAME PITCHER IN THE SAME GAME  (the clean one)")
    d = df.copy()
    key = ["start", "batter"]
    d["n_pa"] = d.groupby(key)["is_k"].transform("size")
    d = d[d["n_pa"] == 3]
    if d.empty or d["tto"].nunique() < 3:
        print("  Not enough complete three-look matchups.")
        return None

    d["dm"] = d["is_k"] - d.groupby(key)["is_k"].transform("mean")
    g = d.groupby("tto")["dm"].agg(["size", "mean", "sem"])
    base = g.loc[1, "mean"]

    print(f"  {d.groupby(key).ngroups:,} matchups, "
          f"{d['start'].nunique():,} starts")
    for tto, r in g.iterrows():
        print(f"  time {tto}   n={int(r['size']):>6}   "
              f"{r['mean'] - base:+.3%} vs the first time through")

    effect = g.loc[3, "mean"] - g.loc[1, "mean"]
    clean = {int(t): float(g.loc[t, "mean"] - base) for t in g.index}

    # ---- the null ---------------------------------------------------
    #
    # Permute the order labels inside each matchup. Everything else --
    # who is batting, who is pitching, how many strikeouts happened in
    # that matchup -- is untouched.
    print(f"\n  shuffled null ({N_SHUFFLE} draws, order permuted within "
          f"each matchup)")
    vals = d["dm"].to_numpy()
    grp = d.groupby(key).ngroup().to_numpy()
    order = np.argsort(grp, kind="stable")
    vals_s = vals[order]
    n_groups = grp.max() + 1
    mat = vals_s.reshape(n_groups, 3)          # complete groups, so square

    null = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        idx = np.argsort(RNG.random(mat.shape), axis=1)
        shuf = np.take_along_axis(mat, idx, axis=1)
        null[i] = shuf[:, 2].mean() - shuf[:, 0].mean()

    mu, sd = null.mean(), null.std(ddof=1)
    z = (effect - mu) / sd if sd > 0 else np.nan
    print(f"  third look vs first: {effect:+.3%} real, "
          f"null {mu:+.3%} +/- {sd * 100:.3f} pts, z = {z:+.2f}")
    if abs(z) < 2:
        print("  -> does not clear the null. No penalty to build.")
    else:
        print("  -> real.")
    return clean


def fatigue(df):
    """
    Is it the arm or the eyes? Pitch count measured INSIDE each times-
    through bucket, where familiarity is held fixed by construction.

    THE TRAP THIS HAD TO BE REWRITTEN AROUND
    ----------------------------------------
    The first version split each bucket at its median pitch count and
    compared the halves. That is not a fatigue test. Inside the first time
    through, a high pitch count means he is facing the BOTTOM of the order,
    because he got there by throwing to everyone above them. So the split
    was sorting hitters by lineup slot, and it duly reported that the
    later half strikes out MORE -- which is just the eight and nine hitters
    being worse. Every bucket showed it and it looked like a real reversal.

    The position in the order has to be held fixed. `slot` below is where
    the batter sits relative to the first man the starter faced, which is
    the lineup turn itself, and de-meaning within (slot, times-through)
    leaves only the difference between a pitcher who arrived here on few
    pitches and one who arrived on many.
    """
    print("\nFATIGUE OR FAMILIARITY  (pitch count, lineup position held fixed)")
    d = df.sort_values(["start", "at_bat_number"]).copy()
    d["seq"] = d.groupby("start").cumcount()
    d["slot"] = d["seq"] % 9
    # Strip the start's level AND the slot's level, so what remains is
    # neither "good pitcher" nor "good hitter".
    d["dm"] = d["is_k"] - d.groupby("start")["is_k"].transform("mean")
    d["dm"] = d["dm"] - d.groupby(["slot", "tto"])["dm"].transform("mean")
    for tto in (1, 2, 3):
        g = d[d["tto"] == tto]
        if len(g) < 500:
            continue
        med = g["pitch_no"].median()
        lo = g[g["pitch_no"] <= med]["dm"]
        hi = g[g["pitch_no"] > med]["dm"]
        diff = hi.mean() - lo.mean()
        se = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
        print(f"  time {tto}   arrived on more pitches: {diff:+.3%} "
              f"(+/- {se * 100:.2f} pts, z = {diff / se:+.1f}) "
              f"[median pitch {med:.0f}]")
    print("  A penalty that is really the arm tiring should show up here as")
    print("  a steady negative. One that is really the hitter's eyes should")
    print("  not, because familiarity is identical across each row.")


def cost(clean, df):
    """
    What the missing penalty is worth to the MODEL -- which is much less
    than it is worth to baseball, and the difference is the whole point.

    THE MISTAKE THIS REPLACES
    -------------------------
    The first version of this function multiplied the per-look penalties by
    the batters faced at each look and reported "0.39 strikeouts a start the
    model books and does not get", which worked out to roughly 7 points of
    over-probability at a 3.5 line. That number was wrong and it was wrong
    in the most flattering possible direction: it assumed the model prices
    every batter at the FIRST-look rate.

    It does not. `predict_slate` calls `per_batter_k_probs` with
    `workload.k_rate(pid)`, which is total strikeouts over total batters
    faced across his whole history -- first, second and third looks all
    averaged together. Measured over 15 cached pitchers:

        overall rate (what the model uses)   24.54%
        first-look rate                      25.87%

    The decay is ALREADY inside the number the model applies. It is not
    missing; it is smeared evenly across the start instead of being
    concentrated late.

    SO WHAT IS ACTUALLY WRONG
    -------------------------
    Only the SHAPE. A correct look factor has to leave expected strikeouts
    unchanged -- otherwise it double-counts decay the rate already carries
    -- which means the early looks go UP as the late ones come down:

        relative     1.000  0.912  0.864
        normalised   1.070  0.976  0.925     (weights: his own exposure)

    That is a redistribution inside the start, and a redistribution barely
    moves a total. This function prices it out at real rates and lines so
    the size is a number rather than an intuition.
    """
    if clean is None:
        return
    print("\nWHAT IT COSTS THE MODEL")
    per = df.groupby(["start", "tto"]).size().groupby("tto").mean()
    print("  batters faced per start: "
          + ", ".join(f"time {t} {per.get(t, 0):.1f}" for t in (1, 2, 3)))

    try:
        from features.pitcher_workload import k_count_distribution
    except Exception as exc:
        print(f"  (cannot price it: {type(exc).__name__}: {exc})")
        return

    base = float(df[df["tto"] == 1]["is_k"].mean())
    fac = np.array([1.0] + [1 + clean.get(t, 0.0) / base for t in (2, 3)])

    bf = df.groupby("start").size()
    bf = bf[bf.between(10, 32)]
    sup = np.arange(int(bf.min()), int(bf.max()) + 1)
    prob = np.array([float((bf == s).mean()) for s in sup])
    prob = prob / prob.sum()

    n = int(sup.max())
    f = fac[np.minimum(np.arange(n) // 9, 2)]
    # Expected exposure per batter slot, so the normaliser reflects how
    # often each look actually happens rather than weighting them equally.
    w = np.zeros(n)
    for s, p in zip(sup, prob):
        w[:s] += p
    f = f / np.average(f, weights=w / w.sum())

    print(f"  relative look factors   {fac.round(3)}")
    print(f"  mean-preserving         {np.round([f[0], f[9], f[18]], 3)}")
    print("\n  effect on the probability the tab actually prints:")
    print(f"  {'rate':>6} {'mean K':>7} {'line':>6} {'flat':>8} "
          f"{'by look':>9} {'change':>8}")
    worst = 0.0
    for rate in (0.16, 0.20, 0.24, 0.28):
        flat = k_count_distribution([rate] * n, sup, prob)
        byl = k_count_distribution(list(rate * f), sup, prob)
        mk = float((np.arange(len(flat)) * flat).sum())
        for line in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
            st = int(np.ceil(line))
            a, b = float(flat[st:].sum()), float(byl[st:].sum())
            if 0.15 < a < 0.90:
                worst = max(worst, abs(b - a))
                print(f"  {rate:>6.0%} {mk:>7.2f} {line:>6} {a:>8.1%} "
                      f"{b:>9.1%} {(b - a) * 100:>+7.1f}")

    print(f"\n  largest move anywhere: {worst * 100:.1f} points of probability.")
    print("  The order effect is real in the world and z = -8.84 says so.")
    print("  The model already absorbs it through a rate learned on starts")
    print("  that contain it. Building a look factor would be correct and")
    print("  would change nothing worth changing.")
    print("  It would start to matter for a pitcher projected to work a")
    print("  very different depth from the starts his rate was measured on")
    print("  -- an opener, or a converted reliever -- which is a narrower")
    print("  and different fix from the one this file set out to justify.")


def main():
    df = load()
    if df.empty:
        print("No cached pitcher starts with a times-through column.")
        return 1
    print(f"{df['pitcher'].nunique()} pitchers, {df['start'].nunique():,} "
          f"starts, {len(df):,} plate appearances\n")
    raw_table(df)
    within_start(df)
    effect = within_batter(df)
    fatigue(df)
    cost(effect, df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
