"""
Does every pitcher decay the same way the third time through?

    python lab_tto_who.py

THE QUESTION
------------
lab_tto.py measured the order penalty league-wide: the same hitter, against
the same pitcher, in the same game, strikes out 3.46 points less on his
third look than his first (z = -8.84 against a permuted null). The model has
no term for it and books about 0.39 strikeouts a start it does not get.

The obvious fix is one multiplier applied to every pitcher. That assumes
every pitcher decays identically, which is its own untested claim -- and the
intuition says otherwise. A man with one fastball and one slider should be
easier to solve on the third look than a man with six pitches. If that is
true, a single league-wide multiplier is wrong in both directions at once:
too small for the two-pitch arms, too large for the deep ones.

So, in order:

  1. SPREAD    do per-pitcher penalties vary more than sampling noise?
  2. PERSIST   does a pitcher's penalty repeat on held-out starts?
  3. EXPLAIN   does arsenal depth predict who decays hardest?

Two and three are the ones that decide whether anything gets built. This
project has now twice found a real-looking effect that did not survive a
split-half test -- the vs-opponent pairing (lab_opponent.py, r = +0.019
against a null of -0.001 +/- 0.030) being the closest analogue. A spread
that does not persist is a spread of measurement error, and shipping a
per-pitcher multiplier built on it would add noise to every start.

The order also matters. If 1 fails there is nothing to explain and 3 would
be fitting arsenal features to sampling error, which with 50 pitchers and
three candidate features would find something about a third of the time.

NULLS
-----
Parametric, not label shuffles. Every null here simulates each pitcher's
matchups from HIS OWN first-look strikeout rate with the POOLED penalty
imposed on top -- so the null world is "every pitcher decays identically,"
which is exactly the hypothesis being tested, and it preserves the fact
that a high-strikeout pitcher has noisier paired differences than a soft
one. A plain shuffle of pitcher labels would destroy that and make the
observed spread look significant on base-rate differences alone.


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
USE = ["game_pk", "game_type", "events", "batter", "at_bat_number",
       "pitch_number", "pitch_type", TTO_COL]
MIN_BF = 12
MIN_MATCHUPS = 150         # below this a per-pitcher penalty is a rumour
FASTBALLS = {"FF", "SI", "FT", "FC"}
N_SIM = 500
RNG = np.random.default_rng(20260914)


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load():
    """
    Plate appearances and arsenal, in one pass over the caches.

    Both are wanted per pitcher and the files are ~6 MB each, so reading
    them twice to build two frames is a minute of nothing.
    """
    pa_frames, arsenal = [], []
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
        pid = int(re.search(r"(\d+)", os.path.basename(path)).group(1))

        # ---- arsenal, measured on every pitch he threw ---------------
        if "pitch_type" in raw.columns:
            mix = raw["pitch_type"].dropna()
            mix = mix[mix.ne("") & ~mix.isin({"PO", "IN", "UN"})]
            if len(mix) >= 500:
                share = mix.value_counts(normalize=True)
                # Shannon entropy of the pitch mix: one number for "how
                # many different things does he throw, weighted by how
                # often". A 95/5 two-pitch mix scores far below a 40/30/30
                # three-pitch mix, which a raw count of pitch types would
                # miss entirely.
                ent = float(-(share * np.log(share)).sum())
                arsenal.append({
                    "pitcher": pid,
                    "n_pitch_types": int((share >= 0.10).sum()),
                    "mix_entropy": ent,
                    "fastball_share": float(
                        share[share.index.isin(FASTBALLS)].sum()),
                    "pitches": int(len(mix)),
                })

        # ---- plate appearances ---------------------------------------
        raw = raw.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        pa = raw[raw["events"].notna() & raw["events"].ne("")].copy()
        if pa.empty:
            continue
        pa["pitcher"] = pid
        pa_frames.append(pa[["game_pk", "batter", "events", TTO_COL,
                             "pitcher"]])

    if not pa_frames:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.concat(pa_frames, ignore_index=True)
    df["is_k"] = df["events"].isin(K_EVENTS).astype(float)
    df["tto"] = pd.to_numeric(df[TTO_COL], errors="coerce") + 1
    df = df.dropna(subset=["tto"])
    df["tto"] = df["tto"].astype(int)
    df["start"] = df["pitcher"].astype(str) + "_" + df["game_pk"].astype(str)
    df = df[df.groupby("start")["is_k"].transform("size") >= MIN_BF]
    return df[df["tto"].between(1, 3)], pd.DataFrame(arsenal)


def matchups(df):
    """
    One row per (start, batter) seen exactly three times.

    Columns k1/k2/k3 are that hitter's outcome on each look. The paired
    difference k3 - k1 is the penalty for that matchup with the hitter,
    the pitcher, the park, the day and the umpire all differenced away.
    """
    d = df.copy()
    key = ["start", "batter"]
    d = d[d.groupby(key)["is_k"].transform("size") == 3]
    if d.empty:
        return pd.DataFrame()
    wide = (d.pivot_table(index=["pitcher", "start", "batter"],
                          columns="tto", values="is_k", aggfunc="first")
              .rename(columns={1: "k1", 2: "k2", 3: "k3"})
              .dropna()
              .reset_index())
    wide["d31"] = wide["k3"] - wide["k1"]
    wide["d21"] = wide["k2"] - wide["k1"]
    return wide


# ----------------------------------------------------------------------
# 1. spread
# ----------------------------------------------------------------------
def simulate(per, pooled, n_sim=N_SIM):
    """
    Null worlds in which every pitcher has the SAME penalty.

    Returns an (n_sim, n_pitchers) array of simulated per-pitcher penalty
    estimates, each built from that pitcher's own first-look rate and his
    own number of matchups.
    """
    p1 = per["p1"].to_numpy()
    n = per["n"].to_numpy()
    p3 = np.clip(p1 + pooled, 1e-6, 1 - 1e-6)
    out = np.empty((n_sim, len(per)))
    for j, (a, b, m) in enumerate(zip(p1, p3, n)):
        k1 = RNG.binomial(1, a, size=(n_sim, m))
        k3 = RNG.binomial(1, b, size=(n_sim, m))
        out[:, j] = (k3 - k1).mean(axis=1)
    return out


def spread(w):
    print("1. SPREAD  — do pitchers differ more than sampling noise?")
    g = w.groupby("pitcher")
    per = pd.DataFrame({
        "n": g.size(),
        "pen": g["d31"].mean(),
        "se": g["d31"].std(ddof=1) / np.sqrt(g.size()),
        "p1": g["k1"].mean(),
    }).reset_index()
    per = per[per["n"] >= MIN_MATCHUPS].reset_index(drop=True)
    pooled = float(w["d31"].mean())

    obs_sd = float(per["pen"].std(ddof=1))
    print(f"  {len(per)} pitchers with {MIN_MATCHUPS}+ complete matchups "
          f"(median {per['n'].median():.0f} each)")
    print(f"  pooled penalty {pooled:+.3%}")
    print(f"  observed spread of per-pitcher penalties: {obs_sd * 100:.2f} pts")
    print(f"  typical sampling error on one pitcher:    "
          f"{per['se'].mean() * 100:.2f} pts")

    sim = simulate(per, pooled)
    null_sd = sim.std(axis=1, ddof=1)
    z = (obs_sd - null_sd.mean()) / null_sd.std(ddof=1)
    print(f"  null spread (identical penalties): "
          f"{null_sd.mean() * 100:.2f} +/- {null_sd.std(ddof=1) * 100:.2f} pts")
    print(f"  z = {z:+.2f}")
    # Variance decomposition: what is left after removing the part that
    # measurement error alone would produce.
    true_var = obs_sd ** 2 - (per["se"] ** 2).mean()
    if true_var > 0:
        print(f"  implied TRUE spread between pitchers: "
              f"{np.sqrt(true_var) * 100:.2f} pts")
    else:
        print("  implied TRUE spread between pitchers: none — observed "
              "spread is at or below what noise alone produces")
    print("  -> " + ("real spread" if z > 2 else
                     "no more spread than noise; a single league-wide "
                     "penalty is the honest model"))
    return per, pooled, z


# ----------------------------------------------------------------------
# 2. persistence
# ----------------------------------------------------------------------
def persist(w, per, pooled):
    """
    Split each pitcher's GAMES in two and correlate the two penalties.

    Games, not matchups: two hitters in the same start share that night's
    weather, umpire and whatever the pitcher had that day, so splitting
    matchups at random would put correlated observations on both sides and
    manufacture persistence out of within-game clustering. That is the
    mistake lab_opponent.py made in its first version.
    """
    print("\n2. PERSIST  — does a pitcher's penalty repeat on other starts?")
    d = w.copy()
    # Alternate whole games, so the halves are interleaved in time and a
    # pitcher who changed mid-season does not get his early self in one
    # half and his late self in the other.
    codes = d.groupby("pitcher")["start"].transform(
        lambda s: pd.factorize(s)[0])
    d["half"] = codes % 2

    piv = (d.groupby(["pitcher", "half"])["d31"].agg(["mean", "size"])
             .unstack("half"))
    piv.columns = ["a", "b", "na", "nb"]
    piv = piv.dropna()
    piv = piv[(piv["na"] >= MIN_MATCHUPS / 2) & (piv["nb"] >= MIN_MATCHUPS / 2)]
    if len(piv) < 12:
        print("  too few pitchers with both halves populated.")
        return None
    r = float(np.corrcoef(piv["a"], piv["b"])[0, 1])
    print(f"  {len(piv)} pitchers, {piv['na'].median():.0f}/"
          f"{piv['nb'].median():.0f} matchups per half")
    print(f"  split-half r = {r:+.4f}")

    # Null: identical penalties for everyone, same half sizes.
    sub = per.set_index("pitcher").loc[piv.index]
    null = np.empty(N_SIM)
    p1 = sub["p1"].to_numpy()
    p3 = np.clip(p1 + pooled, 1e-6, 1 - 1e-6)
    na, nb = piv["na"].to_numpy().astype(int), piv["nb"].to_numpy().astype(int)
    for i in range(N_SIM):
        a = np.array([RNG.binomial(1, q, m).mean() - RNG.binomial(1, p, m).mean()
                      for p, q, m in zip(p1, p3, na)])
        b = np.array([RNG.binomial(1, q, m).mean() - RNG.binomial(1, p, m).mean()
                      for p, q, m in zip(p1, p3, nb)])
        null[i] = np.corrcoef(a, b)[0, 1]
    z = (r - null.mean()) / null.std(ddof=1)
    print(f"  null r = {null.mean():+.4f} +/- {null.std(ddof=1):.4f}, "
          f"z = {z:+.2f}")
    print("  -> " + ("persists — a per-pitcher penalty is learnable"
                     if z > 2 else
                     "does NOT persist — the per-pitcher spread is "
                     "measurement error"))
    return z


# ----------------------------------------------------------------------
# 3. arsenal
# ----------------------------------------------------------------------
def explain(per, ars, persist_z):
    print("\n3. EXPLAIN  — does arsenal depth predict who decays hardest?")
    if ars.empty:
        print("  no arsenal data.")
        return
    m = per.merge(ars, on="pitcher", how="inner")
    if len(m) < 20:
        print(f"  only {len(m)} pitchers matched; not enough to correlate.")
        return
    if persist_z is not None and persist_z <= 2:
        print("  Reported for completeness only. The penalty did not persist")
        print("  above, so the per-pitcher numbers these are correlated")
        print("  against are mostly noise, and a hit here would be a")
        print("  coincidence among three candidate features on "
              f"{len(m)} pitchers.")
    print(f"  {len(m)} pitchers")
    # A pitcher with a bigger penalty has a MORE NEGATIVE `pen`, so a
    # positive r means "more of this feature goes with a smaller penalty".
    # `p1` is deliberately NOT in this list. It is his own first-look
    # strikeout rate, and `pen` is mean(k3) - mean(k1) -- the two share the
    # k1 term, so a pitcher whose k1 ran high by chance gets a negative pen
    # by the same chance. Correlating them measures regression to the mean
    # and returns a confident number every time. It is asked properly in
    # section 4, against a base rate estimated on different games.
    for col, label in (("mix_entropy", "pitch-mix entropy (deeper arsenal)"),
                       ("n_pitch_types", "pitch types thrown 10%+ of the time"),
                       ("fastball_share", "fastball share")):
        if col not in m.columns:
            continue
        x, y = m[col].to_numpy(), m["pen"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        # Fisher z for a rough two-sided p.
        n = len(m)
        se = 1 / np.sqrt(n - 3)
        zz = np.arctanh(r) / se
        print(f"  {label:42} r = {r:+.3f}   z = {zz:+.2f}")
    print("  With three candidate features on this many pitchers, one r")
    print("  near 0.28 is what chance alone produces about a third of the")
    print("  time. Nothing here should be built without its own split-half.")


# ----------------------------------------------------------------------
# 4. the shape of the penalty
# ----------------------------------------------------------------------
def shape(w):
    """
    Is the penalty a fixed number of POINTS, or a fixed FRACTION?

    This decides how the fix is written and it is not the same question as
    "do pitchers differ". A single league-wide effect can still land
    differently on different pitchers:

        additive         every starter loses 3.46 points of K rate
        multiplicative   every starter keeps ~86% of his rate

    On a 28% strikeout pitcher those are 24.5% and 24.1% -- nearly the same.
    On a 15% pitcher they are 11.5% and 12.9%, which over 24 batters is a
    third of a strikeout, at exactly the low lines where the model is
    already measured to run hot. So the wrong choice here would put most of
    its error in the worst place.

    THE TRAP
    --------
    The obvious test -- correlate each pitcher's first-look rate against his
    own penalty -- is fatally circular: penalty = mean(k3) - mean(k1), so
    the k1 noise appears on both sides with opposite signs. It returned
    r = -0.437, z = -2.93 in an earlier version of this file and it means
    nothing. Section 3's note explains why it was removed from there.

    Here the base rate comes from a pitcher's OTHER games: split his starts
    into alternating halves, estimate the rate on one half and the penalty
    on the other, then do it again with the halves swapped so nothing is
    wasted. The two sides then share no plate appearance and no noise.
    """
    print("\n4. SHAPE  — a fixed number of points, or a fixed fraction?")
    d = w.copy()
    codes = d.groupby("pitcher")["start"].transform(
        lambda s: pd.factorize(s)[0])
    d["half"] = codes % 2

    rows = []
    for base_half in (0, 1):
        b = d[d["half"] == base_half]
        t = d[d["half"] != base_half]
        # Base rate from ALL looks in the base half: three times the sample
        # of first-looks alone, so much less attenuation.
        base = (b.melt(id_vars="pitcher", value_vars=["k1", "k2", "k3"],
                       value_name="k").groupby("pitcher")["k"]
                 .agg(["mean", "size"]))
        tgt = t.groupby("pitcher").agg(k1=("k1", "mean"), k3=("k3", "mean"),
                                       n=("k1", "size"))
        j = base.join(tgt, how="inner").dropna()
        j = j[(j["size"] >= 150) & (j["n"] >= 75)]
        for pid, r in j.iterrows():
            rows.append({"pitcher": pid, "base": r["mean"], "k1": r["k1"],
                         "k3": r["k3"], "pen": r["k3"] - r["k1"],
                         "n": r["n"]})
    m = pd.DataFrame(rows)
    if len(m) < 24:
        print("  not enough pitcher-halves.")
        return

    r = float(np.corrcoef(m["base"], m["pen"])[0, 1])
    zz = np.arctanh(r) * np.sqrt(len(m) - 3)
    print(f"  {len(m)} pitcher-halves ({m['pitcher'].nunique()} pitchers, "
          f"each contributing both directions)")
    print(f"  held-out K rate vs penalty: r = {r:+.3f}  z = {zz:+.2f}")
    print("  (negative would mean the better strikeout pitchers lose MORE "
          "points, which is what a multiplicative penalty looks like)")

    # Terciles on the held-out rate, so the grouping cannot be contaminated
    # by the penalty being measured.
    pooled_pen = float((m["k3"] - m["k1"]).mean())
    ratio = float(m["k3"].sum() / m["k1"].sum())
    m["band"] = pd.qcut(m["base"], 3, labels=["soft", "middle", "power"])
    print(f"\n  {'held-out K rate':>16}  {'n':>4}  {'actual':>8}  "
          f"{'additive':>9}  {'x-factor':>9}")
    sse_a = sse_m = 0.0
    for lab, g in m.groupby("band", observed=True):
        act = float(g["pen"].mean())
        pred_m = float(g["k1"].mean() * (ratio - 1.0))
        sse_a += (act - pooled_pen) ** 2
        sse_m += (act - pred_m) ** 2
        print(f"  {str(lab):>16}  {len(g):>4}  {act:>+7.2%}  "
              f"{pooled_pen:>+8.2%}  {pred_m:>+8.2%}")
    print(f"\n  league-wide penalty {pooled_pen:+.2%} of K rate, "
          f"or keep {ratio:.3f} of it")
    better = "multiplicative" if sse_m < sse_a else "additive"
    print(f"  across the three bands the {better} form fits better "
          f"(SSE {min(sse_a, sse_m):.2e} vs {max(sse_a, sse_m):.2e})")
    if abs(zz) < 2:
        print("  but the correlation does not clear 2 sigma, so this is a")
        print("  preference, not a finding. Both forms are inside the noise")
        print("  and the simpler one is defensible.")


def main():
    # Distilled first, for the reason in distilled.py: this file's
    # published numbers came off 42 pitchers because the sandbox held 50
    # of 196 caches, and a loader that announces its sample size is the
    # cheapest guard against that happening twice.
    w = ars = None
    try:
        import distilled
        w, ars = distilled.matchups(), distilled.arsenal()
    except Exception:
        w = ars = None
    if w is None or w.empty:
        df, a2 = load()
        if df.empty:
            print("No cached starts with a times-through column.")
            return 1
        w, ars = matchups(df), a2
    if w is None or w.empty:
        print("No complete three-look matchups.")
        return 1
    if ars is None:
        ars = pd.DataFrame()
    print(f"{w['pitcher'].nunique()} pitchers, {len(w):,} complete "
          f"three-look matchups\n")
    per, pooled, _ = spread(w)
    pz = persist(w, per, pooled)
    explain(per, ars, pz)
    shape(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
