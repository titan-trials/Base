"""
Are bad nights also short nights, and does the model know?

    python lab_coupling.py

THE ASSUMPTION UNDER TEST
-------------------------
`k_count_distribution` builds a start out of two pieces:

    how good is he tonight   a mixture over multipliers, K_DISPERSION_SD
    how long does he last    bf_probs, the batters-faced distribution

and combines them independently. The mixture loop reads:

    for mult, mw in zip(multipliers, mult_weights):
        ...
        for faced in range(max_bf + 1):
            w = weight_of.get(faced)      # <- the SAME bf_probs every time

Every version of tonight -- the sharp one and the flat one -- is handed an
identical distribution for how deep he goes. In real baseball they are
welded together: a pitcher getting hit hard in the third is gone in the
fourth. Bad and short arrive as a package.

WHY IT WOULD PRODUCE EXACTLY THE OBSERVED SYMPTOM
-------------------------------------------------
If the two are coupled and the model treats them as independent, the model
is missing the scenario where a low rate and a low batter count multiply
together. That scenario is the disaster start, and its absence makes the
left tail too thin -- which is what k-overconfidence-2026-09-14.md measured:

    9.8% of starts land >1.5 sd BELOW projection   (6.7% expected)
    2.9% land >1.5 sd above                        (6.7% expected)

The right tail needs no such mechanism, because the upside is capped by
the manager: a pitcher cruising through five still comes out at 100
pitches. Only the downside compounds.

WHAT THIS MEASURES
------------------
1. COUPLE   within a pitcher, is a start's rate residual correlated with
            its batters-faced residual?
2. NULL     how much of that correlation is mechanical? A rate is K/BF and
            a short start has a noisier rate, so some relationship appears
            even under perfect independence. The null simulates strikeouts
            drawn independently of the outing length and re-measures.
3. COST     does adding the measured coupling to the model's own
            distribution machinery reproduce the observed tail asymmetry?

Step 3 is the one that decides anything. A real correlation that moves the
tails by a tenth of a point is lab_tto.py all over again -- true about
baseball, irrelevant to this model -- and the only way to tell the two
apart is to price it.


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
USE = ["game_pk", "game_date", "game_type", "events"]
MIN_BF = 10
MIN_PRIOR = 8              # starts of history before a residual means anything
N_SIM = 400
RNG = np.random.default_rng(20260914)


def per_start():
    # Prefer the distilled table when distill_caches.py has been run: same
    # rows, seconds instead of minutes, and the sample size is printed so
    # it can never again be whatever files happened to be on disk.
    try:
        import distilled
        got = distilled.starts()
    except Exception:
        got = None
    if got is not None and not got.empty:
        # The distiller keeps everything down to 8 batters faced so one
        # table can serve labs with different thresholds. Each lab still
        # owes its own filter -- without this the distilled path silently
        # analysed 4,408 starts where the raw path analysed 4,198, and the
        # two would have disagreed for a reason that had nothing to do
        # with the question being asked.
        return got[got["bf"] >= MIN_BF]
    rows = []
    for path in sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv"))):
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in USE,
                              low_memory=False)
        except Exception:
            continue
        if raw.empty or "events" not in raw.columns:
            continue
        if "game_type" in raw.columns:
            raw = raw[raw["game_type"] == "R"]
        raw = raw[raw["events"].notna() & raw["events"].ne("")]
        if raw.empty:
            continue
        g = raw.assign(k=raw["events"].isin(K_EVENTS)).groupby("game_pk").agg(
            date=("game_date", "first"), bf=("k", "size"), k=("k", "sum"))
        g = g[g["bf"] >= MIN_BF]
        if g.empty:
            continue
        g["pitcher"] = int(re.search(r"(\d+)",
                                     os.path.basename(path)).group(1))
        rows.append(g.reset_index())
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values(["pitcher", "date"])


def residuals(s):
    """
    Each start against what his PRIOR starts predicted.

    shift(1).expanding() throughout: a residual computed against a mean
    that includes the start itself is guaranteed to correlate with
    everything, which is the standard way this measurement goes wrong.
    """
    f = s.copy()
    g = f.groupby("pitcher", group_keys=False)
    f["exp_bf"] = g["bf"].apply(lambda x: x.shift(1).expanding().mean())
    f["prior_k"] = g["k"].apply(lambda x: x.shift(1).expanding().sum())
    f["prior_bf"] = g["bf"].apply(lambda x: x.shift(1).expanding().sum())
    f["n_prior"] = g.cumcount()
    f = f[(f["n_prior"] >= MIN_PRIOR) & f["exp_bf"].notna()
          & (f["prior_bf"] > 0)]
    f["exp_rate"] = f["prior_k"] / f["prior_bf"]
    f["rate"] = f["k"] / f["bf"]
    f["rate_res"] = f["rate"] - f["exp_rate"]
    f["bf_res"] = f["bf"] - f["exp_bf"]
    return f


def couple(f):
    print("1. COUPLE  — is a bad night also a short night?")
    r = float(np.corrcoef(f["rate_res"], f["bf_res"])[0, 1])
    print(f"  {len(f):,} starts, {f.pitcher.nunique()} pitchers")
    print(f"  corr(rate residual, batters-faced residual) = {r:+.4f}")
    print(f"  mean BF {f.bf.mean():.1f}, mean rate {f.rate.mean():.1%}")

    print("\n  by how long he lasted:")
    f = f.copy()
    f["band"] = pd.qcut(f["bf"], 5,
                        labels=["shortest", "short", "mid", "long", "longest"])
    for lab, g in f.groupby("band", observed=True):
        se = g["rate_res"].std(ddof=1) / np.sqrt(len(g))
        print(f"    {str(lab):>9}  n={len(g):>4}  BF {g.bf.mean():>4.1f}   "
              f"rate vs his own norm {g.rate_res.mean():+.2%} "
              f"(+/- {se * 100:.2f} pts)")
    return r


def null(f, observed):
    """
    How much of that correlation is arithmetic rather than baseball?

    Strikeouts are redrawn from his expected rate at his ACTUAL batters
    faced -- so outing length is untouched and the rate is independent of
    it by construction. Anything the correlation shows in this world is
    mechanical: short starts have noisier rates, and a rate has BF in its
    denominator.
    """
    print("\n2. NULL  — how much of that is just arithmetic?")
    bf = f["bf"].to_numpy()
    p = f["exp_rate"].to_numpy()
    bf_res = f["bf_res"].to_numpy()
    out = np.empty(N_SIM)
    for i in range(N_SIM):
        k = RNG.binomial(bf, np.clip(p, 1e-6, 1 - 1e-6))
        out[i] = np.corrcoef(k / bf - p, bf_res)[0, 1]
    mu, sd = out.mean(), out.std(ddof=1)
    z = (observed - mu) / sd
    print(f"  independent-world correlation: {mu:+.4f} +/- {sd:.4f}")
    print(f"  observed {observed:+.4f}   z = {z:+.2f}")
    print("  -> " + ("real coupling, beyond the arithmetic"
                     if z > 2 else "no coupling beyond the arithmetic"))
    return z


def cost(f):
    """
    Price it with the model's own machinery.

    Two worlds, same pitcher, same average outing:
      independent  a form multiplier drawn separately from outing length
      coupled      the flat nights are also the short ones, at the
                   measured strength
    Then compare each world's tails against what really happens.
    """
    print("\n3. COST  — what the coupling does to the distribution")
    try:
        from features.pitcher_workload import (k_count_distribution,
                                               K_DISPERSION_SD)
    except Exception as exc:
        print(f"  (cannot price it: {type(exc).__name__}: {exc})")
        return

    bfd = f["bf"].astype(int)
    sup = np.arange(int(bfd.min()), int(bfd.max()) + 1)
    prob = np.array([float((bfd == s).mean()) for s in sup])
    prob = prob / prob.sum()
    rate = float(f["rate"].mean())

    # The model as it stands: one bf distribution for every form draw.
    indep = k_count_distribution([rate] * int(sup.max()), sup, prob)

    # Coupled: walk the same form mixture, but slide the batters-faced
    # distribution with the multiplier. `slide` is calibrated so the
    # implied correlation matches the measured one.
    offsets = np.linspace(-2.0, 2.0, 9)
    wts = np.exp(-0.5 * offsets ** 2)
    wts /= wts.sum()
    obs_r = float(np.corrcoef(f["rate_res"], f["bf_res"])[0, 1])
    sd_bf = float(f["bf_res"].std(ddof=1))
    # A one-sigma bad night costs this many batters.
    slide = obs_r * sd_bf

    total = np.zeros(int(sup.max()) + 1)
    for off, w in zip(offsets, wts):
        mult = 1.0 + K_DISPERSION_SD * off
        shifted = sup + slide * off
        # Re-bin the shifted distribution onto the integer support.
        pr = np.zeros_like(prob)
        for s, p0 in zip(shifted, prob):
            lo = int(np.floor(np.clip(s, sup.min(), sup.max())))
            hi = min(lo + 1, sup.max())
            frac = float(np.clip(s, sup.min(), sup.max())) - lo
            pr[lo - sup.min()] += p0 * (1 - frac)
            pr[hi - sup.min()] += p0 * frac
        pr = pr / pr.sum()
        d = k_count_distribution([rate * mult] * int(sup.max()), sup, pr,
                                 dispersion_sd=0.0)
        total[:len(d)] += w * d
    total /= total.sum()

    def tails(dist):
        x = np.arange(len(dist))
        m = float((x * dist).sum())
        sd = float(np.sqrt((((x - m) ** 2) * dist).sum()))
        lo = float(dist[x < m - 1.5 * sd].sum())
        hi = float(dist[x > m + 1.5 * sd].sum())
        return m, sd, lo, hi

    mi, si, li, hi_ = tails(indep)
    mc, sc, lc, hc = tails(total)

    # What really happens, standardised the same way.
    z = (f["k"] - f["exp_rate"] * f["bf"])
    z = z / z.std(ddof=1)
    act_lo = float((z < -1.5).mean())
    act_hi = float((z > 1.5).mean())

    print(f"  measured coupling: a one-sigma flat night is "
          f"{abs(slide):.1f} batters shorter")
    print(f"\n  {'':>14} {'mean':>6} {'sd':>6} {'>1.5sd low':>11} "
          f"{'>1.5sd high':>12}")
    print(f"  {'model today':>14} {mi:>6.2f} {si:>6.2f} {li:>10.1%} "
          f"{hi_:>11.1%}")
    print(f"  {'with coupling':>14} {mc:>6.2f} {sc:>6.2f} {lc:>10.1%} "
          f"{hc:>11.1%}")
    print(f"  {'what happens':>14} {'':>6} {'':>6} {act_lo:>10.1%} "
          f"{act_hi:>11.1%}")

    gap_before = abs(li - act_lo)
    gap_after = abs(lc - act_lo)
    print(f"\n  left-tail gap: {gap_before * 100:.1f} pts -> "
          f"{gap_after * 100:.1f} pts")
    if gap_after < gap_before * 0.7:
        print("  -> the coupling explains a real share of the missing tail.")
    elif gap_after < gap_before:
        print("  -> helps, but does not close it.")
    else:
        print("  -> does not explain the missing tail.")


def main():
    s = per_start()
    if s.empty:
        print("No cached starts.")
        return 1
    f = residuals(s)
    if f.empty:
        print("No starts with enough history.")
        return 1
    r = couple(f)
    null(f, r)
    cost(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
