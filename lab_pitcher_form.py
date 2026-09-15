"""
Would a hot/cold marker work for a STARTER's strikeouts?

    python lab_pitcher_form.py

WHY TEST BEFORE BUILDING
------------------------
The hitter marker has been live for 13 slates and shows nothing:

    Hot    predicted 36.0%   actual 33.5%
    Cold   predicted 33.9%   actual 32.1%
    HOT minus COLD residual  -0.71 points +/- 2.32,  z = -0.31

Both groups undershot by about the same amount, which is the model's own
calibration drift rather than a form effect, and the point estimate is the
wrong sign. That is not a kill -- 490 hot player-props is a small sample --
but it is not a green light to build a second one on faith either.

And the pitcher version has a much worse sampling problem in the live log:
fifteen starters a night against 270 hitters, so a self-testing marker
would take the better part of a decade to speak.

The way out is that pitchers leave a record hitters do not. Every start is
in cache/statcast_pitcher_*.csv going back to 2022 -- 50 cached starters,
4,599 starts -- so the question can be answered NOW, on history, instead
of by shipping a column and waiting.

THE TWO CANDIDATE SIGNALS
-------------------------
    velocity   mean fastball speed over his last few starts, against his
               own season baseline. This is the one worth caring about:
               it is a direct PHYSICAL measurement, not an outcome
               filtered through defence and luck. context-features-
               availability.md makes exactly this argument for exit
               velocity on the hitter side -- "a hitter whose exit
               velocity is up 3 mph over 15 games has changed something
               physical" -- and notes that the decline half (fatigue,
               a nagging injury) has causes a hot streak does not.

    recent K   his strikeout rate over the last few starts against the
               same baseline. The obvious version, and the one most
               likely to be pure noise re-labelled.

WHAT WOULD COUNT AS A RESULT
----------------------------
Not "hot pitchers strike out more" -- of course they do, they are better
pitchers. The baseline already knows that. The question is whether the
DEVIATION predicts the RESIDUAL: given what his trailing rate says he is,
does being 1.2 mph down on his fastball tell you he will miss fewer bats
tonight than that rate implies?

Everything is computed from strictly PRIOR starts. A form feature that
peeks at tonight will find a beautiful effect and mean nothing.


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
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

K_EVENTS = {"strikeout", "strikeout_double_play"}
USE = ["game_pk", "game_date", "game_type", "pitch_type", "release_speed",
       "events", "player_name", "inning_topbot", "home_team", "away_team"]
# Four-seam, sinker, cutter. A changeup's speed moves with the fastball
# and a curveball's does not, so mixing them in measures pitch selection
# rather than arm speed.
FAST = {"FF", "SI", "FT", "FC"}
MIN_BF = 12            # below this it is a relief outing in the cache
MIN_PRIOR = 5          # starts of history before a baseline means anything
RECENT = 3             # how many prior starts make up "recent form"


def per_start():
    # Prefer the distilled table, like lab_shrinkage and lab_coupling. This
    # was the last lab still opening 196 raw caches, and it is the lab whose
    # conclusion changed most when the sample grew.
    try:
        import distilled
        got = distilled.starts()
    except Exception:
        got = None
    if got is not None and not got.empty:
        return got[got["bf"] >= MIN_BF]
    """One row per (pitcher, start): date, batters faced, K, fastball mph."""
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
        if raw.empty:
            continue
        raw = raw.assign(
            is_pa=raw["events"].notna() & raw["events"].ne(""),
            is_k=raw["events"].isin(K_EVENTS),
            fast=raw["pitch_type"].isin(FAST) if "pitch_type" in raw else False)
        g = raw.groupby("game_pk").agg(
            date=("game_date", "first"),
            bf=("is_pa", "sum"), k=("is_k", "sum"),
            pitches=("events", "size"),
            velo=("release_speed", lambda s: np.nan))
        # Fastball-only velocity, computed separately so a start with no
        # fastballs comes back NaN rather than as the mean of his breaking
        # stuff.
        fb = raw[raw["fast"]].groupby("game_pk")["release_speed"].mean()
        g["velo"] = fb
        g = g[g["bf"] >= MIN_BF]
        if g.empty:
            continue
        g["pitcher"] = re.search(r"(\d+)", os.path.basename(path)).group(1)
        rows.append(g.reset_index())
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values(["pitcher", "date"])


def build(starts):
    """
    Baseline and form features, all from STRICTLY PRIOR starts.

    `shift(1)` on every rolling window is the whole ballgame. Without it
    tonight's start is inside its own baseline and inside its own form
    window, and the test finds a large, beautiful, entirely fictional
    effect.
    """
    f = starts.copy()
    g = f.groupby("pitcher", group_keys=False)

    # Baseline: his K rate over everything before tonight. Expanding
    # rather than trailing-12-months because it is a control, not a
    # projection -- it only has to absorb "how good is he" so that what
    # is left is deviation.
    f["prior_k"] = g["k"].apply(lambda s: s.shift(1).expanding().sum())
    f["prior_bf"] = g["bf"].apply(lambda s: s.shift(1).expanding().sum())
    f["n_prior"] = g.cumcount()
    f["base_rate"] = f["prior_k"] / f["prior_bf"]

    # Recent form: the last RECENT starts, ending before tonight.
    f["recent_k"] = g["k"].apply(
        lambda s: s.shift(1).rolling(RECENT, min_periods=RECENT).sum())
    f["recent_bf"] = g["bf"].apply(
        lambda s: s.shift(1).rolling(RECENT, min_periods=RECENT).sum())
    f["recent_rate"] = f["recent_k"] / f["recent_bf"]

    # Velocity, same shape. Baseline is his own expanding mean, so the
    # deviation is against himself and not against the league -- a 91 mph
    # sinkerballer is not "cold" for throwing 91.
    f["velo_base"] = g["velo"].apply(lambda s: s.shift(1).expanding().mean())
    f["velo_sd"] = g["velo"].apply(lambda s: s.shift(1).expanding().std())
    f["velo_recent"] = g["velo"].apply(
        lambda s: s.shift(1).rolling(RECENT, min_periods=RECENT).mean())

    f["rate"] = f["k"] / f["bf"]
    f["resid"] = f["rate"] - f["base_rate"]
    f["krate_z"] = (f["recent_rate"] - f["base_rate"]) / np.sqrt(
        f["base_rate"] * (1 - f["base_rate"]) / f["recent_bf"].clip(lower=1))
    f["velo_z"] = (f["velo_recent"] - f["velo_base"]) / f["velo_sd"]
    f["velo_drop"] = f["velo_recent"] - f["velo_base"]
    return f[f["n_prior"] >= MIN_PRIOR]


def permutation_null(f, col, label, n_perm=200):
    """
    The null as a DISTRIBUTION, not as one draw.

    This started as a single shuffle with a fixed seed, and that is not
    enough to read. On 11,148 starts a single permuted correlation has a
    standard error near 0.0095, so an ordinary draw lands anywhere inside
    roughly +/-0.019 -- and one did:

        recent K rate, z (shuffled)   r = +0.0183  p = 0.053

    which reads like the control firing and the real result being
    contaminated. It was neither. Averaged over 200 permutations the null
    sits where it should, at -0.0005 +/- 0.0095, and the observed +0.0340
    is z = +3.63 against it.

    A one-draw control is not conservative. It is just noisy, and it
    happened to be noisy in the direction that would have retired a real
    effect.
    """
    d = f.dropna(subset=[col, "resid"]).copy()
    rng = np.random.default_rng(20260915)

    def r_of(frame):
        x = frame.copy()
        for c in (col, "resid"):
            x[c] = x[c] - x.groupby("pitcher")[c].transform("mean")
        return float(stats.pearsonr(x[col], x["resid"])[0])

    obs = r_of(d)
    grp = d.groupby("pitcher")[col]
    null = np.empty(n_perm)
    for i in range(n_perm):
        c = d.copy()
        c[col] = grp.transform(lambda v: rng.permutation(v.values))
        null[i] = r_of(c)
    mu, sd = null.mean(), null.std(ddof=1)
    print(f"  {label:<28} observed r={obs:+.4f}   null {mu:+.4f} "
          f"+/- {sd:.4f}   z={(obs - mu) / sd:+.2f}   ({n_perm} permutations)")
    return (obs - mu) / sd


def report(f, col, label, unit=""):
    """
    Does this form feature predict the residual, WITHIN a pitcher?

    Both the feature and the residual are de-meaned inside each pitcher
    before anything is correlated. That is not a refinement, it is the
    fix for a bug the control caught:

        recent K rate, z            r = +0.0621   z = +4.6
        recent K rate, z (SHUFFLED) r = +0.0487   z = +2.5   <-- should be 0

    Shuffling the feature within a pitcher preserves every BETWEEN-pitcher
    difference -- a pitcher whose form scores run high stays high, and if
    such pitchers also run positive residuals the "null" finds an effect
    with the pairing already destroyed. The question was never "do
    high-form pitchers beat their baseline", which is a fact about which
    pitchers they are. It is "when HE is off his own norm, does he miss
    fewer bats than his own rate implies". De-meaning asks that.
    """
    d = f.dropna(subset=[col, "resid", "bf"]).copy()
    d = d[np.isfinite(d[col])]
    if len(d) < 200:
        print(f"  {label:28} n={len(d)} -- too few")
        return
    for c in (col, "resid"):
        d[c] = d[c] - d.groupby("pitcher")[c].transform("mean")
    w = d["bf"]
    # Weighted least squares: a 28-batter start is more evidence than a
    # 13-batter one, and the residual's own variance scales with 1/bf.
    b, a = np.polyfit(d[col], d["resid"], 1, w=np.sqrt(w))
    r = stats.pearsonr(d[col], d["resid"])
    # Effect size that means something: the residual gap between the
    # bottom and top decile of the form feature.
    lo, hi = d[col].quantile(.1), d[col].quantile(.9)
    g_lo = d[d[col] <= lo]; g_hi = d[d[col] >= hi]
    # Both residuals are already within-pitcher deviations, so a simple
    # batter-weighted mean of each decile is the effect.
    m_lo = float(np.average(g_lo["resid"], weights=g_lo["bf"]))
    m_hi = float(np.average(g_hi["resid"], weights=g_hi["bf"]))
    se = float(np.sqrt(d["resid"].var() * (1 / len(g_lo) + 1 / len(g_hi))))
    z = (m_hi - m_lo) / se if se else 0
    print(f"  {label:28} n={len(d):>5}  r={r.statistic:+.4f} "
          f"p={r.pvalue:.3f}")
    print(f"  {'':28}   bottom decile {m_lo:+.3%} vs top {m_hi:+.3%} "
          f"-> {m_hi - m_lo:+.2%} K rate, z={z:+.1f}")
    return m_hi - m_lo, z


def joint(f):
    """
    Two features that both clear a null are not two features.

    Recent K rate and velocity correlate at r = +0.19 -- an arm that is
    down a mile an hour is also, often, an arm that has been missing fewer
    bats. So the question a model actually has to answer is not "does
    recent K rate predict the residual" but "does it predict anything
    velocity has not already said". Fitted together, on the same
    within-pitcher de-meaned data:

        recent K rate   +0.00187 +/- 0.00092   z = +2.04
        velocity        +0.01099 +/- 0.00133   z = +8.23

    Velocity barely moves from its solo slope of +0.01151. Recent K rate
    falls from +0.00333 to +0.00187 -- it loses nearly half its effect to a
    feature that was already there, and what survives is marginal.

    This is why the marker was built on velocity and deliberately NOT on
    recent strikeout rate. That call was made when recent K rate looked
    like noise (r = +0.019, p = 0.21 on a quarter of the caches). It reads
    differently now -- z = +3.54 on its own -- and the call still holds,
    for a better reason than the one it was made for.
    """
    d = f.dropna(subset=["krate_z", "velo_z", "resid", "bf"]).copy()
    for c in ("krate_z", "velo_z", "resid"):
        d[c] = d[c] - d.groupby("pitcher")[c].transform("mean")
    print("\nDO THEY SAY DIFFERENT THINGS?")
    print(f"  the two features correlate at r = "
          f"{stats.pearsonr(d.krate_z, d.velo_z)[0]:+.4f}")
    X = np.column_stack([np.ones(len(d)), d.krate_z, d.velo_z])
    y = d["resid"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / (len(d) - 3)))
    print(f"  {'':<16} {'joint':>10} {'z':>7} {'alone':>10}   worth at 2 sd")
    for nm, col, i in (("recent K rate", "krate_z", 1),
                       ("velocity", "velo_z", 2)):
        solo = float(np.polyfit(d[col], y, 1)[0])
        print(f"  {nm:<16} {beta[i]:>+10.5f} {beta[i] / se[i]:>+7.2f} "
              f"{solo:>+10.5f}   {beta[i] * 2 * d[col].std() * 23:+.2f} K "
              f"over 23 batters")
    print("  -> velocity is the feature; recent K rate is mostly velocity")
    print("     showing through, and what is left of it is marginal.")


def main():
    starts = per_start()
    if starts.empty:
        print("No cached pitcher starts found.")
        return 1
    f = build(starts)
    print(f"{starts.pitcher.nunique()} pitchers, {len(starts)} starts, "
          f"{len(f)} usable after {MIN_PRIOR} starts of history")
    print(f"{starts['date'].min():%Y-%m-%d} to {starts['date'].max():%Y-%m-%d}")
    v = f["velo"].notna().sum()
    print(f"fastball velocity on {v}/{len(starts)} starts "
          f"({v / len(starts):.0%})\n")

    print("DOES RECENT FORM PREDICT THE STRIKEOUT RESIDUAL?")
    print("(residual = this start's K rate minus his own prior rate)\n")
    report(f, "krate_z", "recent K rate, z")
    print()
    report(f, "velo_z", "fastball velocity, z")
    print()
    report(f, "velo_drop", "fastball velocity, raw mph")

    # The control. Shuffle the form feature within each pitcher: the
    # baseline, the residual and the sample all stay exactly as they are,
    # and only the pairing is destroyed. Anything the real test finds that
    # this also finds is an artefact of the construction, not a signal.
    joint(f)
    print("\nCONTROL -- form feature shuffled within each pitcher")
    permutation_null(f, "krate_z", "recent K rate, z")
    permutation_null(f, "velo_z", "fastball velocity, z")

    # ---- the obvious alternative explanation --------------------------
    #
    # Fastball velocity climbs through a season -- April arms are slower
    # than July arms -- and strikeout rate has its own seasonal arc. If
    # both simply follow the calendar, "velocity is down" is a clock and
    # not a signal, and de-meaning within a pitcher does not remove it
    # because the arc is inside every pitcher's career.
    #
    # So: de-mean within (pitcher, season-month) instead. If the effect is
    # the calendar it dies here. If it survives, it is about the pitcher.
    print("\nCONFOUND CHECK -- is this just the seasonal velocity arc?")
    #
    # Fastball velocity climbs through a season -- April arms are slower
    # than July arms -- and strikeout rate has its own arc. If both follow
    # the calendar then "velocity is down" is a clock, not a signal, and
    # within-pitcher de-meaning does not remove it because the arc sits
    # inside every career.
    #
    # The WRONG fix, tried first: de-mean within (pitcher, month). With
    # about five starts a month and rolling windows that overlap the very
    # residuals being de-meaned, that manufactures mean reversion out of
    # nothing -- it returned r = -0.31, z = -14.2 for recent K rate, an
    # "effect" larger than anything real in this data and produced
    # entirely by the arithmetic.
    #
    # The right fix removes the LEAGUE's month effect, not the pitcher's:
    # subtract, from each variable, the all-pitcher mean for that calendar
    # month. The seasonal arc goes; the within-pitcher sample stays whole.
    m = f.copy()
    m["mon"] = m["date"].dt.month
    for c in ("velo_drop", "velo_z", "krate_z", "resid"):
        league = m.groupby("mon")[c].transform("mean")
        m[c] = m[c] - league
    report(m, "velo_drop", "velocity mph, season removed")
    print()
    report(m, "velo_z", "velocity z, season removed")
    print()
    report(m, "krate_z", "recent K rate, season removed")

    # ---- and does it move the OTHER half of the model? -----------------
    #
    # expected_k is batters faced times rate. A pitcher who has lost two
    # miles an hour may also be getting pulled earlier, which would show
    # up in the outs prop rather than the strikeout one -- and would be
    # worth more than the rate effect.
    print("\nDOES IT ALSO PREDICT HOW LONG HE LASTS?")
    b = f.dropna(subset=["velo_drop", "bf"]).copy()
    b = b[np.isfinite(b["velo_drop"])]
    b["bf_base"] = b.groupby("pitcher")["bf"].transform("mean")
    b["bf_resid"] = b["bf"] - b["bf_base"]
    for c in ("velo_drop", "bf_resid"):
        b[c] = b[c] - b.groupby("pitcher")[c].transform("mean")
    r = stats.pearsonr(b["velo_drop"], b["bf_resid"])
    lo, hi = b["velo_drop"].quantile(.1), b["velo_drop"].quantile(.9)
    d_lo = b[b.velo_drop <= lo]["bf_resid"].mean()
    d_hi = b[b.velo_drop >= hi]["bf_resid"].mean()
    print(f"  batters faced vs velocity    n={len(b)}  "
          f"r={r.statistic:+.4f} p={r.pvalue:.3f}")
    print(f"  {'':28}   bottom decile {d_lo:+.2f} batters vs "
          f"top {d_hi:+.2f} -> {d_hi - d_lo:+.2f}")

    # Does it survive the thing that matters -- predicting a whole start's
    # strikeout COUNT, not just the rate?
    print("\nIF IT WERE WIRED IN, WHAT WOULD IT BE WORTH?")
    d = f.dropna(subset=["velo_z", "resid"])
    d = d[np.isfinite(d["velo_z"])]
    if len(d) > 200:
        b, a = np.polyfit(d["velo_z"], d["resid"], 1, w=np.sqrt(d["bf"]))
        print(f"  slope {b:+.4f} K-rate per sd of velocity")
        print(f"  a pitcher 2 sd down would be {b * 2:+.3%} on his K rate,")
        print(f"  which over 23 batters is {b * 2 * 23:+.2f} strikeouts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
