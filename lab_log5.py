"""
When a good hitter meets a good pitcher, does the model land in the right
place?

    python lab_log5.py

NOLAN'S QUESTION, WHICH IS NOT THE ONE ALREADY ANSWERED
-------------------------------------------------------
    "We know when a good player plays, we expect him to do good. Take
     Shohei Ohtani. We know he's good. But when he also goes up against a
     good pitcher, he's still good even though that pitcher is also good.
     There should be something somewhere."

Three opponent questions have been confused with each other in this
project, and only two of them have been answered:

  1. Does the model know WHO is batting?          YES -- per_batter_k_probs
     walks tonight's real lineup, each hitter carrying his own rate.
  2. Is there extra signal in the OPPONENT'S       NO -- lab_opponent.py.
     IDENTITY, beyond its hitters?                 869 pairings, median 3
                                                   starts; does not persist.
  3. Does the model COMBINE hitter and pitcher     <- never tested
     correctly?

This file is 3, and it is the one Nolan is actually pointing at. The model
does not add the two rates or average them. It uses log5:

    odds(K) = [odds(batter) / odds(league)] x odds(pitcher)

That is a formula, and a formula is an assumption. It says the two effects
multiply in odds space, which has a specific and testable consequence at
the extremes: facing an ace should cost a contact hitter a LOT of
percentage points, because his odds get multiplied by a big number.

If real hitters resist more than that -- if elite bat-to-ball holds up
against elite stuff better than the multiplication says -- then log5
overstates the interaction, and it overstates it exactly in the corner of
the grid where the model's most confident strikeout calls live. Nobody has
looked.

THE ARTIFACT THAT WOULD FAKE THIS RESULT
----------------------------------------
Rates are estimated with noise. Bin hitters by an estimate and the extreme
bins are populated by hitters who got lucky as well as hitters who are
good, so their TRUE rates are less extreme than their measured ones. A
prediction built on the measured rate then overshoots at both ends, and the
residual grid shows a beautiful symmetric "log5 is too strong" pattern that
is nothing but regression to the mean.

So the null here is not a shuffle. It is a full simulation: take the
estimated rates as if they were truth, generate outcomes from log5 exactly,
re-estimate every rate from the simulated data with the same noise and the
same leave-one-game-out rule, and rebuild the grid. Any pattern that
appears in THAT world is the artifact. The observed grid only means
something where it differs from it.
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
USE = ["game_pk", "game_type", "events", "batter", "stand", "p_throws"]
MIN_BAT_PA = 40            # below this a hitter's rate is a rumour
MIN_PIT_PA = 200
# Measured, not chosen. Sweeping this and watching the contact/whiff row
# bias cross zero puts the right strength at ~35 plate appearances; at the
# 60 this file was first written with, contact hitters came out under-
# predicted by 0.8 points across the board and that main effect leaked into
# the corner cell as a fake z = -4.32 "log5 is broken" finding.
#
# Production does not hard-code this at all -- features/form_features.py
# calls estimate_prior_strength(), empirical Bayes on the data. This
# constant exists only because the lab re-derives rates from the statcast
# caches instead of reading the model's own hitter table.
BAT_PRIOR = 35.0           # plate appearances of league mixed into a hitter
PIT_PRIOR = 200.0          # batters faced of league mixed into a pitcher
N_SIM = 60
RNG = np.random.default_rng(20260914)


def odds(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return p / (1.0 - p)


def load():
    frames = []
    for path in sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv"))):
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in USE,
                              low_memory=False)
        except Exception:
            continue
        if raw.empty or "batter" not in raw.columns:
            continue
        if "game_type" in raw.columns:
            raw = raw[raw["game_type"] == "R"]
        raw = raw[raw["events"].notna() & raw["events"].ne("")]
        if raw.empty:
            continue
        raw = raw.copy()
        raw["pitcher"] = int(re.search(r"(\d+)",
                                       os.path.basename(path)).group(1))
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["is_k"] = df["events"].isin(K_EVENTS).astype(float)
    if {"stand", "p_throws"} <= set(df.columns):
        df["platoon"] = (df["stand"] != df["p_throws"]).astype(int)
    else:
        df["platoon"] = 0
    return df[["game_pk", "batter", "pitcher", "is_k", "platoon"]]


def rates(df, outcome="is_k"):
    """
    Leave-one-GAME-out rates for every batter and pitcher, shrunk to league.

    Leaving out the game rather than the single plate appearance matters:
    a hitter's four at-bats in one game share that night's pitcher, park
    and weather, so leaving out only the row being predicted would let the
    other three leak the answer back in.
    """
    lg = float(df[outcome].mean())

    b_tot = df.groupby("batter")[outcome].agg(["sum", "size"])
    b_gm = df.groupby(["batter", "game_pk"])[outcome].agg(["sum", "size"])
    p_tot = df.groupby("pitcher")[outcome].agg(["sum", "size"])
    p_gm = df.groupby(["pitcher", "game_pk"])[outcome].agg(["sum", "size"])

    bi = pd.MultiIndex.from_arrays([df["batter"], df["game_pk"]])
    pi = pd.MultiIndex.from_arrays([df["pitcher"], df["game_pk"]])

    bk = b_tot["sum"].reindex(df["batter"]).to_numpy() \
        - b_gm["sum"].reindex(bi).to_numpy()
    bn = b_tot["size"].reindex(df["batter"]).to_numpy() \
        - b_gm["size"].reindex(bi).to_numpy()
    pk = p_tot["sum"].reindex(df["pitcher"]).to_numpy() \
        - p_gm["sum"].reindex(pi).to_numpy()
    pn = p_tot["size"].reindex(df["pitcher"]).to_numpy() \
        - p_gm["size"].reindex(pi).to_numpy()

    b_rate = (bk + BAT_PRIOR * lg) / (bn + BAT_PRIOR)
    p_rate = (pk + PIT_PRIOR * lg) / (pn + PIT_PRIOR)
    return b_rate, p_rate, bn, pn, lg


def grid(df, b_rate, p_rate, lg, label, show=True):
    """
    Mean residual in each (batter tercile x pitcher tercile) cell.

    Positive means MORE strikeouts happened than log5 predicted.
    """
    pred = odds(b_rate) * odds(p_rate) / odds(lg)
    pred = pred / (1.0 + pred)
    res = df["is_k"].to_numpy() - pred

    # The platoon main effect is real and is not what is being tested, so
    # it comes out first. per_batter_k_probs applies it as a separate
    # factor in production for the same reason.
    d = pd.DataFrame({"res": res, "b": b_rate, "p": p_rate,
                      "batter": df["batter"].to_numpy(),
                      "platoon": df["platoon"].to_numpy()})
    d["res"] = d["res"] - d.groupby("platoon")["res"].transform("mean")

    # ---- the second de-meaning, and the reason for it -----------------
    #
    # The raw grid cannot separate two different mistakes:
    #
    #   log5 combines wrongly        a real interaction
    #   the batter prior is too strong   a main effect
    #
    # If BAT_PRIOR over-shrinks, every contact hitter is truly better than
    # his estimate, so he strikes out less than predicted against EVERY
    # pitcher -- and his whole row reads negative. The simulated null
    # cannot catch that, because the null takes the shrunk estimate as
    # truth by construction.
    #
    # Removing each BATTER's own mean residual deletes the main effect
    # exactly, whatever caused it. What survives is the only thing being
    # asked about: does this same hitter do relatively better against
    # power arms than against soft ones, compared with what log5 says?
    d["res_w"] = d["res"] - d.groupby("batter")["res"].transform("mean")

    d["bb"] = pd.qcut(d["b"], 3, labels=["contact", "mid", "whiff"])
    d["pb"] = pd.qcut(d["p"], 3, labels=["soft", "mid", "power"])
    cell = d.groupby(["bb", "pb"], observed=True)["res"].mean().unstack()
    cellw = d.groupby(["bb", "pb"], observed=True)["res_w"].mean().unstack()

    if show:
        print(f"\n  {label}: mean residual, actual minus log5")
        print(f"  {'':>10}" + "".join(f"{c:>12}" for c in cell.columns)
              + "   <- pitcher")
        for idx, row in cell.iterrows():
            print(f"  {str(idx):>10}"
                  + "".join(f"{v:>11.2%} " for v in row.to_numpy()))
        print("  ^ batter")
        print(f"\n  {label}: same grid with each BATTER's own mean removed")
        print(f"  {'':>10}" + "".join(f"{c:>12}" for c in cellw.columns)
              + "   <- pitcher")
        for idx, row in cellw.iterrows():
            print(f"  {str(idx):>10}"
                  + "".join(f"{v:>11.2%} " for v in row.to_numpy()))
        print("  ^ batter")

    # One number for the whole grid: does the residual tilt along the
    # diagonal? Standardise both axes and take the interaction slope.
    bz = (d["b"] - d["b"].mean()) / d["b"].std(ddof=1)
    pz = (d["p"] - d["p"].mean()) / d["p"].std(ddof=1)
    x = (bz * pz).to_numpy()
    # Two versions, for the same reason the grid has two. `slope` runs on
    # the raw residual and inherits any batter main effect; `slope_w` runs
    # on the within-batter residual and cannot. On the 50-cache sample the
    # difference did not matter because neither cleared its null. At 288k
    # plate appearances the raw one reached z = +1.97 while the within-
    # batter headline sat at -0.55, which is exactly the divergence a
    # contaminated statistic produces -- so both are reported and only the
    # second is quotable.
    slope = float(np.cov(x, d["res"].to_numpy(), ddof=1)[0, 1]
                  / np.var(x, ddof=1))
    slope_w = float(np.cov(x, d["res_w"].to_numpy(), ddof=1)[0, 1]
                    / np.var(x, ddof=1))
    corner = float(cell.loc["contact", "power"])
    # The headline: within one hitter, power arms against soft arms.
    cw = float(cellw.loc["contact", "power"] - cellw.loc["contact", "soft"])
    return slope, corner, cw, slope_w


def main():
    df = load()
    if df.empty:
        print("No cached plate appearances.")
        return 1

    b_rate, p_rate, bn, pn, lg = rates(df)
    keep = (bn >= MIN_BAT_PA) & (pn >= MIN_PIT_PA)
    df = df[keep].reset_index(drop=True)
    b_rate, p_rate = b_rate[keep], p_rate[keep]

    print(f"{len(df):,} plate appearances, {df.batter.nunique():,} batters, "
          f"{df.pitcher.nunique()} pitchers")
    print(f"league K rate {lg:.2%}")
    print(f"batter rates {b_rate.min():.1%} to {b_rate.max():.1%}, "
          f"pitcher rates {p_rate.min():.1%} to {p_rate.max():.1%}")

    print("\nOBSERVED")
    slope, corner, cw, slope_w = grid(df, b_rate, p_rate, lg, "observed")
    print(f"\n  interaction slope {slope:+.5f}  "
          f"(negative = log5 overstates the clash of extremes)")
    print(f"  same slope, within batter {slope_w:+.5f}")
    print(f"  contact hitter vs power pitcher: {corner:+.2%}")
    print(f"  within-batter, power minus soft (contact hitters): {cw:+.2%}")

    # ---- the null ----------------------------------------------------
    print(f"\nNULL  ({N_SIM} simulated worlds where log5 is exactly true)")
    print("  Rates taken as truth, outcomes generated from log5, then every")
    print("  rate re-estimated from the simulated data with the same noise.")
    truth = odds(b_rate) * odds(p_rate) / odds(lg)
    truth = truth / (1.0 + truth)
    sl, co, cws = np.empty(N_SIM), np.empty(N_SIM), np.empty(N_SIM)
    slw = np.empty(N_SIM)
    sim = df.copy()
    for i in range(N_SIM):
        sim["is_k"] = RNG.binomial(1, truth).astype(float)
        b2, p2, _, _, lg2 = rates(sim)
        sl[i], co[i], cws[i], slw[i] = grid(sim, b2, p2, lg2, "", show=False)

    zs = (slope - sl.mean()) / sl.std(ddof=1)
    zc = (corner - co.mean()) / co.std(ddof=1)
    print(f"\n  interaction slope  null {sl.mean():+.5f} +/- "
          f"{sl.std(ddof=1):.5f}   observed {slope:+.5f}   z = {zs:+.2f}")
    print(f"  contact vs power   null {co.mean():+.2%} +/- "
          f"{co.std(ddof=1) * 100:.2f} pts   observed {corner:+.2%}   "
          f"z = {zc:+.2f}")

    zsw = (slope_w - slw.mean()) / slw.std(ddof=1)
    print(f"  slope within-batter null {slw.mean():+.5f} +/- "
          f"{slw.std(ddof=1):.5f}   observed {slope_w:+.5f}   z = {zsw:+.2f}")
    zw = (cw - cws.mean()) / cws.std(ddof=1)
    print(f"  within-batter      null {cws.mean():+.2%} +/- "
          f"{cws.std(ddof=1) * 100:.2f} pts   observed {cw:+.2%}   "
          f"z = {zw:+.2f}")
    print("\n  The within-batter line is the one that matters. The two above")
    print("  it cannot tell a real interaction from an over-strong batter")
    print("  prior; this one has every batter main effect removed.")

    print()
    if abs(zw) < 2:
        print("  -> once each hitter is compared with himself, log5 lands")
        print("     where it should. The negative corner in the raw grid is")
        print("     a batter main effect -- the prior, not the formula.")
    elif abs(zs) < 2 and abs(zc) < 2:
        print("  -> log5 lands where it should. The way the model combines")
        print("     a hitter and a pitcher is not where the error is, and")
        print("     the regression-to-the-mean artifact is fully accounted")
        print("     for by the null -- which is most of what the raw grid")
        print("     shows.")
    else:
        print("  -> log5 does NOT land where it should. The residual in the")
        print("     flagged cell survives the artifact, which means the")
        print("     combination rule itself is carrying a bias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
