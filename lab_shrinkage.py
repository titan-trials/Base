"""
Is the strikeout-rate prior too strong for pitchers far from league?

    python lab_shrinkage.py

THE CASE THAT PROMPTED IT
-------------------------
Nolan, reading the Pitchers tab on a starter the model liked and the
market did not:

    Brandon Young, line 3.5. Model 73% over, market 52%, edge +21 points.
    Recent form panel: 2, 7, 4, 3, 3, 4 strikeouts.

Three of six clear 3.5 -- 50%, almost exactly the market's 52%. His own
record says he is a 3-4 strikeout pitcher who occasionally runs into a
seven. The model says five.

Where the model's number comes from: `k_rate` 0.194 and 0.202 on his two
logged starts, against roughly 23 K in 139 batters faced over those six
outings -- a **16.5%** rate. The model has him three and a half points
high, and 3.5 points over 24 batters is most of a strikeout, which at a
line sitting on his own mean is a twenty-point swing in probability.

THE MECHANISM, AND WHY THE CALIBRATION TEST CANNOT SEE IT
---------------------------------------------------------
`K_PRIOR_BF = 250` batters faced. A pitcher's own rate is shrunk toward
the league's, so with ~575 batters of his own history the league still
supplies 30% of his number. For a pitcher AT league average that costs
nothing. For one 6 points below it, it hands back nearly two of those
points.

lab_k_spread.py already found the aggregate symptom -- the model moves
0.67 strikeouts for every 1 the market moves, high on soft arms and low
on aces -- and found the model sitting on the diagonal against outcomes.
Both are true at once: **a model can be conditionally unbiased in the
aggregate and still systematically wrong on the pitchers furthest from
the mean**, because the bins that sit on the diagonal are dominated by
the many pitchers near league average.

And lab_pitcher_form.py found that recent K RATE adds nothing over a
pitcher's own expanding history (r = +0.019, p = 0.21). That is not in
tension with this: there the baseline WAS his own unshrunk history, so
there was nothing left for form to add. Here the baseline is pulled
toward league, and the question is whether that pull is too hard.

WHAT THIS MEASURES
------------------
Directly, on 4,599 cached starts: sweep the prior strength, predict each
start from only PRIOR starts, and see which value of K_PRIOR_BF actually
minimises out-of-sample error. Actual batters faced is used so the
question is purely about the RATE, not about workload.


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
MIN_BF = 12
MIN_PRIOR_STARTS = 8       # enough history that his own rate means something
CURRENT = 250.0            # K_PRIOR_BF in features/pitcher_workload.py


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
        g["pitcher"] = re.search(r"(\d+)", os.path.basename(path)).group(1)
        rows.append(g.reset_index())
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values(["pitcher", "date"])


def build(s):
    """Prior K and BF for every start, from strictly earlier starts."""
    f = s.copy()
    g = f.groupby("pitcher", group_keys=False)
    f["prior_k"] = g["k"].apply(lambda x: x.shift(1).expanding().sum())
    f["prior_bf"] = g["bf"].apply(lambda x: x.shift(1).expanding().sum())
    f["n_prior"] = g.cumcount()
    f = f[f["n_prior"] >= MIN_PRIOR_STARTS].dropna(subset=["prior_bf"])
    f = f[f["prior_bf"] > 0]
    # The league rate as of that date, from every OTHER pitcher's prior
    # work. A fixed constant would leak the whole sample's average into
    # every early start.
    f = f.sort_values("date")
    f["lg_k"] = f["k"].cumsum().shift(1).fillna(0)
    f["lg_bf"] = f["bf"].cumsum().shift(1).fillna(0)
    f["league"] = (f["lg_k"] / f["lg_bf"].replace(0, np.nan)).ffill()
    f["own_rate"] = f["prior_k"] / f["prior_bf"]
    return f.dropna(subset=["league"])


def sweep(f, priors):
    """Out-of-sample error for each candidate prior strength."""
    out = []
    for k in priors:
        rate = (f["prior_k"] + k * f["league"]) / (f["prior_bf"] + k)
        pred = rate * f["bf"]
        err = f["k"] - pred
        # Mean absolute error in strikeouts per start is the unit that
        # means something at the table; RMSE is reported alongside
        # because a single blow-up start should not decide this.
        out.append({
            "prior": k,
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()),
        })
    return pd.DataFrame(out)


def main():
    s = per_start()
    if s.empty:
        print("No cached pitcher starts found.")
        return 1
    f = build(s)
    print(f"{f.pitcher.nunique()} pitchers, {len(f)} starts with "
          f"{MIN_PRIOR_STARTS}+ of history")
    print(f"league K rate at the end of the window: {f.league.iloc[-1]:.1%}\n")

    priors = [0, 25, 50, 75, 100, 150, 200, 250, 350, 500, 750]
    res = sweep(f, priors)
    best = res.loc[res["rmse"].idxmin()]
    cur = res[res["prior"] == CURRENT].iloc[0]

    print("PRIOR STRENGTH SWEEP  (predicting each start from prior starts only)")
    print(f"  {'K_PRIOR_BF':>10} {'MAE':>7} {'RMSE':>7} {'bias':>7}")
    for _, r in res.iterrows():
        mark = ""
        if r["prior"] == CURRENT:
            mark = "  <- in the model today"
        if r["prior"] == best["prior"]:
            mark += "  <- best"
        print(f"  {r['prior']:>10.0f} {r['mae']:>7.4f} {r['rmse']:>7.4f} "
              f"{r['bias']:>+7.4f}{mark}")
    print(f"\n  best {best['prior']:.0f} vs current {CURRENT:.0f}: "
          f"RMSE {best['rmse']:.4f} against {cur['rmse']:.4f} "
          f"({(cur['rmse'] - best['rmse']) / cur['rmse']:+.2%})")

    # ---- who pays for it ------------------------------------------
    #
    # The aggregate barely moves, which is exactly why this is invisible
    # in a calibration check: most pitchers sit near league and the
    # shrinkage costs them nothing. Split by how far a pitcher's own rate
    # is from the league and the picture changes.
    print("\nWHO PAYS FOR THE PRIOR  (error by distance from league rate)")
    f = f.copy()
    f["gap"] = f["own_rate"] - f["league"]
    f["shrunk"] = ((f["prior_k"] + CURRENT * f["league"])
                   / (f["prior_bf"] + CURRENT))
    f["unshrunk"] = f["own_rate"]
    for lo, hi, lab in [(-1, -0.05, "6+ pts BELOW league (soft)"),
                        (-0.05, -0.02, "2-5 below"),
                        (-0.02, 0.02, "within 2 pts"),
                        (0.02, 0.05, "2-5 above"),
                        (0.05, 1, "5+ pts ABOVE league (power)")]:
        g = f[(f["gap"] > lo) & (f["gap"] <= hi)]
        if len(g) < 60:
            continue
        e_s = (g["k"] - g["shrunk"] * g["bf"])
        e_u = (g["k"] - g["unshrunk"] * g["bf"])
        print(f"  {lab:30} n={len(g):>4}  shrunk bias {e_s.mean():+.3f} K  "
              f"unshrunk {e_u.mean():+.3f} K")

    # ---- the Brandon Young question -------------------------------
    #
    # How big is the error on a pitcher exactly like him: well below
    # league, plenty of history, and a line sitting on his own mean?
    print("\nA SOFT ARM WITH REAL HISTORY")
    soft = f[(f["gap"] <= -0.05) & (f["prior_bf"] >= 400)]
    if len(soft) >= 40:
        pred_s = soft["shrunk"] * soft["bf"]
        pred_u = soft["unshrunk"] * soft["bf"]
        print(f"  n={len(soft)} starts")
        print(f"  his own rate      {soft['unshrunk'].mean():.1%}")
        print(f"  what the model uses {soft['shrunk'].mean():.1%}  "
              f"(+{(soft['shrunk'] - soft['unshrunk']).mean() * 100:.1f} pts)")
        print(f"  projected K       {pred_s.mean():.2f} shrunk vs "
              f"{pred_u.mean():.2f} unshrunk")
        print(f"  actual K          {soft['k'].mean():.2f}")
        print(f"  over-projection   {(pred_s - soft['k']).mean():+.2f} K "
              f"per start (shrunk), "
              f"{(pred_u - soft['k']).mean():+.2f} (unshrunk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
