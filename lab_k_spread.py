"""
Is the strikeout model compressed, and if so where does it enter?

RUN
    python lab_k_spread.py            # all three sections
    python lab_k_spread.py market     # model vs market, every captured night
    python lab_k_spread.py spread     # how wide SHOULD the projections be
    python lab_k_spread.py calib      # mis-calibrated, or under-informed?

THE QUESTION THIS ANSWERS
-------------------------
Nolan noticed the Pitchers tab offering an over 5.5 column on a pitcher the
market had priced at 8.5. The columns were the symptom; this is the
investigation into the cause.

The three sections have to be read in order, because each one overturns the
obvious reading of the one before it:

  market   the model regresses on the market at slope 0.67 -- two thirds of
           a strikeout for every full strikeout the market moves. Looks
           like the model is broken.

  spread   the model's projections carry 45% of the true between-pitcher
           spread in batters faced. Looks like confirmation.

  calib    but binned against what actually happened, the model sits on the
           diagonal in 5 of 5 bins for pitchers it has 10+ starts on. It is
           not mis-calibrated. It is correctly shrunk, and a correctly
           shrunk estimator IS narrower than the truth:

               sd(projection) = r x sd(true value)

           So the width gap measures how much the model KNOWS, not how
           wrong it is, and stretching the output would break the one thing
           that currently holds.

WHAT IS ACTUALLY BROKEN
-----------------------
Two things, both visible in the calib section:

  * pitchers with no history: projected 22.6 batters faced, actual 10.4.
    A -12.2 batter error on 20 starts. This is a real bug, not shrinkage.
  * a uniform -0.4 strikeout bias that survives excluding them.

Neither is fixed by widening anything.
"""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy import stats

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _log():
    log = pd.read_csv(os.path.join(CACHE, "pitcher_row_log.csv"))
    log = log.dropna(subset=["expected_k", "strikeouts", "expected_bf",
                             "batters_faced", "k_rate"])
    # A start that ended early for a reason the model cannot see -- an
    # ejection, rain, an injury -- is not evidence about the projection.
    return log[log["batters_faced"] >= 3].copy()


def _market():
    """One row per pitcher-night: the market's central strikeout estimate."""
    parts = []
    for f in sorted(glob.glob(os.path.join(CACHE, "market_compare_*.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if {"pitcher", "line", "prob_over"} <= set(d.columns):
            d["date"] = os.path.basename(f)[15:25]
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    mc = pd.concat(parts, ignore_index=True)
    mc = mc[mc.get("market", "pitcher_strikeouts") == "pitcher_strikeouts"]
    # A book's line, plus how far its over price sits from even, scaled by
    # the spread of a real strikeout distribution (sd ~ 2.3 at these means).
    mc["mk"] = mc["line"] + (mc["prob_over"] - 0.5) * 3.0
    # The line closest to a coin flip is the one the book is surest about.
    mc["d"] = (mc["prob_over"] - 0.5).abs()
    return mc.sort_values("d").groupby(["date", "pitcher"],
                                       as_index=False).first()


# ===================================================================== 1
def fit(x, y, label):
    """OLS with the slope's confidence interval, which is the whole point.

    Correlation says whether the RANKING is right. Slope says whether the
    SPREAD is right. A model can rank perfectly and still quote everyone a
    number half as far from average as it should be -- invisible to
    correlation, to Brier skill, and to AUC.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 10:
        print(f"  {label:34} n={len(x)} -- too few")
        return None
    r = stats.linregress(x, y)
    lo, hi = r.slope - 1.96 * r.stderr, r.slope + 1.96 * r.stderr
    flag = "  <- too SPREAD" if hi < 1 else ("  <- too BUNCHED" if lo > 1 else "")
    print(f"  {label:34} n={len(x):4}  slope {r.slope:5.2f} "
          f"[{lo:4.2f}, {hi:4.2f}]  r={r.rvalue:.2f}{flag}")
    return r


def section_market():
    one, log = _market(), _log()
    if one.empty:
        print("no market_compare files"); return
    print(f"1. MODEL vs MARKET   {one.date.nunique()} nights, "
          f"{len(one)} pitcher-nights\n")
    fit(one["mk"], one["expected_k"], "model_k on market_k")
    print(f"  {'model sd / market sd':34} "
          f"{one.expected_k.std() / one.mk.std():.0%}\n")
    print("   by night:")
    for date, g in one.groupby("date"):
        if len(g) >= 10:
            r = stats.linregress(g["mk"], g["expected_k"])
            print(f"     {date}  n={len(g):3}  slope {r.slope:5.2f}  "
                  f"r={r.rvalue:.2f}")
    print()
    print("2. WHO IS RIGHT?  the market is not truth -- outcomes are\n")
    fit(log["expected_k"], log["strikeouts"], "actual_k on model_k")
    j = log.merge(one[["date", "pitcher", "mk"]], left_on=["game_date", "pitcher"],
                  right_on=["date", "pitcher"], how="inner")
    if len(j) >= 10:
        print(f"   on the {len(j)} starts where a closing line exists:")
        fit(j["expected_k"], j["strikeouts"], "actual_k on model_k")
        fit(j["mk"], j["strikeouts"], "actual_k on market_k")
    print()
    print("3. WHICH FACTOR?   expected_k = expected_bf x k_rate\n")
    log["actual_rate"] = log["strikeouts"] / log["batters_faced"]
    fit(log["expected_bf"], log["batters_faced"], "actual_bf on expected_bf")
    fit(log["k_rate"], log["actual_rate"], "actual_rate on model k_rate")
    print()
    print("4. BY HOW MUCH HISTORY THE PITCHER HAS\n")
    for lo, hi, lab in [(0, 0, "0 starts seen"), (1, 9, "1-9"),
                        (10, 19, "10-19"), (20, 99, "20+")]:
        g = log[(log.starts_seen >= lo) & (log.starts_seen <= hi)]
        if len(g) < 8:
            continue
        print(f"  {lab:14} n={len(g):3}  "
              f"bf {g.expected_bf.mean():5.2f} -> {g.batters_faced.mean():5.2f} "
              f"({g.batters_faced.mean() - g.expected_bf.mean():+6.2f})   "
              f"k {g.expected_k.mean():4.2f} -> {g.strikeouts.mean():4.2f} "
              f"({g.strikeouts.mean() - g.expected_k.mean():+5.2f})")


# ===================================================================== 2
def between_within(frame, value, group="pitcher_id"):
    """
    One-way random effects: split observed spread into between-pitcher and
    within-pitcher (start to start).

        var_between = var(pitcher means) - var_within / starts_per_pitcher

    A projection is only ever trying to hit the BETWEEN term. A model that
    matched the TOTAL spread would be badly overconfident.

    Pitchers with a single start are excluded -- they carry no information
    about within-pitcher spread, and including them would load all of their
    variance onto the between term.
    """
    counts = frame.groupby(group)[value].size()
    f = frame[frame[group].isin(counts[counts >= 2].index)]
    if f.empty:
        return np.nan, np.nan, 0
    g = f.groupby(group)[value]
    within = float(((g.transform("mean") - f[value]) ** 2).sum()
                   / (len(f) - g.ngroups))
    n = g.size()
    harmonic = len(n) / (1 / n).sum()
    return (np.sqrt(max(g.mean().var(ddof=1) - within / harmonic, 0)),
            np.sqrt(within), g.ngroups)


def section_spread():
    log = _log()
    log["actual_rate"] = log["strikeouts"] / log["batters_faced"]
    for label, f in [("ALL starts", log),
                     ("10+ starts of history", log[log.starts_seen >= 10])]:
        print(f"=== {label}   {len(f)} starts, {f.pitcher_id.nunique()} "
              f"pitchers ===\n")
        for value, col, name in [("batters_faced", "expected_bf", "BATTERS FACED"),
                                 ("strikeouts", "expected_k", "STRIKEOUTS"),
                                 ("actual_rate", "k_rate", "STRIKEOUT RATE")]:
            sb, sw, ng = between_within(f, value)
            sd = f[col].std()
            print(f"  {name}")
            print(f"    actual, total spread        sd {f[value].std():6.3f}")
            print(f"    ... within a pitcher        sd {sw:6.3f}")
            print(f"    ... BETWEEN pitchers ({ng:3})   sd {sb:6.3f}"
                  f"   <- what a projection aims at")
            print(f"    model projections           sd {sd:6.3f}"
                  f"   ({sd / sb:.0%})\n" if sb > 0 else "\n")
        # Independent route for the rate: at ~22 batters faced a rate has a
        # known binomial floor, so the true spread is what is left after it.
        # Two routes agreeing is the only reason to believe either.
        p, n_bf = f["actual_rate"].mean(), f["batters_faced"].mean()
        floor = np.sqrt(p * (1 - p) / n_bf)
        true = np.sqrt(max(f["actual_rate"].std() ** 2 - floor ** 2, 0))
        print(f"  RATE, second route (binomial floor)")
        print(f"    per-start spread {f['actual_rate'].std():.3f}  minus "
              f"binomial floor {floor:.3f}  -> true {true:.3f}")
        print(f"    model {f['k_rate'].std():.3f}  "
              f"({f['k_rate'].std() / true:.0%} as wide)\n")


# ===================================================================== 3
def bins(frame, pred, actual, label, n=5):
    f = frame.dropna(subset=[pred, actual]).copy()
    f["bin"] = pd.qcut(f[pred], n, duplicates="drop", labels=False)
    print(f"  {label}")
    print(f"    {'bin':>3} {'n':>4} {'said':>7} {'happened':>9} {'gap':>7} {'+/-':>6}")
    off = 0
    for b, g in f.groupby("bin"):
        said, hap = g[pred].mean(), g[actual].mean()
        se = g[actual].std() / np.sqrt(len(g))
        bad = se and abs(hap - said) > 2 * se
        off += bool(bad)
        print(f"    {int(b) + 1:>3} {len(g):>4} {said:>7.2f} {hap:>9.2f} "
              f"{hap - said:>+7.2f} {se:>6.2f}{'  *' if bad else ''}")
    print(f"    bins off by more than 2 SE: {off}\n")


def section_calib():
    log = _log()
    print("A. ALL STARTS\n")
    bins(log, "expected_k", "strikeouts", "strikeouts")
    bins(log, "expected_bf", "batters_faced", "batters faced")
    print("B. EXCLUDING PITCHERS WITH UNDER 10 STARTS OF HISTORY\n")
    thick = log[log.starts_seen >= 10]
    bins(thick, "expected_k", "strikeouts", "strikeouts")
    bins(thick, "expected_bf", "batters_faced", "batters faced")

    one = _market()
    if one.empty:
        return
    j = log.merge(one[["date", "pitcher", "mk"]], left_on=["game_date", "pitcher"],
                  right_on=["date", "pitcher"], how="inner")
    print(f"C. MODEL AND MARKET, HEAD TO HEAD ON {len(j)} STARTS\n")
    bins(j, "expected_k", "strikeouts", "model", n=4)
    bins(j, "mk", "strikeouts", "market", n=4)
    print("  how much each knows:")
    for col, name in (("expected_k", "model"), ("mk", "market")):
        r = stats.pearsonr(j[col], j["strikeouts"]).statistic
        rmse = np.sqrt(((j["strikeouts"] - j[col]) ** 2).mean())
        print(f"    {name:8} r={r:.3f}  rmse={rmse:.3f}  "
              f"sd of its projections {j[col].std():.2f}")
    print()
    print("  OPTIMAL WIDTH:  sd(projection) should equal r x sd(true value).")
    print("  A shrunk estimator is SUPPOSED to be narrower than reality --")
    print("  the ratio measures information, not error.")
    sb, sw, _ = between_within(log[log.starts_seen >= 10], "strikeouts")
    rel = sb ** 2 / (sb ** 2 + sw ** 2)
    for col, name in (("expected_k", "model"), ("mk", "market")):
        r = stats.pearsonr(j[col], j["strikeouts"]).statistic
        r_true = min(r / np.sqrt(rel), 0.999)
        print(f"    {name:8} r with the true mean ~{r_true:.2f}  -> optimal sd "
              f"{r_true * sb:.2f}   actual {j[col].std():.2f}  "
              f"({j[col].std() / (r_true * sb):.0%})")


SECTIONS = {"market": section_market, "spread": section_spread,
            "calib": section_calib}

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for i, name in enumerate(want):
        if name not in SECTIONS:
            print(f"unknown section {name!r}; pick from {list(SECTIONS)}")
            sys.exit(2)
        print(f"\n{'=' * 72}\n{name.upper()}\n{'=' * 72}\n")
        SECTIONS[name]()
