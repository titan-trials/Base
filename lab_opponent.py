"""
Does a pitcher (or hitter) genuinely have a thing against a given team?

RUN
    python lab_opponent.py           # all three sections
    python lab_opponent.py sample    # how much data is there at all
    python lab_opponent.py effect    # is there a real effect beyond noise
    python lab_opponent.py persist   # does it PERSIST -- the decisive one

WHY THIS EXISTS
---------------
Nolan asked for "how is Cease historically against BAL". The first answer
was a sample-size objection built on Corbin Burnes alone. He pushed back:
one player is not a measurement, and a thin sample for Burnes could be a
Burnes problem -- he went MIL -> BAL -> AZ in four years, which splits his
opponent history across three divisions. Correct objection.

So this measures it across every cached starter, and then asks the better
question, which is not about sample size at all.

THE THREE CLAIMS HIDING IN ONE NUMBER
-------------------------------------
"Burnes strikes out 37.6% of Giants" is three claims stacked:

  1. Burnes strikes out a lot of everyone          -- the model has this
  2. The Giants strike out a lot against everyone  -- the model has this
     too, because it walks the actual lineup and uses each hitter's own
     strikeout rate
  3. Burnes specifically does something to the Giants that neither of
     those explains

Only (3) is new information. So `effect` tests the RESIDUAL after both main
effects, and `persist` asks whether that residual is a stable property of
the pairing or something that merely happened.

WHAT IT FOUND (50 pitchers, 111,313 PA, 2022-2026)
--------------------------------------------------
  sample    median 3 starts against a given opponent; 62% of pairings are
            3 or fewer. Burnes was the MEDIAN, not an outlier -- but that
            turned out not to be the binding constraint.
  effect    real overdispersion, ~2.3 points of K rate, z = +5.4 against a
            calibrated bootstrap null. So there IS something there.
  persist   split-half r = +0.019 against a null of -0.001 +/- 0.030,
            z = +0.66. It does NOT persist. The overdispersion is drift
            and within-start clustering, not a trait of the pairing.

The conclusion is stronger than the sample-size one it replaces: even with
unlimited data there is nothing here to predict with, because what is there
does not repeat.
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
K_EVENTS = {"strikeout", "strikeout_double_play"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
MIN_PA = 25
RNG = np.random.default_rng(7)

# One shared column list for every reader in this file. A statcast pitcher
# cache is ~100 columns and 6 MB; reading only these cuts both by an order
# of magnitude, and all three sections need the same handful.
#   inning_topbot is how side is recovered -- the pitcher is on the
#   FIELDING side, so Top of the inning (away team batting) means HIS team
#   is the home team and his opponent is the away team. Getting this
#   backwards silently reports a pitcher's own team as his opponent.
USE = ["game_pk", "game_date", "game_type", "home_team", "away_team",
       "inning_topbot", "events", "player_name", "stand"]


def _odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


# ===================================================================== sample


def starts_for(path):
    """One row per (pitcher, game) with opponent, side, pitch count, PA, K."""
    try:
        raw = pd.read_csv(path, usecols=lambda c: c in USE, low_memory=False)
    except Exception:
        return None
    if raw.empty or "inning_topbot" not in raw:
        return None
    if "game_type" in raw:
        raw = raw[raw["game_type"] == "R"]
    if raw.empty:
        return None

    # The pitcher is on the FIELDING side. Top of the inning means the away
    # team is batting, so the pitcher's own team is the home team and his
    # opponent is the away team.
    is_home = raw["inning_topbot"].eq("Top")
    raw = raw.assign(
        opp=np.where(is_home, raw["away_team"], raw["home_team"]),
        home=is_home,
        k=raw["events"].isin(K_EVENTS),
        pa=raw["events"].notna() & raw["events"].ne(""),
    )
    g = raw.groupby("game_pk").agg(
        date=("game_date", "first"), opp=("opp", "first"),
        home=("home", "first"), pitches=("events", "size"),
        pa=("pa", "sum"), k=("k", "sum"))
    # Relief outings and bulk-guy cameos are not what the Pitchers tab is
    # about, and they would flatter the per-opponent counts.
    g = g[g["pa"] >= 12]
    if g.empty:
        return None
    g["pitcher"] = re.search(r"(\d+)", os.path.basename(path)).group(1)
    g["name"] = raw["player_name"].iloc[0] if "player_name" in raw else ""
    return g.reset_index()


# ===================================================================== effect


def cells():
    """One row per (pitcher, opponent, season): PA and K."""
    rows = []
    for path in sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv"))):
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in USE, low_memory=False)
        except Exception:
            continue
        if raw.empty or "inning_topbot" not in raw:
            continue
        if "game_type" in raw:
            raw = raw[raw["game_type"] == "R"]
        raw = raw[raw["events"].notna() & raw["events"].ne("")]
        if raw.empty:
            continue
        home = raw["inning_topbot"].eq("Top")
        raw = raw.assign(
            opp=np.where(home, raw["away_team"], raw["home_team"]),
            k=raw["events"].isin(K_EVENTS),
            year=pd.to_datetime(raw["game_date"], errors="coerce").dt.year)
        pid = re.search(r"(\d+)", os.path.basename(path)).group(1)
        g = raw.groupby(["opp", "year"]).agg(pa=("k", "size"), k=("k", "sum"))
        g["pitcher"] = pid
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def _fit(pa, k, pit_idx, opp_idx, n_pit, n_opp):
    """Two main effects, combined multiplicatively in odds."""
    lg = k.sum() / pa.sum()
    pit = np.bincount(pit_idx, k, n_pit) / np.maximum(
        np.bincount(pit_idx, pa, n_pit), 1)
    opp = np.bincount(opp_idx, k, n_opp) / np.maximum(
        np.bincount(opp_idx, pa, n_opp), 1)
    o = _odds(pit[pit_idx]) * _odds(opp[opp_idx]) / _odds(lg)
    return o / (1 + o)


def dispersion(pa, k, pit_idx, opp_idx, n_pit, n_opp):
    """Mean squared Pearson residual. 1.0 under pure binomial noise."""
    p = _fit(pa, k, pit_idx, opp_idx, n_pit, n_opp)
    var = pa * p * (1 - p)
    ok = var > 0
    return float((((k[ok] - pa[ok] * p[ok]) ** 2) / var[ok]).mean()), p


def test(df, label, draws=400):
    pit_idx, pits = pd.factorize(df["pitcher"])
    opp_idx, opps = pd.factorize(df["opp"])
    pa = df["pa"].to_numpy(float)
    k = df["k"].to_numpy(float)
    n_pit, n_opp = len(pits), len(opps)

    obs, p_hat = dispersion(pa, k, pit_idx, opp_idx, n_pit, n_opp)

    # Null: the fitted main effects are true, only the coin flips vary.
    # Refit inside each draw so the degrees of freedom spent on fitting are
    # charged to the null as well as to the observation.
    null = np.empty(draws)
    for i in range(draws):
        sim = RNG.binomial(pa.astype(int), p_hat).astype(float)
        null[i], _ = dispersion(pa, sim, pit_idx, opp_idx, n_pit, n_opp)

    mu, sd = null.mean(), null.std()
    z = (obs - mu) / sd
    pval = (null >= obs).mean()

    print(f"{label}")
    print(f"  cells {len(df):5}   trials {int(pa.sum()):>8,}")
    print(f"  observed dispersion   {obs:.4f}")
    print(f"  null (bootstrap)      {mu:.4f} +/- {sd:.4f}")
    print(f"  z                     {z:+.2f}        p = {pval:.3f}")
    if obs - mu > 2 * sd:
        med = np.median(pa)
        p = np.median(p_hat)
        sdk = np.sqrt(max(obs - mu, 0) * p * (1 - p) / med)
        print(f"  -> real effect sd     {sdk:.1%} K rate at the median "
              f"{med:.0f}-PA cell")
    else:
        print(f"  -> no effect beyond the two the model already has")
    print()
    return obs, mu, sd


# ===================================================================== persist


def pa_rows():
    """One row per plate appearance: pitcher, opponent, date, K."""
    out = []
    for path in sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv"))):
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in USE,
                              low_memory=False)
        except Exception:
            continue
        if raw.empty or "inning_topbot" not in raw:
            continue
        if "game_type" in raw:
            raw = raw[raw["game_type"] == "R"]
        raw = raw[raw["events"].notna() & raw["events"].ne("")]
        if raw.empty:
            continue
        home = raw["inning_topbot"].eq("Top")
        out.append(pd.DataFrame({
            "pitcher": re.search(r"(\d+)", os.path.basename(path)).group(1),
            "opp": np.where(home, raw["away_team"], raw["home_team"]),
            "date": pd.to_datetime(raw["game_date"], errors="coerce"),
            "k": raw["events"].isin(K_EVENTS).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True).dropna(subset=["date"])


def residuals(pa, key, col):
    """
    Per (subject, opponent, half): observed rate minus what the subject's
    own rate and the opponent's own rate predict.

    The main effects are fitted on EACH HALF SEPARATELY. Fitting them once
    over everything would leak the second half into the first and
    manufacture a correlation out of nothing.
    """
    frames = []
    for half, part in pa.groupby("half"):
        lg = part[col].mean()
        own = part.groupby(key)[col].mean()
        opp = part.groupby("opp")[col].mean()
        cell = part.groupby([key, "opp"]).agg(n=(col, "size"),
                                              y=(col, "sum")).reset_index()
        o = _odds(cell[key].map(own)) * _odds(cell["opp"].map(opp)) / _odds(lg)
        cell["exp"] = o / (1 + o)
        cell["resid"] = cell["y"] / cell["n"] - cell["exp"]
        cell["half"] = half
        frames.append(cell)
    return pd.concat(frames, ignore_index=True)


def split_half(pa, label, key="pitcher", col="k"):
    """
    Correlate a pairing's early-half residual against its late-half one.

    A real trait correlates across halves. The two things that produce
    real-but-useless overdispersion do not:

      drift       the pitcher was a different pitcher in 2022, and he
                  happened to face this opponent then and that one now,
                  so "opponent" stands in for "season"
      clustering  a bad start is 25 correlated plate appearances, not 25
                  independent ones, so one blow-up inflates a cell forever

    Each PAIRING is split at its own midpoint rather than at a global date,
    so a pairing that only ever happened early does not land entirely in
    one half.
    """
    pa = pa.sort_values("date").copy()
    size = pa.groupby([key, "opp"])[col].transform("size")
    pa["half"] = (pa.groupby([key, "opp"]).cumcount() >= size / 2) * 1

    res = residuals(pa, key, col)
    res = res[res["n"] >= MIN_PA]
    wide = res.pivot_table(index=[key, "opp"], columns="half",
                           values=["resid", "n"]).dropna()
    if len(wide) < 30:
        if label:
            print(f"{label}: only {len(wide)} pairings with {MIN_PA}+ PA "
                  f"in both halves -- cannot test")
        return None
    a, b = wide[("resid", 0)].to_numpy(), wide[("resid", 1)].to_numpy()
    # Weighted, because a 120-PA pairing says more than a 26-PA one.
    w = np.minimum(wide[("n", 0)], wide[("n", 1)]).to_numpy()
    r = stats.pearsonr(a, b)
    rw = np.cov(a, b, aweights=w)[0, 1] / np.sqrt(
        np.cov(a, a, aweights=w)[0, 1] * np.cov(b, b, aweights=w)[0, 1])
    if label:
        print(f"{label}")
        print(f"  pairings tested       {len(wide)}")
        print(f"  median PA per half    {np.median(w):.0f}")
        print(f"  r unweighted          {r.statistic:+.4f}  p={r.pvalue:.3f}")
    return r.statistic, rw


# ===================================================================== run
def section_sample():
    paths = sorted(glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv")))
    frames = [f for f in (starts_for(p) for p in paths) if f is not None]
    if not frames:
        print("no statcast_pitcher_*.csv in cache/"); return
    allg = pd.concat(frames, ignore_index=True)
    allg["date"] = pd.to_datetime(allg["date"], errors="coerce")
    print(f"{len(frames)} pitchers, {len(allg)} starts, "
          f"{allg.date.min():%Y-%m-%d} to {allg.date.max():%Y-%m-%d}\n")
    per = allg.groupby("pitcher").size()
    print(f"cached starts per pitcher: median {per.median():.0f}  "
          f"quartiles {per.quantile(.25):.0f}/{per.quantile(.75):.0f}  "
          f"range {per.min()}-{per.max()}\n")

    vs = allg.groupby(["pitcher", "opp"]).agg(starts=("k", "size"),
                                              pa=("pa", "sum"))
    print("STARTS AGAINST ONE OPPONENT")
    print(f"  pairings           {len(vs)}")
    for q in (.25, .5, .75, .9, .95):
        print(f"  {q:>4.0%}               {vs.starts.quantile(q):.0f} starts")
    print(f"  max                {vs.starts.max()}")
    print(f"  share <= 3 starts  {(vs.starts <= 3).mean():.0%}\n")
    print("BATTERS FACED against one opponent, and what that buys")
    for q in (.25, .5, .75, .9):
        n = vs.pa.quantile(q)
        print(f"  {q:>4.0%}   {n:5.0f} PA   -> K-rate standard error "
              f"+/-{np.sqrt(.23 * .77 / max(n, 1)):.1%}")
    print()
    best = vs.groupby("pitcher").starts.max()
    print(f"best single opponent, median across pitchers: {best.median():.0f}")
    print(f"pitchers whose best opponent is <= 4 starts:  {(best <= 4).mean():.0%}\n")
    ha = allg.groupby(["pitcher", "home"]).agg(starts=("k", "size"),
                                               pa=("pa", "sum"))
    both = ha.unstack("home").dropna()
    print("HOME / AWAY, for contrast -- this is what a usable split looks like")
    print(f"  median starts on the lighter side: "
          f"{both['starts'].min(axis=1).median():.0f}")
    print(f"  median PA on the lighter side:     "
          f"{both['pa'].min(axis=1).median():.0f}")


def section_effect():
    df = cells()
    print(f"{df.pitcher.nunique()} pitchers, {len(df)} "
          f"(pitcher, opponent, season) cells, {int(df.pa.sum()):,} PA\n")
    # Per season: a pitcher's true rate drifts year to year, so pooling
    # seasons would book aging as an opponent effect.
    test(df, "A. pitcher x opponent x season")
    pooled = df.groupby(["pitcher", "opp"], as_index=False)[["pa", "k"]].sum()
    test(pooled, "B. pitcher x opponent, career pooled")
    # Positive control. A test that cannot detect a known effect proves
    # nothing when it reports none.
    inj = pooled.copy()
    pi, pn = pd.factorize(inj["pitcher"])
    oi, on = pd.factorize(inj["opp"])
    p0 = _fit(inj["pa"].to_numpy(float), inj["k"].to_numpy(float),
              pi, oi, len(pn), len(on))
    inj["k"] = RNG.binomial(inj["pa"].to_numpy(int),
                            np.clip(p0 + RNG.normal(0, .03, len(inj)), .01, .99))
    test(inj, "C. POSITIVE CONTROL -- 3.0% sd effect injected")


def _persist(pa, key, col, label, draws=20):
    real = split_half(pa, label, key=key, col=col)
    if real is None:
        return
    nulls = []
    for _ in range(draws):
        fake = pa.copy()
        fake["opp"] = fake.groupby(key)["opp"].transform(
            lambda s: RNG.permutation(s.values))
        out = split_half(fake, None, key=key, col=col)
        if out:
            nulls.append(out[1])
    nulls = np.array(nulls)
    z = (real[1] - nulls.mean()) / nulls.std()
    print(f"  real r (PA-weighted)  {real[1]:+.4f}")
    print(f"  null                  {nulls.mean():+.4f} +/- {nulls.std():.4f}"
          f"   ({draws} draws)")
    print(f"  z                     {z:+.2f}")
    print(f"  -> {'PERSISTS' if z > 2 else 'does NOT persist'}\n")


def section_persist():
    pa = pa_rows()
    print(f"PITCHERS: {pa.pitcher.nunique()} pitchers, {len(pa):,} PA, "
          f"{pa.date.min():%Y-%m-%d} to {pa.date.max():%Y-%m-%d}\n")
    _persist(pa, "pitcher", "k", "A. pitcher vs opponent -- STRIKEOUT")

    # Hitters are the other half of the question, and a better case on
    # paper: a hitter faces a division rival's whole staff many times a
    # year, so samples are 2-3x a starter's. Underpowered here at 9
    # hitters -- the null band is wide -- so read it as "no evidence",
    # not as a matching null result.
    hp = hitter_rows()
    if hp is not None and len(hp):
        print(f"HITTERS: {hp.player.nunique()} hitters, {len(hp):,} PA "
              f"(low power -- wide null band)\n")
        _persist(hp, "player", "k", "B. hitter vs opposing team -- STRIKEOUT")
        _persist(hp, "player", "hit", "C. hitter vs opposing team -- HIT")


def hitter_rows():
    out = []
    for path in sorted(glob.glob(os.path.join(CACHE, "statcast_[A-Z]*.csv"))):
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in (
                "game_date", "game_type", "home_team", "away_team",
                "inning_topbot", "events"), low_memory=False)
        except Exception:
            continue
        if raw.empty or "inning_topbot" not in raw:
            continue
        if "game_type" in raw:
            raw = raw[raw["game_type"] == "R"]
        raw = raw[raw["events"].notna() & raw["events"].ne("")]
        if raw.empty:
            continue
        # The BATTER is on the hitting side: Top of the inning means the
        # away team bats, so his opponent is the home team.
        top = raw["inning_topbot"].eq("Top")
        out.append(pd.DataFrame({
            "player": " ".join(os.path.basename(path).split("_")[1:3]),
            "opp": np.where(top, raw["home_team"], raw["away_team"]),
            "date": pd.to_datetime(raw["game_date"], errors="coerce"),
            "k": raw["events"].isin(K_EVENTS).to_numpy(),
            "hit": raw["events"].isin(HIT_EVENTS).to_numpy()}))
    if not out:
        return None
    return pd.concat(out, ignore_index=True).dropna(subset=["date"])


SECTIONS = {"sample": section_sample, "effect": section_effect,
            "persist": section_persist}

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for name in want:
        if name not in SECTIONS:
            print(f"unknown section {name!r}; pick from {list(SECTIONS)}")
            sys.exit(2)
        print(f"\n{'=' * 72}\n{name.upper()}\n{'=' * 72}\n")
        SECTIONS[name]()
