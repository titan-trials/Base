"""
Real opposing-pitcher rates -- pulled per slate, not in bulk.

WHY THIS WAS THE BIGGEST GAP
----------------------------
Until now a pitcher only existed in this project's data when he happened
to face one of the pooled hitters. Statcast was pulled BATTER-side, so
Shane Baz might appear 20 times across the whole dataset despite having
faced 600 batters that season. The shrinkage estimator did the honest
thing with that -- it pinned him near league average -- which meant the
opposing-pitcher term was contributing almost nothing.

THE APPROACH (and why it beats a bulk pull)
-------------------------------------------
The obvious fix is to pull every pitcher in the league. That's hundreds of
slow requests for data that mostly never gets used.

Only about 30 pitchers start on a given day, and the schedule feed names
them in advance. So: pull those 30, cache them, move on. The library grows
organically -- after a couple of weeks of running slates you have most of
the league, without ever having done a blanket pull.

The payoff per call is large. `statcast_pitcher` returns EVERY batter the
pitcher faced, not just the ones in this project's hitter pool. One call
takes a starter from ~20 observations to ~600.

TWO-LEVEL SHRINKAGE
-------------------
Handedness splits get shrunk twice, and the order matters:

    vs-lefties rate  ->  toward his own overall rate  ->  toward league

A pitcher with 40 plate appearances against left-handers has almost no
information in that split alone, but he isn't league-average either -- he's
mostly himself. Shrinking straight to league would throw away everything
his overall performance says. Shrinking to his own mean first, and letting
that mean shrink to league, keeps both facts in proportion to how much
evidence supports each.

This matters here specifically because platoon splits are one of the few
genuinely large, genuinely stable effects in baseball, and a left-handed
starter really does change a left-handed hitter's night.

BULLPEN EXPOSURE
----------------
A starter does NOT face a hitter all four times. Measured on this data:

    starter share of plate appearances  52.8%
    bullpen share                       47.2%

So applying a starter's rate to the whole game charges him for ~47% of
plate appearances he never throws. The league-average cost of that is
small, because relievers are only slightly tougher overall (HR -4.2%,
hits -2.5%, walks +1.8%, strikeouts +1.7% versus starters).

But the average is the wrong way to look at it. The error scales with how
far the STARTER deviates from league, and it is large at the extremes:

    starter HR/PA   current P(HR)   exposure-weighted   shift
        0.030           0.123             0.171        +0.049
        0.058           0.227             0.225        -0.002
        0.090           0.333             0.283        -0.050

Facing an ace, the old version understated a hitter's home-run chance by
five points; facing a homer-prone starter it overstated by the same. Near
league average it changes nothing. `blend_with_bullpen` fixes this by
weighting the starter's rate by his actual share of exposure and filling
the rest with league reliever rates.

What is still NOT modelled: WHICH bullpen. A team with an elite relief
corps and one with a poor one both get the league reliever rate. That
needs team-level reliever data and is the next refinement, not this one.
"""
import os
import time

import numpy as np
import pandas as pd

from data.cache import cache_path, load_cached, save_cache
from data.game_filter import regular_season_only
from data.refresh import checked_today, mark_checked
from features.rate_features import estimate_prior_strength, shrink

DEDUPE_KEYS = ["game_pk", "at_bat_number", "pitch_number"]
REQUEST_DELAY_SEC = 0.3

# Outcomes a pitcher is charged with allowing, mirroring RATE_TARGETS on
# the batter side so the log5 combination has matching units.
PITCHER_TARGETS = ("is_hr", "is_hit", "is_walk", "is_k")

HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "hit_by_pitch"}

# Prior strength for the handedness split, in plate appearances, shrinking
# toward the pitcher's OWN overall rate. Smaller than the league-level
# priors because the thing being estimated (a deviation from his own mean)
# is a smaller quantity than the mean itself.
SPLIT_PRIOR_PA = 200.0

# Pitchers get their own last-checked log, same mechanism as hitters.
PITCHER_LOG_KEY = "refresh_log_pitchers"


def canonical_key(pitcher_id: int) -> str:
    return f"statcast_pitcher_{int(pitcher_id)}"


def load_pitcher_cache(pitcher_id: int) -> pd.DataFrame:
    cached = load_cached(canonical_key(pitcher_id))
    if cached is None or cached.empty:
        return pd.DataFrame()
    cached["game_date"] = pd.to_datetime(cached["game_date"], errors="coerce")
    return cached.dropna(subset=["game_date"]).sort_values("game_date")


def refresh_pitcher(pitcher_id: int, start_date: str, end_date: str,
                    max_age_days: int = 0, verbose: bool = True) -> pd.DataFrame:
    """
    Top up one pitcher's Statcast history, pulling only what's missing.
    Same incremental pattern as data/refresh.py for hitters, including the
    one-day overlap so same-day revisions land.
    """
    from pybaseball import statcast_pitcher

    cached = load_pitcher_cache(pitcher_id)
    today = pd.Timestamp.today().normalize()
    # Cap at today. `end_date` is the SLATE date, which is in the future --
    # comparing cached data against it meant the staleness test could never
    # pass and all ~28 starters were re-pulled on every run. Same bug that
    # was fixed on the hitter side in data/refresh.py; it survived here.
    effective_end = min(pd.Timestamp(end_date), today)

    if not cached.empty and checked_today(pitcher_id, PITCHER_LOG_KEY,
                                          as_of=today, max_age_days=max_age_days):
        return cached

    if cached.empty:
        pull_from = start_date
    else:
        pull_from = cached["game_date"].max().strftime("%Y-%m-%d")

    try:
        fresh = statcast_pitcher(pull_from, effective_end.strftime("%Y-%m-%d"),
                                 int(pitcher_id))
    except Exception as e:
        if verbose:
            print(f"      pitcher {pitcher_id}: {type(e).__name__} -- keeping cache")
        return cached

    time.sleep(REQUEST_DELAY_SEC)
    # A successful call with nothing new still counts as "checked".
    mark_checked(pitcher_id, PITCHER_LOG_KEY, as_of=today)
    if fresh is None or fresh.empty:
        return cached

    fresh["game_date"] = pd.to_datetime(fresh["game_date"], errors="coerce")
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    combined = combined.dropna(subset=["game_date"])

    keys = [k for k in DEDUPE_KEYS if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)

    save_cache(canonical_key(pitcher_id), combined)
    return combined


def pitcher_pa_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance faced, with outcome flags and the
    batter's handedness."""
    if raw.empty or "events" not in raw.columns:
        return pd.DataFrame()

    # Spring training and postseason are a different sport for a pitcher's
    # rates -- March is arm-building against minor leaguers, October is the
    # best hitters alive. Measured on the pitcher caches, non-regular
    # plate appearances strike out 7% more often than regular ones.
    raw = regular_season_only(raw)
    if raw.empty:
        return pd.DataFrame()

    pa = raw[raw["events"].notna()].copy()
    if pa.empty:
        return pd.DataFrame()

    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_walk"] = pa["events"].isin(WALK_EVENTS).astype(int)
    pa["is_k"] = (pa["events"] == "strikeout").astype(int)
    pa["stand"] = pa.get("stand", pd.Series("R", index=pa.index)).fillna("R")
    return pa


def load_starter_history(pitcher_ids, verbose: bool = True) -> pd.DataFrame:
    """
    Every cached plate appearance for these pitchers, concatenated.

    Reads the caches ONLY -- no refresh, no network. build_pitcher_rates
    has already brought them up to date earlier in the same run, and
    refreshing again here would double the pulls for nothing.

    build_pitcher_rates builds exactly this per-pitcher table internally
    and then throws it away, keeping only the aggregate rates. The
    workload model needs the rows themselves, because how many batters a
    start lasted is not recoverable from a rate.
    """
    frames = []
    for pitcher_id in [int(p) for p in pd.Series(list(pitcher_ids)).dropna().unique()]:
        raw = load_pitcher_cache(pitcher_id)
        if raw.empty:
            continue
        table = pitcher_pa_table(raw)
        if not table.empty:
            frames.append(table)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  Starter history: {len(out):,} plate appearances from "
              f"{out['pitcher'].nunique()} pitchers (cache only).")
    return out


def build_pitcher_rates(pitcher_ids, start_date: str, end_date: str,
                        league_rates: dict, names=None,
                        max_age_days: int = 0, verbose: bool = True) -> pd.DataFrame:
    """
    One row per pitcher with shrunk rates allowed, overall and by batter
    handedness.

    Columns per target:
        pit_{target}_allowed      overall, shrunk toward league
        pit_{target}_vs_L         versus left-handed batters
        pit_{target}_vs_R         versus right-handed batters
        pit_pa_seen               how many plate appearances back the estimate

    `pit_pa_seen` is exported deliberately -- it's the honest measure of
    how much any of these numbers should be trusted, and the dashboard can
    show it rather than presenting a 40-PA estimate identically to a
    600-PA one.
    """
    names = names or {}
    ids = [int(p) for p in pd.Series(list(pitcher_ids)).dropna().unique()]
    if not ids:
        return pd.DataFrame()

    # First pass: pull, and collect per-pitcher counts so the league-level
    # prior strength can be estimated from this population.
    tables = {}
    for i, pitcher_id in enumerate(ids, start=1):
        label = names.get(pitcher_id, str(pitcher_id))
        cached = load_pitcher_cache(pitcher_id)
        if verbose and cached.empty:
            print(f"    [{i}/{len(ids)}] pulling {label} (first time -- slow)")
        raw = refresh_pitcher(pitcher_id, start_date, end_date,
                              max_age_days=max_age_days, verbose=verbose)
        table = pitcher_pa_table(raw)
        if not table.empty:
            tables[pitcher_id] = table

    if not tables:
        if verbose:
            print("    No pitcher data available.")
        return pd.DataFrame()

    totals = {
        target: pd.DataFrame([
            {"pitcher": pid, "sum": t[target].sum(), "size": len(t)}
            for pid, t in tables.items()
        ]).set_index("pitcher")
        for target in PITCHER_TARGETS
    }

    priors = {}
    for target in PITCHER_TARGETS:
        block = totals[target]
        priors[target] = estimate_prior_strength(block["sum"], block["size"],
                                                 min_trials=50)

    if verbose:
        detail = "  ".join(f"{t}: k={priors[t]:.0f} PA" for t in PITCHER_TARGETS)
        print(f"    Pitcher shrinkage strengths -- {detail}")

    rows = []
    for pitcher_id, table in tables.items():
        row = {"pitcher": pitcher_id, "pit_pa_seen": len(table)}
        for target in PITCHER_TARGETS:
            league = league_rates.get(target, float(table[target].mean()))
            overall = float(shrink(table[target].sum(), len(table),
                                   league, priors[target]))
            row[f"pit_{target}_allowed"] = overall

            for hand in ("L", "R"):
                side = table[table["stand"] == hand]
                # Second level: shrink the split toward HIS overall rate,
                # not toward league. See the module docstring.
                row[f"pit_{target}_vs_{hand}"] = float(shrink(
                    side[target].sum(), len(side), overall, SPLIT_PRIOR_PA
                )) if len(side) else overall
                row[f"pit_pa_vs_{hand}"] = len(side)
        rows.append(row)

    rates = pd.DataFrame(rows)
    if verbose:
        print(f"    Built rates for {len(rates)} pitchers "
              f"(median {rates['pit_pa_seen'].median():.0f} plate appearances "
              f"seen each).")
    return rates


# Measured on the cached pool: the share of a hitter's plate appearances
# that come against the opposing STARTER rather than the bullpen.
STARTER_PA_SHARE = 0.528

# League reliever rates. Defaults measured from the same data (plate
# appearances not against that game's starter). Passed explicitly by
# callers that compute their own.
LEAGUE_BULLPEN_RATES = {
    "is_hr": 0.0571,
    "is_hit": 0.2279,
    "is_walk": 0.1267,
    "is_k": 0.2336,
}


def measure_bullpen_rates(pa_table: pd.DataFrame) -> tuple:
    """
    Compute the starter share and league reliever rates from a plate-
    appearance table, rather than trusting the constants above.

    A GAME HAS TWO STARTERS, ONE PER SIDE
    -------------------------------------
    The first version of this grouped by game_pk alone and took the first
    pitcher in the group. That identifies the pitcher who faced the AWAY
    team -- so every plate appearance by a home-team hitter against the
    away starter was counted as bullpen.

    The bug was invisible while the pool was small, because a 60-hitter
    pool rarely contains both teams from the same game: whichever side we
    happened to have, its opposing starter was usually the one picked. At
    270 hitters covering all 30 teams, nearly every game has both sides,
    and the measured share collapsed from 52.8% to 22.8%.

    That is not a cosmetic error. `blend_with_bullpen` uses this share to
    weight tonight's actual starting pitcher against generic league
    reliever rates, so a share of 0.228 gives the one pitcher we know the
    identity of less than half the influence he should have -- and the
    "reliever" rates are contaminated with starter plate appearances,
    pulling them the wrong way at the same time.

    THE STARTER IS THE INNING-1 PITCHER, NOT THE FIRST ROW WE HOLD
    -------------------------------------------------------------
    "First pitcher in the group" also assumes our cache contains the
    game's opening plate appearance. It often doesn't -- the pool holds
    some hitters, not whole lineups -- in which case the first row we hold
    is a middle reliever, and every genuine starter plate appearance in
    that game gets scored as bullpen.

    Defining the starter as whoever pitched in inning 1 to that side is
    robust to partial coverage. Validated by the decay: 99.8% of inning-1
    plate appearances are against him, 45% by the sixth, 1.2% by the
    eighth -- which is what a starting pitcher's usage actually looks like.

    An opener still counts as "the starter" here. Openers are rare enough
    that the aggregate is unaffected.

    Returns (starter_share, {target: reliever_rate}). Falls back to the
    module constants when the table lacks what's needed, or when the
    measurement lands somewhere impossible.
    """
    needed = {"game_pk", "pitcher", "is_home", "inning"}
    missing = needed - set(pa_table.columns)
    if missing:
        print(f"    Bullpen exposure: need {sorted(missing)} on the PA table "
              f"-- using the {STARTER_PA_SHARE:.1%} constant.")
        return STARTER_PA_SHARE, dict(LEAGUE_BULLPEN_RATES)

    df = pa_table.copy()
    sort_cols = [c for c in ("game_pk", "at_bat_number") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    starters = (df[df["inning"] == 1]
                .groupby(["game_pk", "is_home"])["pitcher"].first()
                .rename("starter_id"))
    df = df.join(starters, on=["game_pk", "is_home"])

    # Only games where inning 1 is actually in the cache can be classified.
    # Guessing on the rest would mean labelling relievers as starters.
    known = df["starter_id"].notna()
    if known.sum() < 5000:
        print(f"    Bullpen exposure: only {int(known.sum()):,} plate "
              f"appearances have an identifiable starter -- using the "
              f"{STARTER_PA_SHARE:.1%} constant.")
        return STARTER_PA_SHARE, dict(LEAGUE_BULLPEN_RATES)

    classified = df[known].copy()
    classified["vs_starter"] = (
        classified["pitcher"] == classified["starter_id"]).astype(int)
    share = float(classified["vs_starter"].mean())

    # A starting pitcher faces somewhere around half of a hitter's plate
    # appearances. Anything far outside that means the identification
    # broke, not that baseball changed -- exactly the check that would
    # have caught the two bugs above instead of letting 0.228 through.
    if not 0.35 <= share <= 0.75:
        print(f"    Bullpen exposure: measured {share:.1%}, which is not a "
              f"plausible starter share. Falling back to "
              f"{STARTER_PA_SHARE:.1%} -- something is wrong with the "
              f"starter identification, not with the data.")
        return STARTER_PA_SHARE, dict(LEAGUE_BULLPEN_RATES)

    bullpen = classified[classified["vs_starter"] == 0]
    rates = {
        target: float(bullpen[target].mean())
        for target in PITCHER_TARGETS if target in bullpen.columns
    }
    if not rates or bullpen.empty:
        return STARTER_PA_SHARE, dict(LEAGUE_BULLPEN_RATES)
    return share, rates


def blend_with_bullpen(starter_rate: float, bullpen_rate: float,
                       starter_share: float = STARTER_PA_SHARE) -> float:
    """
    Weight the starter's rate by his actual share of a hitter's plate
    appearances, filling the rest with the league reliever rate.

        blended = share * starter + (1 - share) * bullpen

    This is a straight exposure weighting, not a model. Its whole effect
    is to pull an extreme starter's influence back toward league in
    proportion to how much of the game he actually pitches -- which is
    exactly what applying his rate to all four plate appearances failed to
    do.
    """
    return starter_share * float(starter_rate) + (1.0 - starter_share) * float(bullpen_rate)


def pitcher_hand(raw_or_id, cache_lookup=True) -> str:
    """A pitcher's throwing hand from his own cached data. Defaults to
    right-handed, which is roughly 70% of starters."""
    if isinstance(raw_or_id, (int, np.integer)):
        if not cache_lookup:
            return "R"
        raw = load_pitcher_cache(int(raw_or_id))
    else:
        raw = raw_or_id
    if raw is None or raw.empty or "p_throws" not in raw.columns:
        return "R"
    modes = raw["p_throws"].dropna().mode()
    return modes.iloc[0] if len(modes) else "R"
