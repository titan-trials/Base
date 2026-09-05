"""
Per-plate-appearance rate estimation -- the "estimate" half of the V5
engine. model/compound.py handles the "compound" half.

THE PROBLEM THIS SOLVES
-----------------------
You want a number for "what is this hitter's chance of a home run in this
plate appearance, tonight, against this pitcher." Three separate estimation
problems hide in that sentence:

  1. How good is the BATTER, really? His last 50 PA are a tiny sample.
  2. How good is the PITCHER at preventing it? Usually an even tinier
     sample from a batter-centric data pull.
  3. How do you COMBINE a batter rate and a pitcher rate into one number?

This file answers all three. None of it is machine learning, and that's
deliberate -- these are estimation problems with known, better-than-ML
answers at this sample size. The ML layer sits on top and adjusts for
what's left.

1. SHRINKAGE, WITH THE STRENGTH ESTIMATED RATHER THAN GUESSED
--------------------------------------------------------------
A hitter with 3 home runs in 40 plate appearances has an observed rate of
7.5%. His true rate is almost certainly closer to the league's ~4%,
because 40 PA is nearly no information. The fix is the same empirical-Bayes
shrinkage used for the V4 park splits:

    shrunk = (successes + k * prior) / (trials + k)

but in V4 the prior strength `k` was hard-coded at 30 games, which was a
guess. Here it's ESTIMATED from the data by the beta-binomial method of
moments:

    total observed spread between players
      = real skill differences  +  binomial noise

    Subtract the binomial part (which is computable exactly), and what's
    left is the real spread. A large real spread means players genuinely
    differ, so trust individual samples more (small k). A small real
    spread means most apparent differences are noise, so shrink hard
    (large k).

This matters because the right k differs enormously by statistic. Walk
rate is a strong individual skill -- players really do differ, so k is
small and a modest sample is trusted. Home run rate per PA is noisier
relative to its spread, so it shrinks harder. The data decides, not me.

2. ROLLING WINDOWS MEASURED IN PLATE APPEARANCES, NOT GAMES
------------------------------------------------------------
Every earlier version used "last 10 games." That's the wrong denominator:
a 10-game window is ~43 plate appearances for a leadoff hitter and ~35 for
someone batting eighth, so the same window means different amounts of
evidence for different players. Windows here are counted in PA, so the
sample size is the sample size.

3. COMBINING BATTER AND PITCHER: THE ODDS-RATIO METHOD
-------------------------------------------------------
The obvious move is to hand batter rate and pitcher rate to a regression
and let it work out the weights. The better move is to use the formula
that already describes how two rates combine against a shared baseline
(known in baseball circles as "log5", and structurally identical to
combining log-odds):

    odds_matchup = (odds_batter * odds_pitcher) / odds_league

In probability terms: a .400-OBP hitter facing a pitcher who allows a
.400 OBP produces more than .400, because both parties are above average
relative to the league they were measured against. A linear model cannot
express that interaction unless you hand it this exact product -- so it's
computed here and handed over as a feature, rather than hoped for.

LOOKAHEAD GUARD
---------------
Every rolling quantity is shifted by one plate appearance within each
batter, and the table is sorted chronologically once in
features/pa_table.py. A PA's features never include that PA's own outcome.
"""
import numpy as np
import pandas as pd

# Rolling windows in PLATE APPEARANCES. ~150 PA is roughly a month of
# everyday play (recent form); ~600 is roughly a full season (established
# level). Both are offered because they answer different questions and the
# model can weigh them.
DEFAULT_PA_WINDOWS = (150, 600)

# Outcomes modelled. Each gets its own shrinkage strength, because the
# balance of skill to noise is different for every one of them.
RATE_TARGETS = ("is_hr", "is_hit", "is_walk", "is_k")


def estimate_prior_strength(successes, trials, min_trials: int = 50) -> float:
    """
    Beta-binomial method of moments: how many plate appearances of prior
    is one player's sample worth?

    Returns k for use in (x + k*prior) / (n + k).

    The logic in three lines:
      observed variance between players = true variance + binomial noise
      binomial noise is computable exactly, as mean(p(1-p)/n)
      whatever is left over is the true spread, and k = p(1-p)/true_var - 1

    When the leftover is zero or negative -- meaning the players differ no
    more than pure chance would produce -- k is returned as a very large
    number, which shrinks every player to the population mean. That is the
    correct answer in that case, not a failure: it says "there is no
    demonstrated skill difference here to preserve."
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    keep = trials >= min_trials
    successes, trials = successes[keep], trials[keep]
    if len(trials) < 3:
        return 500.0  # too few players to estimate anything; shrink hard

    prior_mean = successes.sum() / trials.sum()
    rates = successes / trials

    observed_var = rates.var(ddof=1)
    binomial_var = np.mean(prior_mean * (1.0 - prior_mean) / trials)
    true_var = observed_var - binomial_var

    if true_var <= 1e-9:
        return 5000.0

    k = prior_mean * (1.0 - prior_mean) / true_var - 1.0
    # Clip to a sane band. Below ~10 PA of prior the shrinkage does nothing;
    # above ~5000 everyone is the population mean either way.
    return float(np.clip(k, 10.0, 5000.0))


def estimate_prior_strength_counts(sums, trials, within_var,
                                   min_trials: int = 50) -> float:
    """
    The same method-of-moments idea for a per-PA rate of a COUNT (RBI,
    total bases) rather than a 0/1 flag.

        observed variance of per-player means
          = true variance + mean(within-player variance / n)

    The binomial term p(1-p)/n is replaced by each player's own sample
    variance over n. Returns k for (sum + k * prior) / (n + k), in plate
    appearances. Clipped to the same band as the binary version.
    """
    sums = np.asarray(sums, dtype=float)
    trials = np.asarray(trials, dtype=float)
    within_var = np.asarray(within_var, dtype=float)
    keep = (trials >= min_trials) & np.isfinite(within_var)
    sums, trials, within_var = sums[keep], trials[keep], within_var[keep]
    if len(trials) < 3:
        return 500.0
    rates = sums / trials
    observed_var = rates.var(ddof=1)
    noise_var = float(np.mean(within_var / trials))
    true_var = observed_var - noise_var
    if true_var <= 1e-12:
        return 5000.0
    # In the binomial version k = p(1-p)/true_var - 1; the analogue with a
    # pooled within-player variance in place of p(1-p).
    k = float(np.mean(within_var)) / true_var - 1.0
    return float(np.clip(k, 10.0, 5000.0))


def shrink(successes, trials, prior_mean: float, k: float):
    """The empirical-Bayes estimate. Vectorised over arrays."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    return (successes + k * prior_mean) / (trials + k)


# game_pk stamped on the synthetic "tonight" rows. Negative so it can never
# collide with a real MLB game id, and a constant so callers can find and
# remove the rows again.
TONIGHT_GAME_PK = -1


def add_tonight_rows(pa_table: pd.DataFrame, batter_ids, game_date) -> pd.DataFrame:
    """
    Append one synthetic, outcome-less plate appearance per batter, dated
    tonight, so the rolling features land on a row that includes his most
    recent real plate appearance.

    WHY THIS EXISTS
    ---------------
    Every rolling column is `.shift(1)` -- a plate appearance never sees
    its own outcome. predict_slate used to read tonight's features off a
    batter's LAST REAL ROW, and the shifted window on that row ends one
    plate appearance before it. So every hitter, every night, was
    predicted from a window missing his most recent trip to the plate.
    One PA in 150 is small; it is also systematic, and it is free to fix.

    The synthetic row has NaN outcomes. Rolling sums and counts skip NaN,
    so its shifted window is exactly the un-shifted window of the last
    real row -- which is the number wanted. Rows are found again by
    `game_pk == TONIGHT_GAME_PK` and must be dropped before anything trains
    or aggregates.

    Sorts the result the way features/pa_table.build_pa_table does, so the
    synthetic row is the last one for each batter.
    """
    ids = [b for b in pd.Series(list(batter_ids)).dropna().unique()]
    if not ids:
        return pa_table
    last = (pa_table[pa_table["batter"].isin(ids)]
            .sort_values(["batter", "game_date", "game_pk", "at_bat_number"]
                         if "at_bat_number" in pa_table.columns
                         else ["batter", "game_date", "game_pk"])
            .groupby("batter").tail(1).copy())
    if last.empty:
        return pa_table
    last["game_pk"] = TONIGHT_GAME_PK
    # Strictly after every real row, even when the slate date is not
    # (re-running a past date), so the sort below keeps it last.
    last["game_date"] = max(pd.Timestamp(game_date),
                            pd.Timestamp(pa_table["game_date"].max()) + pd.Timedelta(days=1))
    if "at_bat_number" in last.columns:
        last["at_bat_number"] = 0
    if "inning" in last.columns:
        last["inning"] = np.nan
    last["pitcher"] = np.nan
    for col in ("is_hr", "is_hit", "is_walk", "is_k", "rbi", "reached",
                "reached_nonhr", "hrr_certain", "total_bases"):
        if col in last.columns:
            last[col] = np.nan
    out = pd.concat([pa_table, last], ignore_index=True)
    sort_cols = [c for c in ["batter", "game_date", "game_pk", "at_bat_number"]
                 if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def add_batter_rolling_rates(pa_table: pd.DataFrame,
                             targets=RATE_TARGETS,
                             windows=DEFAULT_PA_WINDOWS) -> pd.DataFrame:
    """
    Shrunk rolling per-PA rates for each batter, over windows measured in
    plate appearances.

    Produces, for each target and window:
        bat_{target}_{window}    shrunk rate over the trailing window
    plus:
        bat_{target}_career      shrunk rate over everything prior

    The shrinkage strength k is estimated once per target from the
    population, then applied to every window. Note this means a 150-PA
    window shrinks harder than a career window automatically -- not
    because of a rule, but because 150 is smaller than k for most of
    these stats, which is the honest consequence of having less evidence.
    """
    import model_flags
    df = pa_table.copy()
    priors = {}
    trailing = model_flags.LEAGUE_PRIOR_TRAILING_DAYS

    for target in targets:
        if target not in df.columns:
            continue
        # "count", not "size": a synthetic tonight row (add_tonight_rows)
        # carries a NaN outcome and must not count as a trial.
        career_totals = df.groupby("batter")[target].agg(["sum", "count"])
        prior_mean = float(df[target].mean())
        k = estimate_prior_strength(career_totals["sum"], career_totals["count"])
        priors[target] = {"prior_mean": prior_mean, "k": k}

        # The shrinkage TARGET. Default: the mean over the whole cache.
        # With model_flags.LEAGUE_PRIOR_TRAILING_DAYS set, each row is
        # shrunk toward the league rate over the trailing N days as of its
        # own date -- so a 150-PA window in 2026 is pulled toward 2026's
        # league, not toward a 2022-2026 blend. Off until flag_lab.py says
        # it earns its keep.
        row_prior = prior_mean
        if trailing:
            row_prior = trailing_league_rate(df, target, int(trailing)).to_numpy()

        grouped = df.groupby("batter", sort=False)[target]
        for window in windows:
            successes = grouped.transform(
                lambda s: s.rolling(window, min_periods=20).sum().shift(1)
            )
            trials = grouped.transform(
                lambda s: s.rolling(window, min_periods=20).count().shift(1)
            )
            df[f"bat_{target}_{window}"] = shrink(
                successes.fillna(0), trials.fillna(0), row_prior, k
            )

        successes = grouped.transform(lambda s: s.expanding(min_periods=1).sum().shift(1))
        trials = grouped.transform(lambda s: s.expanding(min_periods=1).count().shift(1))
        df[f"bat_{target}_career"] = shrink(
            successes.fillna(0), trials.fillna(0), row_prior, k
        )

    df.attrs["rate_priors"] = priors
    return df


def trailing_league_rate(df: pd.DataFrame, target: str, days: int) -> pd.Series:
    """
    League rate of `target` over the trailing `days` ending the day BEFORE
    each row's date, aligned to df's index. Falls back to the full-cache
    mean where the window holds under 5,000 plate appearances.
    """
    daily = (df.dropna(subset=[target])
               .groupby(df["game_date"].dt.normalize())[target]
               .agg(["sum", "count"]).sort_index())
    idx = pd.date_range(daily.index.min(), df["game_date"].max().normalize()
                        + pd.Timedelta(days=1), freq="D")
    daily = daily.reindex(idx, fill_value=0)
    roll = daily.rolling(f"{days}D").sum().shift(1)
    rate = roll["sum"] / roll["count"]
    rate = rate.where(roll["count"] >= 5000, float(df[target].mean()))
    return df["game_date"].dt.normalize().map(rate).fillna(float(df[target].mean()))


def add_pitcher_rolling_rates(pa_table: pd.DataFrame,
                              targets=RATE_TARGETS) -> pd.DataFrame:
    """
    Shrunk career-to-date rates ALLOWED by the opposing pitcher.

    Important limitation, stated rather than buried: this project pulls
    Statcast per BATTER, so a pitcher is only visible in the plate
    appearances where he faced one of the batters in the pool. A pitcher
    might appear 20 times here despite having faced 2,000 hitters that
    season. The shrinkage is therefore doing very heavy lifting -- most
    pitchers will sit close to the league mean, which is the honest
    representation of how little is actually known about them from this
    data.

    Making this feature genuinely strong would need a bulk pitcher-side
    Statcast pull. That's the single highest-value data upgrade available
    to this project.
    """
    df = pa_table.copy()

    for target in targets:
        if target not in df.columns:
            continue
        prior_mean = float(df[target].mean())
        career = df.groupby("pitcher")[target].agg(["sum", "count"])
        k = estimate_prior_strength(career["sum"], career["count"], min_trials=20)

        # Sorted WITHIN the game as well as across games. With game_date
        # alone the order inside a game was whatever the concat left, so a
        # training plate appearance's pitcher rate could include later
        # plate appearances from the same game -- a small lookahead that
        # the batter side never had, because pa_table sorts by
        # at_bat_number for batters.
        order = [c for c in ("pitcher", "game_date", "game_pk", "at_bat_number")
                 if c in df.columns]
        ordered = df.sort_values(order)
        grouped = ordered.groupby("pitcher", sort=False)[target]
        import model_flags
        window_days = model_flags.PITCHER_RATE_WINDOW_DAYS
        if window_days:
            # Trailing time window instead of the whole career, matching
            # what build_pitcher_rates does at serving time under the
            # same flag. Per-pitcher time-based rolling, then the usual
            # one-row shift.
            successes = _rolling_days(ordered, "pitcher", target, int(window_days), "sum")
            trials = _rolling_days(ordered, "pitcher", target, int(window_days), "count")
        else:
            successes = grouped.transform(lambda s: s.expanding(min_periods=1).sum().shift(1))
            trials = grouped.transform(lambda s: s.expanding(min_periods=1).count().shift(1))

        shrunk = pd.Series(
            shrink(successes.fillna(0), trials.fillna(0), prior_mean, k),
            index=successes.index,
        )
        df[f"pit_{target}_allowed"] = shrunk.reindex(df.index)
        df[f"pit_{target}_n"] = trials.reindex(df.index).fillna(0)

    return df


def _rolling_days(ordered: pd.DataFrame, key: str, target: str, days: int,
                  how: str) -> pd.Series:
    """Per-`key` trailing sum/count of `target` over `days`, shifted one
    row so a row never sees itself. `ordered` must already be sorted by
    (key, game_date, ...). Returns a Series aligned to ordered.index."""
    out = pd.Series(np.nan, index=ordered.index)
    for _, g in ordered.groupby(key, sort=False):
        s = g[target]
        r = s.set_axis(pd.DatetimeIndex(g["game_date"])).rolling(f"{days}D")
        vals = (r.sum() if how == "sum" else r.count()).to_numpy()
        vals = np.concatenate([[np.nan], vals[:-1]])
        out.loc[g.index] = vals
    return out


def odds(p):
    """p / (1 - p), clipped so a rate of exactly 0 or 1 can't explode."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return p / (1.0 - p)


def log5(batter_rate, pitcher_rate, league_rate):
    """
    Combine a batter rate and a pitcher rate against a shared league
    baseline, via the odds ratio.

        odds_matchup = odds_batter * odds_pitcher / odds_league

    Why not just average them: averaging says a great hitter facing a
    great pitcher lands in the middle, which is wrong in a specific way.
    Both parties' rates were measured against roughly league-average
    opposition, so each carries a baseline that has to be divided back out
    before they're combined. The odds ratio does exactly that, and it
    reduces correctly at the edges -- a league-average pitcher leaves the
    batter's rate untouched, which is the sanity check to remember.
    """
    combined = odds(batter_rate) * odds(pitcher_rate) / odds(league_rate)
    return combined / (1.0 + combined)


def add_matchup_rates(pa_table: pd.DataFrame, targets=RATE_TARGETS,
                      batter_window="career") -> pd.DataFrame:
    """
    The log5 blend of batter and pitcher rates, per target.

    This is the single most useful column per outcome: one number that
    already encodes "how good is he, how good is the guy on the mound, and
    how do those combine." The ML layer's job is to adjust it for
    everything log5 doesn't know about -- park, platoon, form, fatigue --
    rather than to rediscover the matchup itself.
    """
    df = pa_table.copy()
    for target in targets:
        bat_col = f"bat_{target}_{batter_window}"
        pit_col = f"pit_{target}_allowed"
        if bat_col not in df.columns or pit_col not in df.columns:
            continue
        league_rate = float(df[target].mean())
        df[f"matchup_{target}"] = log5(df[bat_col], df[pit_col], league_rate)
    return df


def rate_feature_cols(targets=RATE_TARGETS, windows=DEFAULT_PA_WINDOWS) -> dict:
    """
    Feature columns grouped BY TARGET, since each per-PA model is trained
    separately. Predicting a walk should not lean on home-run features;
    keeping the groups separate keeps each model interpretable and stops
    28 loosely-related columns from diluting every coefficient (the
    failure mode measured in V4).
    """
    groups = {}
    for target in targets:
        cols = [f"bat_{target}_{w}" for w in windows]
        cols += [f"bat_{target}_career", f"pit_{target}_allowed", f"matchup_{target}"]
        cols += platoon_feature_cols() + ["is_home", "park_hr_factor"]
        groups[target] = cols
    return groups


def platoon_feature_cols() -> list:
    """`platoon_edge` alone, or the two same-hand indicators under
    model_flags.PLATOON_SPLIT (left-on-left and right-on-right get their
    own coefficients; the opposite-hand matchup is the reference)."""
    import model_flags
    if model_flags.PLATOON_SPLIT:
        return ["same_hand_lhb", "same_hand_rhb"]
    return ["platoon_edge"]
