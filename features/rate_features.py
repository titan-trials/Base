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


def shrink(successes, trials, prior_mean: float, k: float):
    """The empirical-Bayes estimate. Vectorised over arrays."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    return (successes + k * prior_mean) / (trials + k)


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
    df = pa_table.copy()
    priors = {}

    for target in targets:
        if target not in df.columns:
            continue
        career_totals = df.groupby("batter")[target].agg(["sum", "size"])
        prior_mean = float(df[target].mean())
        k = estimate_prior_strength(career_totals["sum"], career_totals["size"])
        priors[target] = {"prior_mean": prior_mean, "k": k}

        grouped = df.groupby("batter", sort=False)[target]
        for window in windows:
            successes = grouped.transform(
                lambda s: s.rolling(window, min_periods=20).sum().shift(1)
            )
            trials = grouped.transform(
                lambda s: s.rolling(window, min_periods=20).count().shift(1)
            )
            df[f"bat_{target}_{window}"] = shrink(
                successes.fillna(0), trials.fillna(0), prior_mean, k
            )

        successes = grouped.transform(lambda s: s.expanding(min_periods=1).sum().shift(1))
        trials = grouped.transform(lambda s: s.expanding(min_periods=1).count().shift(1))
        df[f"bat_{target}_career"] = shrink(
            successes.fillna(0), trials.fillna(0), prior_mean, k
        )

    df.attrs["rate_priors"] = priors
    return df


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
        career = df.groupby("pitcher")[target].agg(["sum", "size"])
        k = estimate_prior_strength(career["sum"], career["size"], min_trials=20)

        grouped = df.sort_values(["pitcher", "game_date"]).groupby("pitcher", sort=False)[target]
        successes = grouped.transform(lambda s: s.expanding(min_periods=1).sum().shift(1))
        trials = grouped.transform(lambda s: s.expanding(min_periods=1).count().shift(1))

        shrunk = pd.Series(
            shrink(successes.fillna(0), trials.fillna(0), prior_mean, k),
            index=successes.index,
        )
        df[f"pit_{target}_allowed"] = shrunk.reindex(df.index)
        df[f"pit_{target}_n"] = trials.reindex(df.index).fillna(0)

    return df


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
        cols += ["platoon_edge", "is_home", "park_hr_factor"]
        groups[target] = cols
    return groups
