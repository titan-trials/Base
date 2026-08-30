"""
Form deviation -- is this hitter playing above or below his own level?

WHY THIS IS NOT THE FEATURE THAT KEEPS FAILING
-----------------------------------------------
Five feature experiments have now failed in this project for one reason,
recorded in CONTEXT.md as "granular features aren't additive when they
overlap an existing one":

    contact quality           permutation p = 0.841
    weather                   permutation p = 0.448
    calendar & park splits    permutation p = 0.617
    chase rate                (same finding, earlier)
    wRC+ / OPS+ / xwOBA / barrel%   (feature_lab, V8)

Every one of those is a LEVEL -- how hard he hits it, how good he is, how
warm it is, which park. The rolling rates already encode all of that, so a
second measurement of the same latent quantity costs variance and buys
nothing.

This is a CHANGE. The model's entire view of a hitter is:

    bat_{target}_150, bat_{target}_600, bat_{target}_career

Three window lengths of the same rate, and no derivative. Nothing in the
feature list measures a player against his own norm. That is the hole this
fills, and it is the reason this is worth testing when weather was not.

It is not a promise that it works. If this fails too, that is strong
evidence the level/change distinction rescues nothing, and it should be
the last feature experiment of this shape.

WINDOWS IN PLATE APPEARANCES, NOT GAMES
----------------------------------------
"Hot over the last 10 games" is the natural phrasing and the wrong
denominator, for the reason rate_features.py already gives: ten games is
~43 plate appearances for a leadoff hitter and ~35 for someone batting
eighth, so the same window means different amounts of evidence for
different players. Windows here are counted in PA.

25 PA is the short window because that is where Green and Zwiebel found
the hot-hand signal in 2 million MLB at-bats. 75 PA is the longer one,
roughly three weeks of everyday play.

WHY RECENT IS NOT SHRUNK
-------------------------
The baseline is shrunk, using the same empirical-Bayes machinery as the
rest of the engine. The recent window deliberately is NOT.

Shrinking both sides would pull them toward the same population mean and
erase most of the difference being measured -- the estimator would be
working against the question. The baseline needs to be the best estimate
of true talent, which is what shrinkage gives. The recent window needs to
be what actually happened, which is the raw rate. The z-score handles the
uncertainty explicitly in its denominator instead.

WHAT THIS DOES NOT DO
---------------------
It does not touch DEFAULT_PA_WINDOWS. rate_feature_cols() builds the
model's feature list from that tuple, so adding a short window there to
get recent form would silently add columns to every per-PA model --
changing the model while trying to build a marker that observes it.

Nothing here reaches the model. These columns are written to the slate
file and displayed, and that is all, until the self-test says otherwise.
"""
import numpy as np
import pandas as pd

from features.rate_features import estimate_prior_strength, shrink

# Outcomes the deviation is measured on, with the sign that makes
# "positive means hot".
#
# Chosen for how fast they stabilise, not for how interesting they sound.
# Strikeout rate settles at roughly 60 PA, which is the fastest of any
# outcome statistic available in the PA table. Batting average needs about
# 910 at-bats -- a season and a half -- so a hot streak measured in hits
# per at-bat over 25 PA would be almost pure noise. `is_hit` is included
# anyway because it is cheap and the self-test can tell them apart; if it
# turns out to contribute nothing, drop it rather than guessing now.
#
# Exit velocity is the one that should be here and is not: it is a direct
# physical measurement rather than an outcome filtered through defence and
# luck, and it stabilises faster than anything on that list. It is absent
# because PA_COLUMNS does not carry `launch_speed`. That is v2.
FORM_TARGETS = {
    "is_k":    -1.0,   # more strikeouts than usual = cold
    "is_hit":  +1.0,
    "reached": +1.0,
}

FORM_SHORT_PA = 25     # Green-Zwiebel
FORM_LONG_PA = 75      # ~3 weeks of everyday play
FORM_BASELINE_PA = 600  # established level, same as bat_{target}_600

# Below this many plate appearances in the window there is nothing to say,
# and the z-score's denominator starts producing large numbers from small
# evidence. Those rows come back NaN and are reported as "Unknown", not as
# "Normal" -- a hitter we cannot assess is not a hitter we assessed as
# average.
FORM_MIN_PA = 20

# Threshold for the display state, in standard deviations of the composite.
#
# The composite is STANDARDISED before this is applied, and that step is
# not cosmetic. The per-target values are honest z-scores, but averaging
# two correlated windows produces something with a standard deviation of
# about 0.64, not 1.0 -- measured on 35,200 simulated plate appearances
# from players with no real change in ability. Left unscaled, a reported
# "z = 1.2" would actually be 1.9 sigma, and every threshold anyone chose
# later would be wrong by that factor without announcing itself.
#
# False-positive rate at each threshold, measured on 52,800 simulated
# plate appearances from players whose true rates NEVER move -- so every
# flag below is pure noise:
#
#     threshold    flagged    per 270-hitter slate
#     |z| > 1.00    31.9%       86
#     |z| > 1.25    21.3%       57
#     |z| > 1.50    13.5%       36
#     |z| > 1.75     8.1%       22
#     |z| > 2.00     4.5%       12     <- chosen
#     |z| > 2.25     2.3%        6
#     |z| > 2.50     1.1%        3
#
# 2.0 flags about a dozen hitters a slate by chance alone, before any real
# streaks are added on top. Rare enough that a flag is worth reading,
# common enough that the self-test accumulates evidence at a usable rate.
#
# Note these are two-sided rates on a now-standardised composite, so they
# track the normal distribution closely. An earlier draft of this table was
# measured before standardisation and read 1.8% at 1.5 -- the same numbers
# in different units, which is exactly the confusion the scaling step
# exists to prevent.
HOT_Z = 2.0
COLD_Z = -2.0


def _rolling_rate(grouped, window, min_periods):
    """
    Raw trailing rate and its sample size, per batter.

    `.shift(1)` is the lookahead guard and is the single most important
    line in this file. Without it a plate appearance sees its own outcome
    in its own form score -- and because this marker is graded against
    those same outcomes, the self-test would report a spectacular result
    for a column that already knew the answer. A silent, flattering,
    completely wrong success.
    """
    successes = grouped.transform(
        lambda s: s.rolling(window, min_periods=min_periods).sum().shift(1))
    trials = grouped.transform(
        lambda s: s.rolling(window, min_periods=min_periods).count().shift(1))
    return successes, trials


def add_form_deviation(pa_table: pd.DataFrame,
                       short_pa: int = FORM_SHORT_PA,
                       long_pa: int = FORM_LONG_PA,
                       baseline_pa: int = FORM_BASELINE_PA) -> pd.DataFrame:
    """
    Per-PA form deviation for each batter.

    Adds, for each target in FORM_TARGETS:
        form_{target}_z_short   z of the last `short_pa` vs own baseline
        form_{target}_z_long    same over `long_pa`
    plus the composite:
        form_z_short, form_z_long, form_z   (mean of short and long)
        form_state                          Hot / Normal / Cold / Unknown

    Expects the chronological sort that features/pa_table.py applies. It
    does not re-sort: sorting here would silently disagree with the rolling
    rates if the two ever used different keys.
    """
    df = pa_table.copy()
    short_cols, long_cols = [], []

    for target, sign in FORM_TARGETS.items():
        if target not in df.columns:
            continue

        grouped = df.groupby("batter", sort=False)[target]

        # Baseline: shrunk, same estimator the model uses. The prior
        # strength is estimated from career totals across the population,
        # exactly as add_batter_rolling_rates does, so "this player's
        # level" means the same thing in both places.
        career = df.groupby("batter")[target].agg(["sum", "size"])
        prior_mean = float(df[target].mean())
        k = estimate_prior_strength(career["sum"], career["size"])

        base_succ, base_trials = _rolling_rate(grouped, baseline_pa, FORM_MIN_PA)
        baseline = shrink(base_succ.fillna(0), base_trials.fillna(0),
                          prior_mean, k)
        baseline = pd.Series(baseline, index=df.index)

        for window, bucket, suffix in ((short_pa, short_cols, "short"),
                                       (long_pa, long_cols, "long")):
            succ, trials = _rolling_rate(grouped, window, FORM_MIN_PA)

            # Raw, unshrunk -- see the module docstring.
            recent = succ / trials

            # Standard error of the recent rate UNDER THE BASELINE. Using
            # the baseline's variance rather than the observed rate's keeps
            # the denominator from collapsing when a small window happens
            # to be all-successes or all-failures, which would otherwise
            # produce an infinite z from 20 plate appearances.
            se = np.sqrt(baseline * (1.0 - baseline) / trials)

            z = (recent - baseline) / se.replace(0.0, np.nan)
            z = z * sign
            z = z.where(trials >= FORM_MIN_PA)

            col = f"form_{target}_z_{suffix}"
            df[col] = z
            bucket.append(col)

    df["form_z_short"] = df[short_cols].mean(axis=1) if short_cols else np.nan
    df["form_z_long"] = df[long_cols].mean(axis=1) if long_cols else np.nan
    raw = df[["form_z_short", "form_z_long"]].mean(axis=1)

    # Restore unit scale. The mean of correlated z-scores is not itself a
    # z-score -- see the HOT_Z comment. Dividing by the observed spread
    # makes "1.5" mean 1.5 standard deviations again, so a threshold picked
    # here means the same thing as a threshold picked by whoever reads this
    # in six months.
    #
    # The scale is stored rather than hidden, because it is computed from
    # whatever population was passed in. On a full-history PA table it is
    # stable at ~0.64; on a tiny frame it would be meaningless, which is
    # why it is visible in .attrs instead of silently applied.
    scale = float(raw.std())
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    df["form_z"] = raw / scale
    df.attrs["form_scale"] = scale

    df["form_state"] = np.select(
        [df["form_z"].isna(), df["form_z"] >= HOT_Z, df["form_z"] <= COLD_Z],
        ["Unknown", "Hot", "Cold"],
        default="Normal",
    )
    return df


def latest_form_by_batter(pa_with_form: pd.DataFrame) -> pd.DataFrame:
    """
    One row per batter: his form as of his most recent plate appearance.

    This is what the slate export needs -- tonight's prediction wants the
    state going INTO tonight, which is the value carried by the last PA
    already in the table.
    """
    cols = [c for c in pa_with_form.columns
            if c.startswith("form_")] + ["batter"]
    last = (pa_with_form[cols]
            .groupby("batter", as_index=False)
            .tail(1)
            .reset_index(drop=True))
    return last
