"""
How many batters will this starter face, and how many will he strike out?

THE SHAPE OF THE PROBLEM
------------------------
Structurally this is the hitter model with the roles swapped. There, a
per-plate-appearance probability is compounded over a distribution of
plate appearances. Here, a per-batter-faced strikeout probability is
compounded over a distribution of batters faced. model/compound.py
already solves the second half; this file supplies the first.

The difference that matters: a hitter's plate appearances are all against
roughly the same pitcher, but a starter faces NINE DIFFERENT hitters and
then faces them again. Batter faced number n is lineup slot n mod 9, and
with a confirmed lineup you know exactly who that is. Using one average
strikeout rate for all of them throws that away for nothing.

THREE THINGS MEASURED FROM 6,287 STARTS, NOT ASSUMED
-----------------------------------------------------
1. OPENERS ARE NOT STARTERS. "Appeared in the first inning" is the only
   definition of a starter available from a plate-appearance table, and it
   quietly includes openers -- one pitcher in the pool averages 4.0 batters
   faced per "start". Excluding short outings that also end before the
   fourth removes 3.8% of starts and drops the spread of batters faced
   from 5.02 to 4.16. That is not a cosmetic filter; the whole model is a
   distribution over batters faced, and openers were fattening its left
   tail with a different phenomenon.

2. THE ERA SHIFTED, AND POOLING FIVE YEARS WOULD BAKE IT IN. Among the 21
   pitchers with 15+ starts in both 2022 and 2026:

       BF/start   23.02 -> 21.75
       K/start     5.60 ->  4.47
       K per BF    0.245 -> 0.207   (-15%)
       P(over 5.5) 0.489 -> 0.309

   Same men, so this is not the pool composition changing. Some is aging;
   a 15% drop across a mixed-age group is too large for aging alone, and
   the falling batters-faced points at usage -- starters are pulled
   earlier now. Training on all five years would predict P(over 5.5) at
   about 39.5% against a 2026 reality near 34%, a five-point bias on every
   pitcher every night. It would backtest beautifully, because the
   backtest contains the same stale years.

   So: 2025 onward only. 3,092 starts is plenty.

3. BATTERS FACED AND STRIKEOUTS ARE CORRELATED (r = 0.39). A pitcher who
   is dealing stays in longer, so the two quantities being compounded are
   not independent. Marginalising over the batters-faced distribution
   captures most of that, but not all: observed strikeout spread is 1.11x
   what independence predicts. The remainder is handled by dispersing the
   per-batter rate, the same correction compound.py applies to hitters.
"""
import numpy as np
import pandas as pd

from data.game_filter import regular_season_only

# How far back to look, in months, from the day being predicted.
#
# The era shift is not a step between seasons, it is a drift that
# continues inside them. Measured against the same held-out test period,
# the strikeout rate per batter faced comes out:
#
#     all history      0.2345    +6.8% high
#     last 24 months   0.2298    +4.6%
#     last 12 months   0.2246    +2.3%
#     last 6 months    0.2267    +3.2%
#
# Twelve months is the floor of that curve: long enough that a pitcher's
# own history still carries weight, short enough that it is not averaging
# in a league that no longer exists. Six months is not better and costs
# two-thirds of the sample.
#
# A residual +2.3% remains and is not removable -- it is the part of the
# decline that had not happened yet when the window closed.
TRAILING_MONTHS = 12

# A "start" that is this short AND over before this inning is an opener or
# a pitcher who got hurt, not a starter having a bad day. Both conditions
# are required: a genuine starter shelled for 9 batters usually still
# wears the fourth inning, and a 7-inning outing of 9 batters faced is a
# perfect game in progress, not an opener.
OPENER_MAX_BF = 10
OPENER_MAX_INNING = 4

# Shrinkage for a pitcher's expected batters faced, in starts. Estimated
# by variance decomposition on 2025+ data: between-pitcher variance 3.39,
# within-pitcher 15.80, so one pitcher's own history is worth about five
# starts before the league mean stops dominating. Low, because how deep a
# manager lets someone go is only moderately about the pitcher.
BF_PRIOR_STARTS = 4.7

# Shrinkage for a pitcher's strikeout rate, in batters faced. Roughly ten
# starts before his own rate outweighs the league's -- strikeout ability is
# a real and fairly stable skill, but a handful of starts is still mostly
# noise about it.
K_PRIOR_BF = 250.0

# Extra spread in the per-batter strikeout rate: how uncertain the
# pitcher's form tonight is, over and above who he is.
#
# Fitted by matching TOTAL variance, and the word total is the whole
# point. An earlier pass compared the model's mean conditional spread
# (2.17) against the marginal spread of actual strikeout totals (2.36),
# called the model 8% too narrow, and tuned this constant to close the
# gap. That comparison is wrong: those two quantities differ by the
# between-pitcher term, which the model already accounts for by giving
# each pitcher his own rate. Tuning to it double-counted that variance --
# and the giveaway was that "fixing" the spread made Brier skill WORSE,
# which a genuine under-dispersion fix never does.
#
# Done properly, Var(K) = E[conditional var] + Var(conditional mean),
# against an observed 5.562:
#
#     sd = 0.15    5.342    0.96x
#     sd = 0.17    5.489    0.99x   <- chosen
#     sd = 0.19    5.655    1.02x
#
# 0.17 also gives the best over-6.5 calibration (-0.0010) at a Brier cost
# of 0.0007, which is nothing.
K_DISPERSION_SD = 0.17
DISPERSION_POINTS = 5

MAX_BF = 40


def identify_starts(pa: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (pitcher, game) for genuine starting appearances.

    `pa` needs pitcher, game_pk, events, inning, game_date -- the columns
    data/pitcher_data.pitcher_pa_table already produces -- plus game_type
    when it is available.
    """
    pa = regular_season_only(pa)
    work = pa[pa["events"].notna()].copy()
    work["game_date"] = pd.to_datetime(work["game_date"])
    g = (work.groupby(["pitcher", "game_pk"])
             .agg(bf=("events", "size"),
                  k=("events", lambda s: (s == "strikeout").sum()),
                  first_inning=("inning", "min"),
                  last_inning=("inning", "max"),
                  game_date=("game_date", "first"))
             .reset_index())
    starts = g[g["first_inning"] == 1].copy()
    starts["year"] = starts["game_date"].dt.year
    opener = ((starts["bf"] < OPENER_MAX_BF)
              & (starts["last_inning"] < OPENER_MAX_INNING))
    starts["is_opener"] = opener
    return starts


def _tilt_pmf(support: np.ndarray, probs: np.ndarray,
              target_mean: float) -> np.ndarray:
    """
    Re-centre a distribution on a new mean without inventing a shape.

    Exponential tilting: multiply each probability by exp(theta * x) and
    renormalise, solving for the theta that lands the mean on target. Of
    all the distributions with the required mean, this is the one closest
    to the original -- so a pitcher who goes deep gets the league's SHAPE
    of outcomes shifted up, rather than a shape someone made up.

    The same device is already used on plate-appearance distributions in
    features/pa_projection.py.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    lo, hi = float(support.min()), float(support.max())
    target = float(np.clip(target_mean, lo + 1e-6, hi - 1e-6))

    def mean_at(theta):
        w = probs * np.exp(theta * (support - support.mean()))
        w /= w.sum()
        return float((w * support).sum())

    left, right = -5.0, 5.0
    if mean_at(left) > target:
        return _renorm(probs, support, left)
    if mean_at(right) < target:
        return _renorm(probs, support, right)
    for _ in range(80):
        mid = 0.5 * (left + right)
        if mean_at(mid) < target:
            left = mid
        else:
            right = mid
    return _renorm(probs, support, 0.5 * (left + right))


def _renorm(probs, support, theta):
    w = probs * np.exp(theta * (support - support.mean()))
    return w / w.sum()


class WorkloadModel:
    """
    Expected batters faced per start, and the distribution around it.
    """

    def __init__(self, support, league_pmf, league_mean, pitcher_means,
                 pitcher_starts, league_k_per_bf, pitcher_k_rates):
        self.support = support
        self.league_pmf = league_pmf
        self.league_mean = league_mean
        self.pitcher_means = pitcher_means      # shrunk expected BF
        self.pitcher_starts = pitcher_starts    # how many starts back it
        self.league_k_per_bf = league_k_per_bf
        self.pitcher_k_rates = pitcher_k_rates  # shrunk K per batter faced

    @classmethod
    def fit(cls, starts: pd.DataFrame, as_of=None,
            trailing_months: int = TRAILING_MONTHS, verbose: bool = True):
        """
        Fit both halves from ONE window.

        Expected batters faced and strikeouts per batter are estimated
        together, over the same trailing period, because they are two
        readings of the same drifting league. Letting the caller pick a
        window for each is how they end up measured against different
        eras -- which is precisely the bug that made an earlier version of
        this look well calibrated while both halves were wrong in
        opposite directions.
        """
        usable = starts[~starts["is_opener"]].copy()
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            cutoff = as_of - pd.DateOffset(months=trailing_months)
            usable = usable[(usable["game_date"] >= cutoff)
                            & (usable["game_date"] < as_of)]
        if usable.empty:
            raise ValueError(
                "No usable starts in the trailing window. The pitcher "
                "caches may not have been refreshed.")

        counts = usable["bf"].value_counts().sort_index()
        support = counts.index.to_numpy(dtype=float)
        league_pmf = (counts / counts.sum()).to_numpy(dtype=float)
        league_mean = float(usable["bf"].mean())
        k_per_bf = float(usable["k"].sum() / usable["bf"].sum())

        by_pitcher = usable.groupby("pitcher")["bf"]
        n = by_pitcher.size()
        raw = by_pitcher.mean()
        shrunk = (raw * n + league_mean * BF_PRIOR_STARTS) / (n + BF_PRIOR_STARTS)

        # Strikeout rate per batter faced, shrunk toward the league rate
        # over the same window. The prior is in batters faced, so a
        # pitcher needs roughly ten starts before his own rate leads.
        totals = usable.groupby("pitcher").agg(k=("k", "sum"), bf=("bf", "sum"))
        k_rates = ((totals["k"] + K_PRIOR_BF * k_per_bf)
                   / (totals["bf"] + K_PRIOR_BF))

        if verbose:
            span = (f"{usable['game_date'].min().date()} to "
                    f"{usable['game_date'].max().date()}")
            print(f"  Workload model: {len(usable):,} starts, {span}, "
                  f"{usable['pitcher'].nunique()} pitchers.")
            print(f"    League: {league_mean:.2f} batters faced, "
                  f"{k_per_bf:.4f} K per batter.")
            print(f"    Excluded {int(starts['is_opener'].sum())} opener "
                  f"or short outings.")
        return cls(support, league_pmf, league_mean, shrunk.to_dict(),
                   n.to_dict(), k_per_bf, k_rates.to_dict())

    def expected_bf(self, pitcher_id) -> float:
        return float(self.pitcher_means.get(pitcher_id, self.league_mean))

    def k_rate(self, pitcher_id) -> float:
        """Shrunk strikeouts per batter faced for this pitcher."""
        return float(self.pitcher_k_rates.get(pitcher_id, self.league_k_per_bf))

    def starts_seen(self, pitcher_id) -> int:
        return int(self.pitcher_starts.get(pitcher_id, 0))

    def bf_pmf(self, pitcher_id) -> tuple:
        """(support, probabilities) for how many batters he faces tonight."""
        target = self.expected_bf(pitcher_id)
        return self.support, _tilt_pmf(self.support, self.league_pmf, target)


def k_count_distribution(p_by_batter, support, bf_probs,
                         dispersion_sd: float = None) -> np.ndarray:
    """
    Distribution over strikeouts, given the hitters he faces in order.

    `p_by_batter[i]` is the strikeout probability for the i-th batter he
    faces -- so with a confirmed lineup, index 9 is the leadoff hitter
    again. The batters-faced distribution then says how far down that list
    he gets.

    Exact rather than simulated. Walking the lineup once and recording the
    running strikeout distribution after each batter gives the answer for
    every possible number of batters faced in a single pass; weighting
    those by the batters-faced probabilities finishes it. No sampling, so
    no seed and no Monte Carlo error.

    `dispersion_sd` widens the per-batter rate to account for the fact
    that a pitcher's true form tonight is itself unknown -- without it the
    distribution is about 11% too narrow against real starts.
    """
    # Read at CALL time, not bound as a default argument. A module
    # constant used as a default is captured when the function is defined,
    # so reassigning it afterwards -- which is exactly what a tuning sweep
    # does -- changes nothing and the sweep silently reports five
    # identical rows. Found that way.
    if dispersion_sd is None:
        dispersion_sd = K_DISPERSION_SD

    support = np.asarray(support, dtype=int)
    bf_probs = np.asarray(bf_probs, dtype=float)
    max_bf = int(support.max())
    p_by_batter = np.asarray(p_by_batter, dtype=float)[:max_bf]
    if len(p_by_batter) < max_bf:   # lineup shorter than the tail: repeat it
        reps = int(np.ceil(max_bf / max(len(p_by_batter), 1)))
        p_by_batter = np.tile(p_by_batter, reps)[:max_bf]

    weight_of = dict(zip(support.tolist(), bf_probs.tolist()))

    # Mixture over plausible "how good is he tonight" multipliers, which is
    # how the extra spread gets in. Weights are a discretised normal.
    if dispersion_sd > 0:
        offsets = np.linspace(-2.0, 2.0, DISPERSION_POINTS)
        mult_weights = np.exp(-0.5 * offsets ** 2)
        mult_weights /= mult_weights.sum()
        multipliers = 1.0 + dispersion_sd * offsets
    else:
        multipliers, mult_weights = np.array([1.0]), np.array([1.0])

    total = np.zeros(max_bf + 1)
    for mult, mw in zip(multipliers, mult_weights):
        probs = np.clip(p_by_batter * mult, 0.0, 1.0)
        running = np.zeros(max_bf + 1)
        running[0] = 1.0
        out = np.zeros(max_bf + 1)
        for faced in range(max_bf + 1):
            w = weight_of.get(faced)
            if w:
                out += w * running
            if faced < max_bf:
                p = probs[faced]
                nxt = running * (1.0 - p)
                nxt[1:] += running[:-1] * p
                running = nxt
        total += mw * out
    s = total.sum()
    return total / s if s > 0 else total


def prob_over(k_dist: np.ndarray, line: float) -> float:
    """P(strikeouts > line). Lines are half-integers, so no tie handling."""
    first = int(np.floor(line)) + 1
    return float(k_dist[first:].sum()) if first < len(k_dist) else 0.0
