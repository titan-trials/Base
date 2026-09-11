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
from features.pa_table import NON_PA_EVENTS, K_EVENTS

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

# What the prior MEAN is, and why it is not the league mean.
#
# A pitcher with no starts in the window is not an average starter having
# his first start of the year. He is a call-up, a swingman or a spot
# starter, and managers give those a short leash. Measured on the cached
# starts in the trailing twelve months to 2026-09-05, batters faced by how
# many prior starts the pitcher had in the window:
#
#     0 prior     21.02        3-4 prior   21.60
#     1 prior     21.73        5-9 prior   22.66
#     2 prior     22.02        10+ prior   23.21
#
# against a window mean of 22.8. The league mean is what an ESTABLISHED
# starter faces; the prior should be what an UNKNOWN one faces. So the
# shrinkage target, and the fallback for a pitcher never seen, is the mean
# over starts with at most this many prior starts in the window. The
# market already prices this -- a debut gets a 3.5 line, not a 5.5 -- and
# the old league-mean fallback was where most of the "edge" against thin
# pitchers on the market comparison came from.
NEW_PITCHER_MAX_PRIOR = 2

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

# Outs recorded per start: the same prior strength as batters faced,
# because it is the same kind of quantity measured in the same units
# (one number per start) and there is no reason to believe a pitcher's
# outs stabilise at a different rate than his batters faced.
OUTS_PRIOR_STARTS = BF_PRIOR_STARTS
MAX_OUTS = 27

# How many outs a plate appearance records, used for AT MOST ONE plate
# appearance per start -- see _outs_by_start.
#
# Deliberately not the primary method. Measured against real data the map
# is wrong in a way no map can fix: `single` comes out at 0.018 outs and
# `strikeout` at 0.989, because runners get thrown out BETWEEN plate
# appearances and the batter's event knows nothing about it. Applied to
# all 25 plate appearances of a start that drift compounds; applied to
# the last one only, it is bounded.
OUT_EVENTS = {
    "field_out": 1, "strikeout": 1, "force_out": 1, "sac_fly": 1,
    "sac_bunt": 1, "fielders_choice_out": 1, "other_out": 1,
    "grounded_into_double_play": 2, "double_play": 2,
    "strikeout_double_play": 2, "sac_fly_double_play": 2,
    "sac_bunt_double_play": 2, "triple_play": 3,
}


def _outs_by_start(work: pd.DataFrame) -> pd.Series:
    """
    Outs recorded per (pitcher, game), from the half-inning structure.

    THE METHOD, AND WHY NOT THE OBVIOUS ONE
    ---------------------------------------
    The obvious way is to map each event to the outs it records and sum.
    That is wrong, and measurably so: over one pitcher's 2,000 plate
    appearances the implied outs for `single` came out at 0.018 and for
    `strikeout` at 0.989, because runners are thrown out BETWEEN plate
    appearances -- caught stealing, out advancing, picked off -- and the
    batter's event cannot see it. A few percent per plate appearance is
    half an out per start.

    A completed half-inning does not have that problem. It is worth
    exactly `3 - outs when he entered it`, whatever happened on the bases,
    because the inning ends at three outs by definition. And a starter who
    appears in inning n+1 must have completed inning n.

    So every half-inning but his last is exact, and the event map is used
    only for the final plate appearance of the final inning, and only when
    he was pulled mid-inning.

    Checked against published season totals for Corbin Burnes: 17.55 outs
    per start in 2025 against a published 64.1 IP over 11 starts (17.55),
    18.22 in 2024 against 194.1 over 32 (18.22), 18.33 in 2022 against
    202.0 over 33 (18.36) -- one out adrift across a full season.
    """
    need = {"outs_when_up", "inning", "at_bat_number", "events"}
    if not need.issubset(work.columns):
        return pd.Series(dtype=float)
    w = work.dropna(subset=["outs_when_up", "inning"]).copy()
    if w.empty:
        return pd.Series(dtype=float)
    w["inning"] = w["inning"].astype(int)
    w["outs_when_up"] = w["outs_when_up"].astype(int)
    half_keys = ["pitcher", "game_pk", "inning"]
    if "inning_topbot" in w.columns:
        half_keys.append("inning_topbot")
    w = w.sort_values(["pitcher", "game_pk", "inning", "at_bat_number"])

    half = (w.groupby(half_keys, sort=False)
             .agg(entry=("outs_when_up", "min"),
                  last_outs=("outs_when_up", "last"),
                  last_event=("events", "last"))
             .reset_index())
    final = half.groupby(["pitcher", "game_pk"])["inning"].transform("max")
    made = half["last_event"].map(OUT_EVENTS).fillna(0).to_numpy()
    outs = np.where(half["inning"].to_numpy() != final.to_numpy(),
                    3 - half["entry"].to_numpy(),
                    half["last_outs"].to_numpy() + made
                    - half["entry"].to_numpy())
    half["outs"] = np.clip(outs, 0, 3)
    return half.groupby(["pitcher", "game_pk"])["outs"].sum()


def identify_starts(pa: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (pitcher, game) for genuine starting appearances.

    `pa` needs pitcher, game_pk, events, inning, game_date -- the columns
    data/pitcher_data.pitcher_pa_table already produces -- plus game_type
    when it is available.
    """
    pa = regular_season_only(pa)
    work = pa[pa["events"].notna() & ~pa["events"].isin(NON_PA_EVENTS)].copy()
    work["game_date"] = pd.to_datetime(work["game_date"])
    g = (work.groupby(["pitcher", "game_pk"])
             .agg(bf=("events", "size"),
                  k=("events", lambda s: s.isin(K_EVENTS).sum()),
                  first_inning=("inning", "min"),
                  last_inning=("inning", "max"),
                  game_date=("game_date", "first"))
             .reset_index())
    starts = g[g["first_inning"] == 1].copy()
    outs = _outs_by_start(work)
    if len(outs):
        starts["outs"] = (starts.set_index(["pitcher", "game_pk"]).index
                          .map(outs).astype(float))
    else:
        starts["outs"] = np.nan
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
                 pitcher_starts, league_k_per_bf, pitcher_k_rates,
                 new_pitcher_mean=None):
        self.support = support
        self.league_pmf = league_pmf
        self.league_mean = league_mean
        self.pitcher_means = pitcher_means      # shrunk expected BF
        self.pitcher_starts = pitcher_starts    # how many starts back it
        self.league_k_per_bf = league_k_per_bf
        self.pitcher_k_rates = pitcher_k_rates  # shrunk K per batter faced
        # Expected BF for a pitcher with no usable history. See
        # NEW_PITCHER_MAX_PRIOR.
        self.new_pitcher_mean = (float(new_pitcher_mean)
                                 if new_pitcher_mean is not None
                                 else float(league_mean))

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

        # The prior mean is what a pitcher with little history faces, not
        # what the league does -- see NEW_PITCHER_MAX_PRIOR. Prior-start
        # counts are taken inside the window, which is also what the
        # caller can know about a pitcher on the night.
        #
        # "Prior starts" counts the pitcher's whole cached history, not
        # just the window: an established starter's first outing of the
        # window is not a debut. Counting inside the window alone diluted
        # the thin group with veterans and left the new-pitcher rate a
        # hair under league, which the backtest exposed (thin pitchers
        # still 0.5 K per start over their projection).
        career = starts[~starts["is_opener"]].sort_values(
            ["pitcher", "game_date", "game_pk"])
        if as_of is not None:
            career = career[career["game_date"] < as_of]
        career = career.assign(prior_all=career.groupby("pitcher").cumcount())
        in_window = career.index.isin(usable.index)
        thin = career[(career["prior_all"] <= NEW_PITCHER_MAX_PRIOR) & in_window]
        new_pitcher_mean = (float(thin["bf"].mean()) if len(thin) >= 30
                            else league_mean)

        # Which prior a pitcher gets depends on whether he IS new. A
        # veteran with a short window sample is a veteran, and pulling him
        # toward the call-up mean costs 0.3 batters faced across the
        # established group (measured). So: career starts before the
        # origin <= NEW_PITCHER_MAX_PRIOR -> new-pitcher priors; otherwise
        # league priors. Unseen pitchers are new by definition.
        career_starts = career.groupby("pitcher").size()
        is_new = (career_starts <= NEW_PITCHER_MAX_PRIOR + 1)

        by_pitcher = usable.groupby("pitcher")["bf"]
        n = by_pitcher.size()
        raw = by_pitcher.mean()
        prior_bf = pd.Series(np.where(is_new.reindex(n.index).fillna(True),
                                      new_pitcher_mean, league_mean), index=n.index)
        shrunk = ((raw * n + prior_bf * BF_PRIOR_STARTS)
                  / (n + BF_PRIOR_STARTS))

        # Strikeout rate per batter faced, shrunk toward the NEW-PITCHER
        # rate over the same window -- the same argument as the batters
        # faced prior above. A pitcher with little history is not an
        # average starter with a small sample; he is a call-up or a
        # swingman, and those strike out fewer hitters per batter faced.
        # The 2026 rolling-origin backtest (backtest_k_props.py) showed
        # pitchers with under five starts in the window running 0.55 K per
        # start over their projection when shrunk toward the league rate.
        # The prior is in batters faced, so a pitcher needs roughly ten
        # starts before his own rate leads.
        thin_k = (float(thin["k"].sum() / thin["bf"].sum())
                  if len(thin) >= 30 and thin["bf"].sum() > 0 else k_per_bf)
        totals = usable.groupby("pitcher").agg(k=("k", "sum"), bf=("bf", "sum"))
        prior_k = pd.Series(np.where(is_new.reindex(totals.index).fillna(True),
                                     thin_k, k_per_bf), index=totals.index)
        k_rates = ((totals["k"] + K_PRIOR_BF * prior_k)
                   / (totals["bf"] + K_PRIOR_BF))

        if verbose:
            span = (f"{usable['game_date'].min().date()} to "
                    f"{usable['game_date'].max().date()}")
            print(f"  Workload model: {len(usable):,} starts, {span}, "
                  f"{usable['pitcher'].nunique()} pitchers.")
            print(f"    League: {league_mean:.2f} batters faced, "
                  f"{k_per_bf:.4f} K per batter; a pitcher with <= "
                  f"{NEW_PITCHER_MAX_PRIOR} prior starts faces "
                  f"{new_pitcher_mean:.2f} at {thin_k:.4f} K per batter "
                  f"(the shrinkage targets).")
            print(f"    Excluded {int(starts['is_opener'].sum())} opener "
                  f"or short outings.")
        model = cls(support, league_pmf, league_mean, shrunk.to_dict(),
                    n.to_dict(), k_per_bf, k_rates.to_dict(),
                    new_pitcher_mean=new_pitcher_mean)
        model.new_pitcher_k = float(thin_k)
        # Career context, kept so a zero can be told apart from a zero.
        #
        # starts_seen is the TRAILING WINDOW count, and it is 0 for two
        # completely different pitchers: a genuine debut, and a veteran
        # back from a long absence. Corbin Burnes on 2026-09-08 had 108
        # career starts and none since 2025-06-01 -- fifteen months out.
        # The model is right to refuse to project him on pre-injury form,
        # but "0 starts" reads as "nobody" when it means "nobody LATELY",
        # and the two want different priors: a pitcher returning from a
        # long layoff is on a leash, which a debutant is not.
        #
        # Not used by the model. Carried so the dashboard can say which
        # kind of zero this is, the same way the form marker is measured
        # and shown without feeding anything.
        # ---- outs recorded, fitted exactly like batters faced ------
        #
        # Same window, same shrinkage shape, for the reason given at the
        # top of fit(): two readings of one drifting league measured over
        # different periods is how the two halves end up wrong in
        # opposite directions while the total looks right.
        #
        # Kept SEPARATE from batters faced rather than derived from it.
        # Causally outs come first -- a manager pulls a starter by innings
        # and pitch count, and batters faced is the consequence of outs
        # plus whoever reached -- so rebuilding BF on top of outs would be
        # the more correct model. It would also disturb the K prop, which
        # is validated and working. Coherence is checked instead: see
        # implied_baserunners below.
        outs_col = usable["outs"].dropna() if "outs" in usable else pd.Series(dtype=float)
        if len(outs_col) >= 100:
            oc = usable.dropna(subset=["outs"])
            counts_o = oc["outs"].round().astype(int).clip(0, MAX_OUTS) \
                .value_counts().sort_index()
            model.outs_support = counts_o.index.to_numpy(dtype=float)
            model.outs_league_pmf = (counts_o / counts_o.sum()).to_numpy(dtype=float)
            model.outs_league_mean = float(oc["outs"].mean())
            thin_o = thin.dropna(subset=["outs"]) if "outs" in thin else thin.iloc[0:0]
            model.new_pitcher_outs = (float(thin_o["outs"].mean())
                                      if len(thin_o) >= 30
                                      else model.outs_league_mean)
            by_o = oc.groupby("pitcher")["outs"]
            n_o, raw_o = by_o.size(), by_o.mean()
            prior_o = pd.Series(
                np.where(is_new.reindex(n_o.index).fillna(True),
                         model.new_pitcher_outs, model.outs_league_mean),
                index=n_o.index)
            model.pitcher_outs = (
                (raw_o * n_o + prior_o * OUTS_PRIOR_STARTS)
                / (n_o + OUTS_PRIOR_STARTS)).to_dict()
            if verbose:
                print(f"    Outs recorded: league {model.outs_league_mean:.2f} "
                      f"({model.outs_league_mean / 3:.2f} innings); a pitcher "
                      f"with <= {NEW_PITCHER_MAX_PRIOR} prior starts records "
                      f"{model.new_pitcher_outs:.2f}.")
        elif verbose:
            print("    Outs recorded: too few starts carry an outs figure "
                  "to fit; the outs prop will be skipped.")

        model.career_start_counts = career_starts.to_dict()
        model.last_start_dates = (career.groupby("pitcher")["game_date"]
                                  .max().to_dict())
        return model

    def expected_bf(self, pitcher_id) -> float:
        return float(self.pitcher_means.get(pitcher_id, self.new_pitcher_mean))

    def k_rate(self, pitcher_id) -> float:
        """Shrunk strikeouts per batter faced for this pitcher. A pitcher
        never seen gets the new-pitcher rate, not the league's."""
        return float(self.pitcher_k_rates.get(
            pitcher_id, getattr(self, "new_pitcher_k", self.league_k_per_bf)))

    def starts_seen(self, pitcher_id) -> int:
        return int(self.pitcher_starts.get(pitcher_id, 0))

    def career_starts(self, pitcher_id) -> int:
        """Every non-opener start in the cache before tonight."""
        return int(getattr(self, "career_start_counts", {}).get(pitcher_id, 0))

    def days_since_last_start(self, pitcher_id, as_of) -> float:
        """NaN when he has never started; else days since his last one."""
        last = getattr(self, "last_start_dates", {}).get(pitcher_id)
        if last is None or pd.isna(last):
            return float("nan")
        return float((pd.Timestamp(as_of) - pd.Timestamp(last)).days)

    def expected_outs(self, pitcher_id) -> float:
        """Shrunk outs recorded. NaN when the outs half was not fitted."""
        if not _outs_ok(self):
            return float("nan")
        return float(getattr(self, "pitcher_outs", {}).get(
            pitcher_id, getattr(self, "new_pitcher_outs",
                                self.outs_league_mean)))

    def outs_pmf(self, pitcher_id) -> tuple:
        """
        (support, probabilities) for outs recorded tonight.

        The league SHAPE tilted onto his mean, the same device bf_pmf
        uses -- of all the distributions with the required mean this is
        the one closest to the league's, so a pitcher who goes deep gets
        the real shape of outcomes shifted up rather than an invented one.
        """
        if not _outs_ok(self):
            return None, None
        return (self.outs_support,
                _tilt_pmf(self.outs_support, self.outs_league_pmf,
                          self.expected_outs(pitcher_id)))

    def implied_baserunners(self, pitcher_id) -> float:
        """
        Projected batters faced minus projected outs.

        The coherence check between the two halves. It should land near
        the league's baserunners per start (hits + walks + hit batsmen,
        roughly 10-11). A number far from that means the two fits
        disagree about the same pitcher's night, which is exactly the
        failure mode that fitting them over one window is meant to avoid.
        """
        if not _outs_ok(self):
            return float("nan")
        return self.expected_bf(pitcher_id) - self.expected_outs(pitcher_id)

    def bf_pmf(self, pitcher_id) -> tuple:
        """(support, probabilities) for how many batters he faces tonight."""
        target = self.expected_bf(pitcher_id)
        return self.support, _tilt_pmf(self.support, self.league_pmf, target)


def _outs_ok(model) -> bool:
    """Whether the outs half of the model was fitted at all."""
    return getattr(model, "outs_support", None) is not None


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


def _odds(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return p / (1.0 - p)


def per_batter_k_probs(batter_rates, batter_league: float,
                       pitcher_rate: float, platoon_factor=1.0) -> np.ndarray:
    """
    The per-batter-faced strikeout probability the K prop compounds.

    THIS IS THE NUMBER THE BACKTEST VALIDATED, NOW ON THE DEPLOYED PATH
    -------------------------------------------------------------------
    Until 2026-09-05, predict_slate walked the lineup with the HITTER
    model's `p_is_k_vs_starter` -- a logistic regression whose pitcher input
    was the starter's career strikeout rate since 2022, no trailing window,
    shrunk toward a 2022-2026 league mean. `WorkloadModel.k_rate`, the
    12-month rate this module fits and the backtest scored, was computed
    and then only written to the CSV. On the 9/01 and 9/04 slates the two
    agreed at Spearman 0.65 and the deployed rate carried 1.7x the spread;
    live skill on 32 starts was -0.14 against a backtest of +0.06. Stale
    input, wider spread, confident and wrong at the extremes.

    log5 WITH SEPARATE BASELINES
    ----------------------------
    The hitter's rate is measured against the pitchers HE faced (his own
    pool table); the pitcher's rate is measured against the 12-month
    starter league in this module. Plain log5 assumes one shared baseline,
    so it is written out:

        odds(p) = [odds(batter) / odds(batter_league)] x odds(pitcher) x platoon

    The bracket is the hitter's odds ratio -- how much more or less he
    strikes out than his league. Multiplying the pitcher's own odds by it
    lands the answer in the PITCHER's league, which is the one the batters
    faced distribution and K_DISPERSION_SD were fitted in. A league-average
    hitter leaves the pitcher's rate untouched, which is the sanity check.

    `platoon_factor` is an odds multiplier for the handedness matchup,
    measured by the caller from its own plate-appearance table (same-hand
    versus opposite-hand strikeout odds, relative to overall). Pass 1.0 to
    ignore it.
    """
    ratio = _odds(batter_rates) / _odds(batter_league)
    odds = ratio * _odds(pitcher_rate) * np.asarray(platoon_factor, dtype=float)
    return odds / (1.0 + odds)


def starter_exposure_by_slot(support, bf_probs, n_slots: int = 9) -> np.ndarray:
    """
    Expected number of times the starter faces each lineup slot.

    Not a model. The starter faces batters 1, 2, 3, ... up to BF, and
    batter n is lineup slot ((n - 1) mod 9) + 1. So for a given BF the count
    for slot s is exact:

        #{ n <= BF : (n - 1) mod 9 == s - 1 }

        BF = 22   slots 1-4 face him 3 times, slots 5-9 twice
        BF = 27   every slot 3 times
        BF = 18   every slot twice

    Averaged over the batters-faced distribution, that is each slot's
    expected exposure to the starter. Divided by the slot's expected plate
    appearances it is the starter SHARE that `blend_with_bullpen` needs --
    per pitcher and per slot, instead of the single 0.528 constant that was
    applied to every hitter against every starter. An ace at 26 BF gives
    the leadoff hitter ~0.66; a five-inning starter at 19 gives the
    9-hitter ~0.55. pitcher_data.py's own table puts the constant's cost at
    +/-5 points of home-run probability at the extremes.

    Returns an array indexed 0..n_slots-1 (slot 1 at index 0).
    """
    support = np.asarray(support, dtype=int)
    bf_probs = np.asarray(bf_probs, dtype=float)
    bf_probs = bf_probs / bf_probs.sum()
    out = np.zeros(n_slots)
    for bf, w in zip(support, bf_probs):
        full, rem = divmod(int(bf), n_slots)
        counts = np.full(n_slots, full, dtype=float)
        counts[:rem] += 1.0
        out += w * counts
    return out


def prob_over(k_dist: np.ndarray, line: float) -> float:
    """P(strikeouts > line). Lines are half-integers, so no tie handling."""
    first = int(np.floor(line)) + 1
    return float(k_dist[first:].sum()) if first < len(k_dist) else 0.0
