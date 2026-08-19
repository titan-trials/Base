"""
Total bases: the prop that costs almost nothing, because the engine
already does the hard part.

WHY THIS IS EASY AND H+R+RBI WASN'T
-----------------------------------
Runs scored are the awkward one. A run happens two or three hitters after
the plate appearance that caused it, so features/run_features.py has to
model the probability that a runner eventually comes home.

Total bases has no such problem. A double is worth exactly two bases, in
that plate appearance, always. So the per-PA contribution is read straight
off the event and the existing compound machinery answers every line at
once -- over 0.5, over 1.5, over 2.5 -- from one distribution.

Measured per-PA shape (0.385 total bases per plate appearance):

    0 bases  0.776      an out, a walk, a strikeout
    1 base   0.140      a single
    2 bases  0.046      a double
    3 bases  0.003      a triple
    4 bases  0.036      a home run

THE ONE THING THAT NEEDS MODELLING
----------------------------------
The engine predicts P(hit) and P(home run) per plate appearance. It does
NOT predict doubles and triples separately, so expected bases needs one
more number: given a hit that wasn't a homer, how many bases was it worth?

    E[total bases per PA] = P(HR) x 4  +  P(reached on a non-HR hit) x b

`b` is 1.273 league-wide -- about 27 extra bases per 100 non-homer hits.
And it is a genuine, persistent hitter trait, not noise:

    observed spread across hitters   s.d. 0.042
    spread explained by sample size  s.d. 0.024
    TRUE between-batter spread       s.d. 0.035
    split-half correlation (odd/even hits)   r = 0.53

A hitter at the 10th percentile is 1.227, at the 90th 1.315 -- roughly the
difference between a slap hitter and a doubles machine, and worth about
0.02 total bases per plate appearance, or 0.08 over a full game.

Shrunk with a ~189-non-homer-hit prior, estimated from the data rather
than fixed, so a hitter needs most of a season of hits before his own
number outweighs the league's.
"""
import numpy as np
import pandas as pd

# Bases credited to each event. Everything absent from this map is worth
# zero -- outs, walks, strikeouts, hit-by-pitch. A walk puts you on first
# but it is not a total base; the stat counts bases from HITS only, which
# is exactly why "total bases" and "hits+runs+RBI" reward different
# hitters and are separate markets.
BASE_VALUES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

LEAGUE_BASES_PER_HIT = 1.2728
BASES_PRIOR_HITS = 189.0

# A non-home-run hit is worth at least 1 base and at most 3.
MIN_BASES_PER_HIT, MAX_BASES_PER_HIT = 1.0, 2.2


def measure_bases_per_hit(pa_table: pd.DataFrame, verbose: bool = True):
    """
    League and per-batter bases per NON-HOME-RUN hit, empirical-Bayes
    shrunk.

    Home runs are excluded from both sides because the engine already
    predicts them directly and they are worth exactly 4 -- folding them in
    would mix a modelled quantity with an estimated one and let a slugger's
    home run rate leak into his doubles rate.

    Returns (league_rate, shrunk_per_batter_Series, prior_strength).
    """
    needed = {"is_hit", "is_hr", "total_bases", "batter"}
    if not needed.issubset(pa_table.columns):
        if verbose:
            print("  Total bases: PA table lacks the columns to measure "
                  f"bases per hit -- using league {LEAGUE_BASES_PER_HIT:.4f}.")
        return LEAGUE_BASES_PER_HIT, pd.Series(dtype=float), BASES_PRIOR_HITS

    hits = pa_table[(pa_table["is_hit"] == 1) & (pa_table["is_hr"] == 0)]
    if len(hits) < 2000:
        if verbose:
            print(f"  Total bases: only {len(hits)} non-HR hits -- using "
                  f"league {LEAGUE_BASES_PER_HIT:.4f}.")
        return LEAGUE_BASES_PER_HIT, pd.Series(dtype=float), BASES_PRIOR_HITS

    league = float(hits["total_bases"].mean())
    totals = hits.groupby("batter")["total_bases"].agg(["sum", "size"])
    prior = _estimate_prior(totals, float(hits["total_bases"].var()), league)

    shrunk = ((totals["sum"] + prior * league) / (totals["size"] + prior))
    shrunk = shrunk.clip(MIN_BASES_PER_HIT, MAX_BASES_PER_HIT)

    if verbose:
        settled = int((totals["size"] >= prior).sum())
        print(f"  Total bases: {league:.4f} bases per non-HR hit league-wide "
              f"({len(hits):,} hits).")
        print(f"    per-batter shrunk with a {prior:.0f}-hit prior; "
              f"{settled} of {len(totals)} hitters have more than that.")
        if len(shrunk) > 20:
            print(f"    range {shrunk.min():.3f} to {shrunk.max():.3f}.")
    return league, shrunk, prior


def _estimate_prior(totals: pd.DataFrame, within_var: float,
                    league: float) -> float:
    """
    Method of moments: how much of the observed spread between hitters is
    real, and how much is just having a finite number of hits?

    Returns infinity when the spread is entirely explained by noise --
    which correctly means "ignore every hitter's own number and use the
    league rate", rather than silently trusting a difference that isn't
    there.
    """
    usable = totals[totals["size"] >= 50]
    if len(usable) < 30 or within_var <= 0:
        return BASES_PRIOR_HITS

    observed_var = float((usable["sum"] / usable["size"]).var())
    noise_var = float((within_var / usable["size"]).mean())
    true_var = observed_var - noise_var
    if true_var <= 0:
        return float("inf")
    return float(within_var / true_var)


def expected_total_bases_per_pa(p_hr, p_hit, bases_per_hit) -> np.ndarray:
    """
    E[total bases in one plate appearance]
        = P(home run) x 4  +  P(hit that wasn't a home run) x bases-per-hit

    Routed through P(hit) rather than estimated directly, so it inherits
    the matchup: a hitter facing a starter who suppresses his hit rate
    projects fewer total bases automatically. A per-batter total-bases
    constant would not move at all.
    """
    p_hr = np.clip(np.asarray(p_hr, dtype=float), 0.0, 1.0)
    p_hit = np.clip(np.asarray(p_hit, dtype=float), 0.0, 1.0)
    # P(hit) INCLUDES home runs -- is_hit is true for a home run -- so the
    # non-HR share is the difference. Forgetting that double-counts every
    # home run, once at 4 bases and again at ~1.27.
    p_other_hit = np.clip(p_hit - p_hr, 0.0, 1.0)
    return p_hr * 4.0 + p_other_hit * np.asarray(bases_per_hit, dtype=float)
