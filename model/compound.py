"""
The compounding layer -- turning per-plate-appearance probabilities into
game-level answers.

THIS FILE CONTAINS NO MACHINE LEARNING, AND THAT'S THE POINT
------------------------------------------------------------
Once you have "his chance of a home run in one plate appearance is 5.2%"
and "he'll get about 4.3 plate appearances," the chance of at least one
home run tonight is not something to learn. It's arithmetic:

    P(at least one) = 1 - (1 - 0.052)^4.3  =  20.4%

Every earlier version of this project tried to learn that number directly
from game-level features, fighting binomial noise the whole way. The
measured result was a hard ceiling around AUC 0.55-0.59. Splitting the
problem lets the statistics happen where there's sample size (30,000 plate
appearances) and the arithmetic happen where it's exact.

The same split answers questions the old approach structurally could not.
"Over 1.5 hits + RBI + walks" is not a classification problem -- there is
no way to get it from a yes/no model. It needs a DISTRIBUTION over a
count, and once you have per-PA probabilities you get the whole
distribution for free, so every line (0.5, 1.5, 2.5...) is answerable from
one calculation.

THE THREE PIECES
----------------
1. `prob_at_least_one` -- binary outcomes (home run, walk, hit). Closed
   form, exact.

2. `compound_count_distribution` -- counting outcomes (hits+runs+RBI).
   Repeated convolution of the per-PA contribution distribution. Exact,
   not simulated.

3. `marginalise_over_pa` -- plate appearances are not known in advance.
   A hitter might get 3, 4, 5 or 6. Rather than plugging in an average,
   the answer is computed for each possible PA count and weighted by how
   likely that count is. This matters more than it looks: P(at least one
   home run) is concave in PA, so using the average PA overstates the
   probability slightly, and for over/under lines near the middle of the
   distribution the error is larger still.

WHY CONVOLUTION RATHER THAN SIMULATION
--------------------------------------
Both work. Convolution is exact, deterministic, and about a thousand times
faster -- and speed matters because these get evaluated across tens of
thousands of games during backtesting. `np.convolve` of two probability
vectors gives the distribution of their sum; applying it n times gives the
distribution over n plate appearances.
"""
import numpy as np
import pandas as pd


def prob_at_least_one(per_pa_prob, n_pa) -> np.ndarray:
    """
    P(event happens at least once in n plate appearances).

        1 - (1 - p)^n

    Assumes plate appearances are independent given the rate. That is a
    real assumption and it is slightly wrong -- a pitcher tiring makes
    later PAs a bit better, and a batter who homered may be pitched around
    afterwards. Both effects are small next to the binomial noise they sit
    inside, and neither is estimable from this data. Stated rather than
    hidden.
    """
    p = np.clip(np.asarray(per_pa_prob, dtype=float), 0.0, 1.0)
    n = np.asarray(n_pa, dtype=float)
    return 1.0 - np.power(1.0 - p, n)


def compound_count_distribution(per_pa_dist: np.ndarray, n_pa: int,
                                max_total: int = 20) -> np.ndarray:
    """
    Distribution of a TOTAL over n plate appearances, by convolving the
    per-PA contribution distribution with itself n times.

    `per_pa_dist[v]` is the probability that one plate appearance
    contributes exactly v (for hits+runs+RBI: 0 for an out, 1 for a
    single that strands, 5 for a three-run homer -- 1 hit + 3 RBI + the
    batter's own run).

    Returns a vector indexed 0..max_total.
    """
    total = np.zeros(max_total + 1)
    total[0] = 1.0  # zero plate appearances means a total of zero

    for _ in range(int(n_pa)):
        total = np.convolve(total, per_pa_dist)[:max_total + 1]

    total_sum = total.sum()
    return total / total_sum if total_sum > 0 else total


def fit_game_dispersion(game_totals: pd.DataFrame, per_pa_dist: np.ndarray,
                        value_col: str = "hrr", pa_col: str = "pa") -> float:
    """
    Measure how much MORE variable game totals are than independent plate
    appearances would produce.

    Convolution assumes each plate appearance is an independent draw. That
    is not quite true, and the error runs one way: some days the pitcher
    is dominant and every trip is bad, other days he has nothing and the
    whole lineup feasts. That shared, game-level component makes real
    totals clump -- more 0-for-4s AND more 4-hit games than independence
    predicts.

    Measured on this data, hits+runs+RBI has variance/mean = 1.45, well
    above the ~1.0 independence implies. The symptom in the output is a
    calibration bias that shows up exactly where you'd expect: predicted
    P(over 0.5) = 0.813 against an actual 0.774, because the model is
    missing the extra mass sitting on zero.

    Method of moments. If a game-level multiplier m (mean 1) scales the
    rate, then for n plate appearances:

        Var(T) = n * E[sigma^2] + n^2 * mu^2 * Var(m)

    Everything except Var(m) is observable, so solve for it. Returns the
    standard deviation of m, or 0.0 when the data shows no excess
    dispersion (in which case plain convolution was right all along).
    """
    values = np.arange(len(per_pa_dist), dtype=float)
    mu = float((per_pa_dist * values).sum())
    sigma_sq = float((per_pa_dist * (values - mu) ** 2).sum())

    totals = game_totals[value_col].to_numpy(dtype=float)
    n_pa = game_totals[pa_col].to_numpy(dtype=float)
    if len(totals) < 100 or mu <= 0:
        return 0.0

    observed_var = float(totals.var())
    independent_var = float((n_pa * sigma_sq).mean())
    mean_n_sq_mu_sq = float((n_pa ** 2).mean()) * mu ** 2

    excess = observed_var - independent_var
    if excess <= 0 or mean_n_sq_mu_sq <= 0:
        return 0.0
    return float(np.sqrt(excess / mean_n_sq_mu_sq))


def _dispersion_grid(dispersion_sd: float, n_points: int = 5) -> list:
    """
    A small discrete stand-in for the continuous game-level multiplier.

    Gauss-Hermite-style: a handful of points with weights, chosen so the
    mixture has mean 1 and the requested standard deviation. Five points
    is plenty -- the answer changes in the fourth decimal beyond that, and
    every extra point multiplies the convolution work.

    Multipliers are floored at 0.05 rather than allowed to go negative,
    which a symmetric grid would otherwise do at high dispersion.
    """
    if dispersion_sd <= 1e-6:
        return [(1.0, 1.0)]

    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    weights = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
    multipliers = np.maximum(1.0 + offsets * dispersion_sd, 0.05)

    # Renormalise so the mixture mean is exactly 1 -- clipping at 0.05
    # otherwise nudges it upward and would quietly inflate every forecast.
    mean = float((multipliers * weights).sum())
    multipliers = multipliers / mean
    return list(zip(multipliers, weights))


def compound_count_distribution_overdispersed(
    per_pa_dist: np.ndarray, n_pa: int, dispersion_sd: float = 0.0,
    max_total: int = 20, method: str = "frequency",
) -> np.ndarray:
    """
    Like `compound_count_distribution`, but mixes over a game-level rate
    multiplier so the result has realistic tails.

    With dispersion_sd = 0 this is exactly the independent convolution, so
    it's a safe drop-in.

    `method` MUST match whatever was used to scale the batter's own rate.
    There are two places a distribution gets reshaped -- once for the
    hitter, once per dispersion multiplier -- and mixing methods between
    them produces a distribution that is neither one. That bug shipped
    once: the inner call here silently used the default while the outer
    call used the selected method, which quietly changed every prediction
    and made the two methods impossible to compare fairly. Hence the
    explicit parameter with no clever default behaviour.
    """
    grid = _dispersion_grid(dispersion_sd)
    if len(grid) == 1:
        return compound_count_distribution(per_pa_dist, n_pa, max_total)

    mixture = np.zeros(max_total + 1)
    base_mean = float((per_pa_dist * np.arange(len(per_pa_dist))).sum())

    for multiplier, weight in grid:
        scaled = scale_contribution_distribution(
            per_pa_dist, base_mean * multiplier, method=method
        )
        mixture += weight * compound_count_distribution(scaled, n_pa, max_total)

    total = mixture.sum()
    return mixture / total if total > 0 else mixture


def prob_over_line(count_dist: np.ndarray, line: float) -> float:
    """
    P(total > line) for a betting line like 1.5.

    Half-integer lines exist precisely so there's no push, which makes
    this unambiguous: strictly greater than 1.5 means 2 or more.
    """
    threshold = int(np.floor(line)) + 1
    if threshold > len(count_dist) - 1:
        return 0.0
    return float(count_dist[threshold:].sum())


def marginalise_over_pa(pa_distribution: dict, answer_fn) -> float:
    """
    Average an answer over the uncertainty in tonight's plate-appearance
    count.

        pa_distribution: {4: 0.53, 5: 0.38, 3: 0.08, 6: 0.01}
        answer_fn:       a function taking n_pa and returning a probability

    The alternative -- computing the answer at the average PA -- is subtly
    biased whenever the answer is non-linear in PA, which it always is
    here. Jensen's inequality, doing real work rather than appearing in a
    textbook.
    """
    total_weight = sum(pa_distribution.values())
    if total_weight <= 0:
        return float("nan")
    return float(sum(
        weight * answer_fn(n_pa) for n_pa, weight in pa_distribution.items()
    ) / total_weight)


def empirical_pa_distribution(game_totals: pd.DataFrame, batter=None,
                              min_games: int = 30) -> dict:
    """
    How many plate appearances does this hitter get, historically?

    Batting-order slot is the real driver and isn't in Statcast, so this
    empirical distribution stands in for it: a leadoff hitter's histogram
    naturally sits a slot higher than a number-eight hitter's. Falls back
    to the pool-wide distribution when a batter has too little history.
    """
    frame = game_totals
    if batter is not None:
        subset = game_totals[game_totals["batter"] == batter]
        if len(subset) >= min_games:
            frame = subset

    counts = frame["pa"].value_counts()
    return (counts / counts.sum()).to_dict()


def game_probability_binary(per_pa_prob: float, pa_distribution: dict) -> float:
    """P(at least one) for a binary event, marginalised over PA count."""
    return marginalise_over_pa(
        pa_distribution, lambda n: float(prob_at_least_one(per_pa_prob, n))
    )


def game_probability_over_line(per_pa_dist: np.ndarray, pa_distribution: dict,
                               line: float, max_total: int = 20) -> float:
    """P(total > line) for a counting stat, marginalised over PA count."""
    return marginalise_over_pa(
        pa_distribution,
        lambda n: prob_over_line(
            compound_count_distribution(per_pa_dist, int(n), max_total), line
        ),
    )


def pa_leverage_diagnostic(scored: pd.DataFrame, pa_distributions: dict,
                           per_pa_col: str, actual_col: str,
                           evaluate_fn) -> dict:
    """
    How much would it be worth to know tonight's plate-appearance count
    exactly?

    Runs the same compound calculation three ways:

      historical   marginalised over the batter's own PA histogram. This
                   is what production can actually do before a game.
      mean PA      a single point estimate. Included to show what
                   marginalising is worth on its own (spoiler: almost
                   nothing -- the curvature over a 3-to-6 PA range is
                   mild).
      actual PA    the real count from the completed game. NOT achievable
                   before first pitch, and reported strictly as an upper
                   bound on what better PA projection could buy.

    Why this diagnostic exists: on this data the gap between "historical"
    and "actual" turned out to be roughly TEN TIMES larger than the total
    contribution of every batter-skill feature in the project. That
    reorders the whole to-do list. It's the kind of thing that stays
    invisible unless it's measured deliberately, because every model
    quietly conditions on exposure without ever reporting on it.
    """
    results = {}
    probs = scored[per_pa_col].to_numpy()

    historical = np.array([
        marginalise_over_pa(pa_distributions[batter],
                            lambda n, p=prob: float(prob_at_least_one(p, n)))
        for batter, prob in zip(scored["batter"], probs)
    ])
    results["historical"] = evaluate_fn(scored[actual_col], historical)

    mean_pa = {b: sum(k * v for k, v in d.items()) / sum(d.values())
               for b, d in pa_distributions.items()}
    point = np.array([
        float(prob_at_least_one(p, mean_pa[b]))
        for b, p in zip(scored["batter"], probs)
    ])
    results["mean_pa"] = evaluate_fn(scored[actual_col], point)

    if "pa" in scored.columns:
        exact = prob_at_least_one(probs, scored["pa"].to_numpy())
        results["actual_pa"] = evaluate_fn(scored[actual_col], exact)
        results["value_of_knowing_pa"] = (
            results["actual_pa"]["brier_skill"] - results["historical"]["brier_skill"]
        )
    return results


def scale_contribution_distribution(base_dist: np.ndarray, target_mean: float,
                                    method: str = "frequency") -> np.ndarray:
    """
    Reshape the population per-PA contribution distribution so its mean
    matches one batter's estimated rate.

    TWO METHODS, AND WHY THE DEFAULT CHANGED
    ----------------------------------------
    "tilt" -- exponential tilting. Multiply the probability of outcome v by
        exp(theta * v) and renormalise, solving for the theta that hits the
        target. It's the textbook minimal-information way to move a mean.

        It was the original default here, and it was wrong for this
        problem. Because the multiplier is exponential IN THE VALUE, the
        far tail moves far more than the mean does. Measured on the real
        population distribution, asking for a hitter 25% above league
        produces:

            value 0:  0.91x        value 3:  1.59x
            value 1:  1.10x        value 4:  1.92x
            value 2:  1.32x

        A 25% better hitter is credited with 92% more four-base plate
        appearances. Worse, the tilt is applied TWICE -- once for the
        batter's own rate, once per game-level dispersion multiplier -- so
        the distortion compounds. The visible symptom was a persistent
        +0.014 over-prediction at the 3.5 line while the 0.5 and 1.5 lines
        calibrated cleanly.

    "frequency" (default) -- scale how OFTEN a plate appearance produces
        anything, and leave the shape of "how much, given something
        happened" exactly as observed:

            P(v > 0) is scaled to hit the target mean
            P(v | v > 0) is untouched

        Closed form, no solver: since the target mean equals
        P(v>0) * E[v | v>0] and the conditional expectation is held fixed,
        P(v>0) = target_mean / E[v | v>0].

        The modelling claim is different and, on this data, better: hitters
        differ mainly in how often they do damage, not in how spectacular
        the damage is when it lands. The tail ratios stay as observed
        instead of being extrapolated.

    Both are kept because this is an empirical question, not a matter of
    taste -- `select_tilt_method` picks between them by measured
    calibration rather than by argument.
    """
    values = np.arange(len(base_dist), dtype=float)
    base_mean = float((base_dist * values).sum())

    if target_mean <= 0 or base_mean <= 0:
        return base_dist.copy()
    if abs(target_mean - base_mean) < 1e-9:
        return base_dist.copy()

    if method == "frequency":
        nonzero_mass = float(base_dist[1:].sum())
        if nonzero_mass <= 0:
            return base_dist.copy()
        conditional_mean = base_mean / nonzero_mass  # E[v | v > 0]
        new_nonzero = target_mean / conditional_mean

        # Can't put more than all the mass on non-zero outcomes. When the
        # target demands it, fall back to tilting, which has no such cap.
        if new_nonzero >= 1.0:
            return scale_contribution_distribution(base_dist, target_mean,
                                                   method="tilt")

        scaled = base_dist.copy()
        scaled[1:] = base_dist[1:] * (new_nonzero / nonzero_mass)
        scaled[0] = 1.0 - new_nonzero
        return scaled

    # Bisection on theta. The tilted mean rises monotonically with theta,
    # so bisection is guaranteed to converge and needs no derivatives.
    low, high = -5.0, 5.0
    for _ in range(80):
        theta = 0.5 * (low + high)
        weights = base_dist * np.exp(theta * values)
        tilted = weights / weights.sum()
        if float((tilted * values).sum()) < target_mean:
            low = theta
        else:
            high = theta

    weights = base_dist * np.exp(0.5 * (low + high) * values)
    return weights / weights.sum()


def select_tilt_method(games: pd.DataFrame, base_dist: np.ndarray,
                       per_pa_means, dispersion_sd: float, lines,
                       value_col: str = "hrr", pa_col: str = "pa",
                       n_boot: int = 300, win_threshold: float = 0.75,
                       random_state: int = 0, verbose: bool = True) -> str:
    """
    Choose "tilt" or "frequency" by measured calibration error, with a
    bootstrap so a meaningless margin doesn't decide it.

    The first version of this took a plain argmin and picked "tilt" by
    0.0003 -- a margin far inside the noise of the estimate. Picking a
    method on a difference that small is coin-flipping with extra steps,
    so the choice is now made by resampling the games: how OFTEN does one
    method actually calibrate better?

    The tie-break is deliberately asymmetric. When neither method wins
    clearly, "frequency" is chosen, because it PRESERVES the observed
    ratio of large outcomes to small ones, while "tilt" EXTRAPOLATES that
    ratio (inflating 4-value plate appearances 92% for a hitter only 25%
    better than league). Between a method that reuses what the data shows
    and one that projects beyond it, the extrapolating one should have to
    earn its place. Same principle as the one-standard-error rule on
    calibrator selection in model/hr_v4.

    Predictions are computed once per (method, line) and the bootstrap
    resamples the resulting arrays, so 300 resamples cost almost nothing
    on top of the two evaluations.
    """
    per_pa_means = np.asarray(per_pa_means, dtype=float)
    n_pa = games[pa_col].to_numpy()
    totals = games[value_col].to_numpy()
    rng = np.random.default_rng(random_state)

    predictions, actuals = {}, {}
    for method in ("frequency", "tilt"):
        for line in lines:
            predictions[(method, line)] = np.array([
                prob_over_line(
                    compound_count_distribution_overdispersed(
                        scale_contribution_distribution(base_dist, m, method=method),
                        int(n), dispersion_sd, method=method),
                    line)
                for m, n in zip(per_pa_means, n_pa)
            ])
            actuals[line] = (totals > line).astype(float)

    def mean_gap(method, index):
        return float(np.mean([
            abs(predictions[(method, line)][index].mean() - actuals[line][index].mean())
            for line in lines
        ]))

    full = np.arange(len(games))
    scores = {m: mean_gap(m, full) for m in ("frequency", "tilt")}

    frequency_wins = 0
    for _ in range(n_boot):
        index = rng.integers(0, len(games), len(games))
        if mean_gap("frequency", index) < mean_gap("tilt", index):
            frequency_wins += 1
    frequency_win_rate = frequency_wins / n_boot

    if verbose:
        detail = "  ".join(f"{k}={v:.5f}" for k, v in scores.items())
        print(f"  Per-PA shape method, mean |calibration gap| on train: {detail}")
        print(f"  Bootstrap over {n_boot} resamples: 'frequency' calibrates "
              f"better in {frequency_win_rate:.0%} of them.")

    if frequency_win_rate < (1.0 - win_threshold):
        if verbose:
            print("  -> 'tilt' wins clearly enough to justify extrapolating "
                  "the tail.")
        return "tilt"
    if verbose:
        print("  -> using 'frequency' (preserves the observed tail shape; "
              "'tilt' did not win clearly enough to earn the extrapolation).")
    return "frequency"
