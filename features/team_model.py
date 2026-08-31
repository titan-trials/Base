"""
Team runs and win probability, built up from the hitters.

THE DECOMPOSITION
-----------------
Every run is scored by exactly one batter. features/run_features.py
already estimates, for each hitter, the probability he scores in a given
plate appearance, and features/pa_projection.py estimates how many plate
appearances he gets. Multiply and sum over a lineup and you have that
team's expected runs -- no new model, just the existing one added up.

Checked against reality: 9 hitters x ~4.0 plate appearances x 0.1158 runs
per plate appearance is 4.17 runs, against a league average near 4.4.
That is the right neighbourhood, which is the point of doing it this way
rather than fitting a separate team model that could disagree with the
hitter model about the same game.

WHY THAT IS NOT ENOUGH FOR A WIN PROBABILITY
---------------------------------------------
A mean is not a distribution. Two teams projected 5.2 and 4.1 do not
produce "the first team wins" -- they produce a spread of outcomes where
the second team wins a great deal of the time. The whole question is how
wide that spread is.

Team runs are NOT Poisson. Poisson would put the variance equal to the
mean, around 4.4. Real team-game run totals have a variance closer to
double that, because innings are not independent trials: a team that gets
runners on tends to keep getting them in the same inning, and a big
inning is a fat right tail that Poisson cannot produce.

So the shape is negative binomial, fitted by method of moments to actual
final scores, and the dispersion is measured rather than assumed:

    r = mean^2 / (variance - mean)

To project a specific team, the DISPERSION is held fixed and only the
mean moves. That says "teams differ in how many runs they score, not in
how lumpy their scoring is", which is close enough to true and much
better supported than fitting a shape per team from a handful of games.

TIES
----
Baseball has no ties, so P(win) is not P(more runs) -- it is
P(more runs) plus a share of the games that reach extra innings. That
share is measured from real games where possible, and defaults to a
coin flip. Extra innings are close to 50/50 with a small home edge from
batting last.
"""
import numpy as np
import pandas as pd

# Runs beyond this are vanishingly rare and only cost compute -- the
# recurrence below is O(MAX_RUNS), so this is nearly free.
#
# Generous on purpose. Truncating the tail and renormalising pulls the
# mean DOWN, and the higher the projection the worse it gets: at 30 the
# error was invisible at 7 runs and half a run at 12. Baseball does not
# project 12, but a cap tuned to the cases you happen to test is a cap
# that fails on the case you did not.
MAX_RUNS = 60

# Fallbacks if there are no cached final scores yet. Roughly league
# average; the printout says loudly when these are being used, because a
# league-average shape applied to every game is a much weaker model than
# it looks.
FALLBACK_MEAN = 4.40
FALLBACK_VAR = 9.60


class TeamRunModel:
    """Negative binomial run distribution, fitted to real final scores."""

    def __init__(self, dispersion_r, league_mean, home_extra_win=0.5,
                 fitted_on=0):
        self.r = float(dispersion_r)
        self.league_mean = float(league_mean)
        self.home_extra_win = float(home_extra_win)
        self.fitted_on = int(fitted_on)

    @classmethod
    def fit(cls, team_lines: pd.DataFrame = None, verbose: bool = True):
        if team_lines is None or team_lines.empty or "runs" not in team_lines:
            if verbose:
                print("  Team runs: NO cached final scores -- using a "
                      "league-average shape. Run "
                      "`python -m data.team_lines` to fit it properly.")
            mean, var, n = FALLBACK_MEAN, FALLBACK_VAR, 0
            home_extra = 0.5
        else:
            runs = pd.to_numeric(team_lines["runs"], errors="coerce").dropna()
            mean, var, n = float(runs.mean()), float(runs.var()), len(runs)
            # Home teams win extra-inning games slightly more often for
            # the same reason they win overall -- batting last. Measured
            # from games decided by one run as a proxy, which is a
            # reasonable stand-in and better than assuming exactly half.
            close = team_lines[
                (team_lines["runs"] - team_lines["runs_allowed"]).abs() == 1]
            if len(close) > 200 and "is_home" in close:
                won = close["runs"] > close["runs_allowed"]
                home_extra = float(won[close["is_home"] == 1].mean())
                home_extra = float(np.clip(home_extra, 0.45, 0.58))
            else:
                home_extra = 0.5

        if var <= mean:
            # Under-dispersed relative to Poisson should not happen for
            # runs; if it does, the sample is tiny or something is wrong.
            # A very large r makes this behave as Poisson rather than
            # producing a negative parameter.
            dispersion = 1e6
        else:
            dispersion = mean * mean / (var - mean)

        if verbose:
            source = (f"{n:,} team-games" if n else "league-average fallback")
            print(f"  Team runs: mean {mean:.3f}, variance {var:.3f} "
                  f"(var/mean {var / mean:.2f}) from {source}.")
            print(f"    Negative binomial dispersion r = {dispersion:.2f}. "
                  f"Poisson would be var/mean = 1.00.")
            print(f"    Home team takes {home_extra:.1%} of games decided "
                  f"by one run.")
        return cls(dispersion, mean, home_extra, n)

    def pmf(self, expected_runs: float) -> np.ndarray:
        """
        Probability of each run total, for a team projected to score
        `expected_runs`.

        Dispersion held fixed, mean shifted -- see the module docstring.
        """
        mean = float(max(expected_runs, 0.05))
        r = self.r
        # NB parameterised by mean: p = r / (r + mean)
        p = r / (r + mean)

        # Built by recurrence rather than with scipy's gammaln:
        #
        #     pmf(0) = p^r
        #     pmf(k) = pmf(k-1) * (k - 1 + r) / k * (1 - p)
        #
        # Exact, and it keeps this file to numpy and pandas. That matters
        # because a function called once per team per game should not be
        # importing scipy on every call, and because the fewer things this
        # module needs the more places it can be used.
        out = np.empty(MAX_RUNS + 1, dtype=float)
        out[0] = p ** r
        for k in range(1, MAX_RUNS + 1):
            out[k] = out[k - 1] * (k - 1 + r) / k * (1.0 - p)
        total = out.sum()
        return out / total if total > 0 else out

    def win_probability(self, home_runs: float, away_runs: float) -> tuple:
        """
        (P home wins, P away wins), accounting for extra innings.

        Returns a pair that sums to 1. Baseball has no ties, so the
        probability of equal run totals is split by how often the home
        team wins from there rather than being left as a third outcome.
        """
        home = self.pmf(home_runs)
        away = self.pmf(away_runs)
        # P(home > away) by summing the joint over the upper triangle.
        away_cum = np.cumsum(away)
        p_home_more = float((home[1:] * away_cum[:-1]).sum())
        p_tie = float((home * away).sum())
        p_home = p_home_more + p_tie * self.home_extra_win
        return float(np.clip(p_home, 0.0, 1.0)), float(np.clip(1 - p_home, 0.0, 1.0))


def team_expected_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the hitter model up to a team total.

    `frame` is predict_slate's per-hitter frame and needs p_run (runs per
    plate appearance) and expected_pa. Every run has exactly one scorer,
    so summing over a lineup double-counts nothing.
    """
    needed = {"p_run", "expected_pa", "team", "game_pk"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    work["exp_runs"] = work["p_run"] * work["expected_pa"]
    out = (work.groupby(["game_pk", "team"], as_index=False)
               .agg(expected_runs=("exp_runs", "sum"),
                    hitters=("exp_runs", "size"),
                    is_home=("is_home", "first"),
                    opponent=("opponent", "first")))
    return out


def build_game_predictions(frame: pd.DataFrame, model: TeamRunModel,
                           verbose: bool = True) -> pd.DataFrame:
    """
    One row per game: projected score and win probability.

    A lineup the model could not fill completely is still projected, but
    `hitters` is carried through so eight-of-nine is visibly different
    from nine-of-nine rather than quietly lower-scoring.
    """
    totals = team_expected_runs(frame)
    if totals.empty:
        return pd.DataFrame()

    rows = []
    for game_pk, side in totals.groupby("game_pk"):
        if len(side) != 2:
            continue
        home = side[side["is_home"] == 1]
        away = side[side["is_home"] == 0]
        if home.empty or away.empty:
            continue
        home, away = home.iloc[0], away.iloc[0]
        p_home, p_away = model.win_probability(
            home["expected_runs"], away["expected_runs"])
        rows.append({
            "game_pk": int(game_pk),
            "home_team": home["team"], "away_team": away["team"],
            "home_runs": float(home["expected_runs"]),
            "away_runs": float(away["expected_runs"]),
            "total_runs": float(home["expected_runs"] + away["expected_runs"]),
            "home_win_prob": p_home, "away_win_prob": p_away,
            "home_hitters": int(home["hitters"]),
            "away_hitters": int(away["hitters"]),
        })
    out = pd.DataFrame(rows)
    if not out.empty and verbose:
        print(f"  {len(out)} games. Projected {out['total_runs'].mean():.2f} "
              f"runs per game, home teams average "
              f"{out['home_win_prob'].mean():.1%} to win.")
        thin = out[(out["home_hitters"] < 9) | (out["away_hitters"] < 9)]
        if len(thin):
            print(f"    {len(thin)} game(s) missing hitters from a lineup -- "
                  f"those totals run low by roughly the missing share.")
    return out


def top_contributors(frame: pd.DataFrame, game_pk: int, top_n: int = 3):
    """Who the model expects to do the scoring, for one game."""
    work = frame[frame["game_pk"] == game_pk].copy()
    if work.empty or "p_run" not in work:
        return pd.DataFrame()
    work["exp_runs"] = work["p_run"] * work["expected_pa"]
    cols = [c for c in ("name", "team", "lineup_slot", "exp_runs",
                        "expected_pa", "prob_hrr_over_0.5") if c in work.columns]
    return work.nlargest(top_n, "exp_runs")[cols]
