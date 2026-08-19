"""
Runs scored: the third of the three, and the only one that isn't yours.

WHY THIS FILE EXISTS
--------------------
The prop is HITS + RUNS + RBI. Hits are the batter's. RBI are mostly his
teammates' (V2 established that driving in runs is largely a function of
who was on base, not of the hitter), and features/rbi_features.py models
that. Runs scored are the mirror image: you score because you got on base
AND because the hitters BEHIND you got you home.

So a run decomposes cleanly:

    P(score in a plate appearance)
        = P(home run)                        <- certain, entirely his
        + P(reached base some other way) x q <- needs the rest of the order

Measured on 306,291 official boxscore lines, 25.5% of the runs a hitter
scores are his own home runs. The other 74.5% flow through q, and q is
what this file estimates.

THREE TERMS, EACH MEASURED
--------------------------
q(batter, slot, lineup) = q_league x slot_factor x obp_adjustment x batter_factor

  q_league        0.308 per non-home-run time on base.

  slot_factor     Big and U-shaped: 1.16 at leadoff, bottoming at 0.91 in
                  the cleanup spot, back up to 1.09 at ninth. That shape
                  is not arbitrary -- slots 1 and 9 have the top of the
                  order coming up behind them, slot 4 has the bottom.

  obp_adjustment  How good the three hitters behind him actually are,
                  relative to what that slot normally gets. This is what
                  makes the estimate transfer when the model is applied to
                  a specific lineup rather than an average one.

  batter_factor   How much this hitter beats his slot's expectation --
                  speed, baserunning, and his team's offence. Real: the
                  odd-game/even-game split-half correlation is 0.50, on
                  452 hitters with 200+ times on base.

WHY THE SLOT TERM AND THE OBP TERM ARE BOTH THERE, AND WHAT THAT COSTS
----------------------------------------------------------------------
The obvious move is to drop the slot table and let on-base-behind explain
everything, since across the nine slots the two correlate at r = 0.90. It
doesn't survive contact with the data. Holding slot fixed and identifying
the effect only from team-to-team variation at the SAME lineup position,
the slope collapses to 35% of its pooled value -- still real (t = 5.2),
but three times smaller. Most of the slot pattern is therefore NOT "who
bats behind you"; it is what kind of hitter occupies that slot. Leadoff
hitters are fast.

That leaves an honest limitation worth stating rather than burying: the
slot term is DESCRIPTIVE. Move a slow slugger from cleanup to leadoff and
this will hand him some of the speed that usually lives there. Hitters
rarely move far in the order, so the error is small, but it is a real one
and it is the reason `batter_factor` is estimated as a ratio to his own
slots rather than to the league -- that much, at least, doesn't
double-count.

SHRINKAGE
---------
The raw per-batter factor has s.d. 0.120, of which 0.075 is binomial noise
from a finite number of times on base. Backing that out leaves a true
spread of about 0.094, which corresponds to a beta-binomial prior worth
roughly 260 times on base -- so a hitter needs a full season and a half on
base before his own number outweighs his slot's. Estimated from the data
at run time rather than hard-coded, with these figures as the fallback.
"""
import numpy as np
import pandas as pd

# League rate: runs scored that were NOT the batter's own home run, per
# time on base that was not a home run. Measured over 306,291 boxscore
# lines. Used when there is not enough history to measure it live.
LEAGUE_SCORE_PROB = 0.3084

# Fallback slot factors, relative to league. Measured over 249,275
# confirmed-starter games. Used only when the live data is too thin.
SLOT_FACTOR_FALLBACK = {
    1: 1.161, 2: 1.058, 3: 0.947, 4: 0.914, 5: 0.934,
    6: 0.929, 7: 0.940, 8: 1.016, 9: 1.085,
}

# Effect of the on-base ability of the hitters behind him, IDENTIFIED
# WITHIN SLOT so it is not just the slot pattern wearing a different hat.
# Units: change in q per unit of on-base rate. The pooled estimate is
# +1.05; this is what survives holding the lineup position fixed.
OBP_BEHIND_SLOPE = 0.367

# How many hitters behind him count. Three is roughly how far a runner on
# first can expect to be driven in from before the inning turns over, and
# it matches LOOKBACK_SLOTS in features/rbi_features.py so the two halves
# of the composite see the order the same way.
LOOKAHEAD_SLOTS = 3

# Fallback prior strength, in non-home-run times on base.
SCORE_PRIOR_TOB = 260.0

# Guard rails. q is a probability and the factors multiply, so an
# unluckily-estimated batter in an unusual lineup could otherwise be
# pushed somewhere absurd.
MIN_SCORE_PROB, MAX_SCORE_PROB = 0.10, 0.60
MIN_BATTER_FACTOR, MAX_BATTER_FACTOR = 0.70, 1.35


def add_lineup_obp_behind(frame: pd.DataFrame, batter_obp: pd.Series,
                          league_obp: float) -> pd.DataFrame:
    """
    For each (game, hitter): the mean on-base rate of the LOOKAHEAD_SLOTS
    hitters batting BEHIND him, wrapping around the order.

    The mirror of add_lineup_obp_context in features/rbi_features.py,
    which looks the other way. RBI depend on who is ahead of you; runs
    depend on who is behind you.

    Hitters whose successors aren't in the pool get the league rate for
    those slots -- "we don't know who bats behind him" rather than
    pretending the gap is zero.
    """
    df = frame.copy()
    if "lineup_slot" not in df.columns:
        df["obp_behind"] = league_obp
        return df

    key = "batter" if "batter" in df.columns else "player_id"
    # NOT "_obp": itertuples() renames any column starting with an
    # underscore to a positional placeholder, silently, and the attribute
    # lookup then fails at run time. Learned the hard way in rbi_features.
    df["obp_self"] = df[key].map(batter_obp).fillna(league_obp)

    behind = {}
    for _, game in df.groupby("game_pk", sort=False):
        by_slot = {}
        for row in game.itertuples():
            slot = getattr(row, "lineup_slot", None)
            if pd.notna(slot):
                by_slot[int(slot)] = row.obp_self

        for row in game.itertuples():
            slot = getattr(row, "lineup_slot", None)
            if pd.isna(slot):
                behind[row.Index] = league_obp
                continue
            slot = int(slot)
            # Forward wrap: slot 9 is followed by 1, 2, 3. The order is a
            # circle. ((s + k - 1) % 9) + 1 walks around it.
            following = [((slot + k - 1) % 9) + 1
                         for k in range(1, LOOKAHEAD_SLOTS + 1)]
            behind[row.Index] = float(np.mean(
                [by_slot.get(s, league_obp) for s in following]))

    df["obp_behind"] = [behind.get(i, league_obp) for i in df.index]
    return df.drop(columns=["obp_self"])


class RunScoringModel:
    """
    Estimates P(the batter scores | he reached base and it wasn't a home
    run), for a specific hitter in a specific lineup position.

    Deliberately not a scikit-learn pipeline. Every term is a measured
    ratio with a name and a magnitude that can be printed and argued with,
    which matters more here than squeezing the last decimal out of a fit --
    this quantity is 20% of the composite prop and the whole point of the
    swap from walks to runs was that the composite should mean what it
    says.
    """

    def __init__(self, league_prob=LEAGUE_SCORE_PROB,
                 slot_factors=None, batter_factors=None,
                 slot_obp_baseline=None, league_obp=0.320,
                 prior_tob=SCORE_PRIOR_TOB, n_games=0):
        self.league_prob = float(league_prob)
        self.slot_factors = dict(slot_factors or SLOT_FACTOR_FALLBACK)
        self.batter_factors = batter_factors if batter_factors is not None \
            else pd.Series(dtype=float)
        self.slot_obp_baseline = dict(slot_obp_baseline or {})
        self.league_obp = float(league_obp)
        self.prior_tob = float(prior_tob)
        self.n_games = int(n_games)

    # -- estimation -----------------------------------------------------

    @classmethod
    def fit(cls, game_lines: pd.DataFrame, batter_obp: pd.Series,
            league_obp: float, verbose: bool = True):
        """
        `game_lines` is one row per (batter, game) carrying the OFFICIAL
        boxscore columns -- runs, home_runs, and times on base -- plus
        lineup_slot. data/batting_lines.py provides the first three;
        data/lineup_slots.py provides the last.

        Falls back to the measured league constants, loudly, rather than
        fitting nonsense on twenty rows.
        """
        needed = {"runs_official", "hr_official", "tob_nonhr", "lineup_slot"}
        missing = needed - set(game_lines.columns)
        if missing:
            if verbose:
                print(f"  Runs: missing {sorted(missing)} -- using league "
                      f"constants ({LEAGUE_SCORE_PROB:.4f}/time on base).")
            return cls(league_obp=league_obp)

        df = game_lines.dropna(subset=["lineup_slot"]).copy()
        df["lineup_slot"] = df["lineup_slot"].astype(int)
        df = df[(df["lineup_slot"] >= 1) & (df["lineup_slot"] <= 9)]
        df["extra_runs"] = (df["runs_official"] - df["hr_official"]).clip(lower=0)
        df = df.dropna(subset=["runs_official", "hr_official"])

        # NOTE what is deliberately NOT filtered out here: player-games with
        # zero official times on base. Some of them still contain a run,
        # because the batter reached on an error or a fielder's choice --
        # neither of which is a hit or a walk, so neither appears in the
        # denominator OR in the p_reach the model is given at prediction
        # time. That happens in about 1.5% of player-games.
        #
        # Dropping those rows discards their runs while discarding nothing
        # from the denominator, so q comes out too low and every projected
        # total is short by the same 1.5%. Measured: total predicted runs
        # landed at -1.63% against actual with the filter in, and inside
        # 0.1% with it out.
        #
        # Keeping them means q absorbs the error-reached runs and spreads
        # them across ordinary times on base. That is not literally what
        # happened in any one game, but it puts the right number of runs in
        # the right places on average, which is what a rate is for.

        if len(df) < 2000 or df["tob_nonhr"].sum() <= 0:
            if verbose:
                print(f"  Runs: only {len(df)} usable player-games -- using "
                      f"league constants.")
            return cls(league_obp=league_obp)

        league_prob = float(df["extra_runs"].sum() / df["tob_nonhr"].sum())

        # -- slot factors, with a floor on sample size per slot ---------
        slot_factors = {}
        slot_obp_baseline = {}
        grouped = df.groupby("lineup_slot")
        for slot, group in grouped:
            tob = float(group["tob_nonhr"].sum())
            if tob < 500:
                slot_factors[slot] = SLOT_FACTOR_FALLBACK.get(slot, 1.0)
            else:
                slot_factors[slot] = float(
                    group["extra_runs"].sum() / tob / league_prob)
            if "obp_behind" in group.columns:
                slot_obp_baseline[slot] = float(group["obp_behind"].mean())

        # -- per-batter factor, as a ratio to his OWN slots -------------
        #
        # Estimated against slot expectation rather than against the
        # league. A hitter who has spent his career batting ninth would
        # otherwise look like a great baserunner purely because slot nine
        # scores above average, and the slot term would then count that
        # same fact a second time.
        df["expected"] = df["tob_nonhr"] * df["lineup_slot"].map(
            slot_factors).fillna(1.0) * league_prob

        by_batter = df.groupby("batter").agg(
            actual=("extra_runs", "sum"),
            expected=("expected", "sum"),
            tob=("tob_nonhr", "sum"),
        )
        prior_tob = cls._estimate_prior_tob(by_batter, league_prob)

        # Shrink toward 1.0 with weight proportional to opportunity. The
        # prior is expressed in times on base, so it converts to expected
        # runs by multiplying through by the league rate -- that keeps the
        # units of the numerator and denominator the same.
        prior_runs = prior_tob * league_prob
        factors = ((by_batter["actual"] + prior_runs)
                   / (by_batter["expected"] + prior_runs))
        factors = factors.clip(MIN_BATTER_FACTOR, MAX_BATTER_FACTOR)

        model = cls(league_prob=league_prob, slot_factors=slot_factors,
                    batter_factors=factors, slot_obp_baseline=slot_obp_baseline,
                    league_obp=league_obp, prior_tob=prior_tob,
                    n_games=len(df))

        if verbose:
            settled = int((by_batter["tob"] >= prior_tob).sum())
            print(f"  Runs: P(score | on base, non-HR) = {league_prob:.4f} "
                  f"from {len(df):,} player-games.")
            print(f"    slot factors {min(slot_factors.values()):.3f} "
                  f"(slot {min(slot_factors, key=slot_factors.get)}) to "
                  f"{max(slot_factors.values()):.3f} "
                  f"(slot {max(slot_factors, key=slot_factors.get)}).")
            print(f"    per-batter factor shrunk with a {prior_tob:.0f}-"
                  f"time-on-base prior; {settled} of {len(by_batter)} hitters "
                  f"have more history than that.")
        return model

    @staticmethod
    def _estimate_prior_tob(by_batter: pd.DataFrame, league_prob: float) -> float:
        """
        Method-of-moments prior strength: how much of the spread in the
        observed per-batter factor is real, and how much is just having
        reached base a finite number of times?

        Same logic as estimate_prior_strength in features/rate_features.py,
        applied to a ratio rather than a rate. Returns the fallback when
        the observed spread is entirely explained by noise -- which is the
        correct answer to "how much should I trust one hitter's number?"
        when the answer is "not at all".
        """
        usable = by_batter[(by_batter["tob"] >= 100) & (by_batter["expected"] > 0)]
        if len(usable) < 50:
            return SCORE_PRIOR_TOB

        ratio = usable["actual"] / usable["expected"]
        observed_var = float(ratio.var())
        # Variance of a ratio whose numerator is Binomial(tob, q):
        # Var = q(1-q)/tob, scaled into ratio units by dividing by q^2.
        noise_var = float(
            ((league_prob * (1 - league_prob)) / usable["tob"]).mean()
        ) / league_prob ** 2

        true_var = observed_var - noise_var
        if true_var <= 0:
            return float("inf")   # no real spread: ignore every hitter's own number
        # Beta-binomial: prior strength k satisfies Var = q(1-q)/k in rate
        # units, so in ratio units Var = (1-q)/(k*q).
        return float((1 - league_prob) / (true_var * league_prob))

    # -- prediction -----------------------------------------------------

    def score_prob(self, frame: pd.DataFrame) -> np.ndarray:
        """
        P(score | reached base, not a home run), one value per row.

        Needs `batter` (or `player_id`) and `lineup_slot`; uses
        `obp_behind` when it's there. Anything missing degrades to the
        league constant for that term alone rather than for the whole
        estimate.
        """
        n = len(frame)
        key = "batter" if "batter" in frame.columns else "player_id"

        if "lineup_slot" in frame.columns:
            slots = pd.to_numeric(frame["lineup_slot"], errors="coerce")
            slot_factor = slots.map(self.slot_factors).fillna(1.0).to_numpy()
        else:
            slots = pd.Series([np.nan] * n, index=frame.index)
            slot_factor = np.ones(n)

        # On-base ability behind him, measured as a DEVIATION from what his
        # slot normally has behind it. Using the level instead would double
        # count -- the slot factor already contains the average lineup shape
        # for that position.
        if "obp_behind" in frame.columns and self.slot_obp_baseline:
            baseline = slots.map(self.slot_obp_baseline).fillna(self.league_obp)
            deviation = (pd.to_numeric(frame["obp_behind"], errors="coerce")
                         .fillna(baseline) - baseline).to_numpy()
            obp_term = OBP_BEHIND_SLOPE * deviation / max(self.league_prob, 1e-6)
        else:
            obp_term = np.zeros(n)

        batter_factor = (frame[key].map(self.batter_factors).fillna(1.0).to_numpy()
                         if len(self.batter_factors) else np.ones(n))

        q = self.league_prob * slot_factor * (1.0 + obp_term) * batter_factor
        return np.clip(q, MIN_SCORE_PROB, MAX_SCORE_PROB)

    def runs_per_pa(self, frame: pd.DataFrame, p_hr, p_reach) -> np.ndarray:
        """
        Expected runs scored per plate appearance.

            E[runs] = P(home run) + P(reached some other way) x q

        Note that this responds to the MATCHUP, which a per-batter runs
        constant would not: a hitter facing a pitcher who suppresses his
        on-base rate scores less, automatically, because p_reach falls.
        That is the whole reason for routing runs through reaching base
        instead of estimating runs-per-PA directly.
        """
        p_hr = np.clip(np.asarray(p_hr, dtype=float), 0.0, 1.0)
        p_reach = np.clip(np.asarray(p_reach, dtype=float), 0.0, 1.0)
        p_reach_nonhr = np.clip(p_reach - p_hr, 0.0, 1.0)
        return p_hr + p_reach_nonhr * self.score_prob(frame)


def build_run_training_frame(game_totals: pd.DataFrame,
                             batting_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Join official runs and home runs onto the per-(batter, game) frame and
    derive the non-home-run times on base that q is denominated in.

    Times on base come from the OFFICIAL line (hits + walks + hit-by-pitch)
    rather than from Statcast, so numerator and denominator come from the
    same source. Mixing them would put a definitional mismatch straight
    into the rate: Statcast counts a reach-on-error as reaching base and
    the boxscore does not, and the difference would land in q as if it were
    baserunning skill.
    """
    lines = batting_lines.rename(columns={"player_id": "batter"})
    keep = ["game_pk", "batter", "hits", "runs", "rbi", "walks", "hbp",
            "home_runs", "pa"]
    lines = lines[[c for c in keep if c in lines.columns]].copy()
    # EVERY official column is renamed, including the ones that look safe.
    # game_totals already carries Statcast `hits`, `walks`, `rbi` and `pa`,
    # so any name left alone collides in the merge and pandas silently
    # renames BOTH sides to `_x` and `_y`. The column then isn't missing --
    # it's gone, under a name nothing looks for, and the next line raises a
    # KeyError a long way from the cause. Suffixing here keeps the Statcast
    # and official versions side by side and comparable, which is the point.
    lines = lines.rename(columns={
        "hits": "hits_official", "runs": "runs_official", "rbi": "rbi_official",
        "home_runs": "hr_official", "pa": "pa_official",
        "walks": "walks_official", "hbp": "hbp_official",
    })

    df = game_totals.copy()
    df["game_pk"] = df["game_pk"].astype("int64")
    df["batter"] = df["batter"].astype("int64")
    lines["game_pk"] = lines["game_pk"].astype("int64")
    lines["batter"] = lines["batter"].astype("int64")

    overlap = (set(lines.columns) & set(df.columns)) - {"game_pk", "batter"}
    if overlap:
        raise ValueError(
            f"Official columns would collide with existing ones: "
            f"{sorted(overlap)}. Rename them above rather than letting the "
            f"merge suffix both sides."
        )

    merged = df.merge(lines, on=["game_pk", "batter"], how="left")
    merged["tob_nonhr"] = (
        merged["hits_official"].fillna(0)
        + merged["walks_official"].fillna(0)
        + merged["hbp_official"].fillna(0)
        - merged["hr_official"].fillna(0)
    ).clip(lower=0)
    merged["hrr"] = (merged["hits_official"]
                     + merged["runs_official"]
                     + merged["rbi_official"])
    return merged
