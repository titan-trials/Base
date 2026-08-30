"""
The plate-appearance table -- the foundation the V5 engine sits on.

WHY THE UNIT OF ANALYSIS CHANGED
--------------------------------
Every model in this project up to V4 had ONE ROW PER GAME and asked "did
something happen in this game." That framing quietly threw away most of
the data and most of the signal:

  - 6,667 game rows, versus 30,000 plate-appearance rows from the same
    pull. Four and a half times the sample, already paid for.
  - "Did he homer in this game" is a rare, high-variance summary. "Did he
    homer in this plate appearance" is the actual repeatable event. The
    game outcome is just 4-5 of those events stacked up.
  - Measured on this data, 98.5% of the game-to-game variation in "did he
    homer" is binomial noise from having only ~4 tries. Modelling at the
    game level means fighting that noise directly. Modelling at the PA
    level means estimating the rate and letting arithmetic handle the
    stacking.

So V5 splits the problem in two:

    ESTIMATE   a per-PA probability          <- statistics, this file feeds it
    COMPOUND   over tonight's PA count       <- arithmetic, model/compound.py

Every prop then falls out of the same engine. "Chance of a home run,"
"chance of a walk," and "over 1.5 hits+runs+RBI" are not three problems;
they are three questions asked of one distribution.

WHAT ONE ROW MEANS HERE
-----------------------
One completed plate appearance. Statcast is pitch-by-pitch, and the row
that carries a non-null `events` value is the last pitch of the PA -- the
one where the outcome resolved. Filtering to those gives exactly one row
per PA.

RBI, AND WHY IT'S COMPUTED THIS WAY
-----------------------------------
Statcast has no RBI column. It does have `bat_score` (batting team's runs
before the pitch) and `post_bat_score` (after). The difference on the
resolving pitch is the runs driven in on that plate appearance.

This is very close to official RBI but not identical: it will credit a run
that scored on an error or a wild pitch during the same PA, which the
official scorer would not. Clipped at zero, since the batting team's score
can never go down. Good enough for modelling; worth remembering before
quoting a number as an official stat.

A NOTE ON WHAT ISN'T HERE
-------------------------
Batting-order slot is not in Statcast, and it's the single biggest driver
of how many plate appearances a hitter gets. `pa_per_game` in
model/compound.py is the workaround -- a rolling average that reflects
slot indirectly. Real lineup data would be a genuine upgrade.
"""
import numpy as np
import pandas as pd

from data.game_filter import regular_season_only

HIT_EVENTS = {"single", "double", "triple", "home_run"}
# Hit-by-pitch is grouped with walks throughout: for the props people
# actually bet, "reached base without swinging the bat" is one category,
# and separating them would split an already-thin count.
WALK_EVENTS = {"walk", "hit_by_pitch"}
# Reaching base without a hit or a walk. These matter for RUNS SCORED --
# a runner who reached on an error can still cross the plate -- but not for
# hits or walks, so they get their own set rather than being folded in.
OTHER_REACH_EVENTS = {"field_error", "fielders_choice", "catcher_interf"}
REACH_EVENTS = HIT_EVENTS | WALK_EVENTS | OTHER_REACH_EVENTS

# Bases credited per event, for the total-bases prop. Kept here rather
# than imported from features/bases_features.py so build_pa_table has no
# dependency on a feature module -- the PA table is the foundation
# everything else sits on and should not import upward.
BASE_VALUES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

PA_COLUMNS = [
    "batter", "pitcher", "game_pk", "game_date",
    "is_hr", "is_hit", "is_walk", "is_k", "rbi",
    "reached", "reached_nonhr", "hrr_certain", "total_bases",
    # away_team joins home_team so the PITCHING side of a plate appearance
    # is recoverable. With only home_team, a home batter's opponent is
    # unknowable, and team-level bullpen rates cannot be measured at all.
    "stand", "p_throws", "home_team", "away_team", "is_home", "platoon_edge",
    # Needed to tell a starting pitcher from a reliever. `inning` gives the
    # definition (whoever pitched in the first) and `at_bat_number` gives
    # the ordering. Both were dropped here for several versions, which left
    # data/pitcher_data.py picking "the first row we happen to hold" as the
    # starter -- often a middle reliever.
    "inning", "at_bat_number",
    # Base-out state, carried through for features/rbi_features.py. These
    # hold the RUNNER'S PLAYER ID, or null for an empty base -- not a 0/1
    # flag. Testing for zero instead of null reads every empty base as
    # occupied.
    "on_1b", "on_2b", "on_3b", "outs_when_up",
]


def build_pa_table(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per completed plate appearance, with outcomes and context.

    Sorted by (batter, game_date, game_pk, at_bat_number) so that every
    rolling calculation downstream walks a batter's career in true
    chronological order. Getting this wrong is the quiet way to introduce
    lookahead, so it's done once, here, rather than assumed later.
    """
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Spring training and postseason out, before anything is counted.
    #
    # Roughly 11% of every cached pull is not regular-season baseball, and
    # nothing in this project looked at `game_type` until 2026-08-30. On a
    # season-long average the correction is small -- home run rate moves
    # 0.85% -- but spring is concentrated in February and March, so a
    # trailing 150-PA window in April is substantially made of pitchers
    # building arm strength against minor-league hitters. See
    # data/game_filter.py for the measured numbers.
    # verbose, because a correctness filter that runs silently is a
    # correctness filter nobody believes is running. It drops ~10.5% of a
    # typical pull and that number should be on screen, not inferred.
    df = regular_season_only(df, verbose=True, label="the Statcast pull")
    if df.empty:
        raise ValueError(
            "No regular-season rows in this pull. If the cache genuinely "
            "holds only spring training, widen the date range; if it has "
            "no game_type column at all, regular_season_only would have "
            "passed it through, so this is something else."
        )

    # The resolving pitch of each plate appearance.
    pa = df[df["events"].notna()].copy()
    if pa.empty:
        raise ValueError(
            "No completed plate appearances found (all `events` are null). "
            "Check that the cached Statcast pull isn't truncated."
        )

    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_walk"] = pa["events"].isin(WALK_EVENTS).astype(int)
    pa["is_k"] = (pa["events"] == "strikeout").astype(int)

    if {"post_bat_score", "bat_score"}.issubset(pa.columns):
        pa["rbi"] = (pa["post_bat_score"] - pa["bat_score"]).clip(lower=0)
    else:
        pa["rbi"] = 0

    # ---- The composite prop: HITS + RUNS + RBI ------------------------
    #
    # Two of the three attach cleanly to a plate appearance. The hit is the
    # plate appearance. The RBI happens during it. The RUN does not: the
    # batter reaches base here and crosses the plate two or three hitters
    # later, on somebody else's plate appearance.
    #
    # So a run cannot be read off this row. What CAN be read off it is
    # whether the batter got himself into position to score, and there are
    # exactly two cases:
    #
    #   HOME RUN      -- he scores, certainly, on this plate appearance.
    #                    Roughly a quarter of all runs a hitter scores are
    #                    his own home runs (25.5%, measured on 306k
    #                    boxscore lines), and they are the only runs that
    #                    are deterministic.
    #
    #   REACHED BASE  -- he scores later with some probability. League-wide
    #     ANY OTHER WAY  that probability is 0.308 per time on base, and it
    #                    varies with his lineup slot and with the hitter
    #                    himself. features/run_features.py estimates it.
    #
    # `hrr_certain` therefore holds only the part this row settles: hits,
    # RBI, and the home-run run. The uncertain part is carried separately
    # as `reached_nonhr` and resolved at distribution level, without ever
    # guessing which particular plate appearance a run "belonged" to.
    #
    # A single plate appearance can pile several of these up at once -- a
    # three-run homer is 1 hit + 3 RBI + 1 run = 5 -- which is exactly why
    # the compounding step needs a full DISTRIBUTION over the per-PA
    # contribution rather than a success probability.
    pa["reached"] = pa["events"].isin(REACH_EVENTS).astype(int)
    pa["reached_nonhr"] = (pa["reached"] - pa["is_hr"]).clip(lower=0)
    pa["hrr_certain"] = pa["is_hit"] + pa["rbi"] + pa["is_hr"]

    # ---- TOTAL BASES -------------------------------------------------
    #
    # The easy sibling of the block above. Unlike a run, this settles
    # entirely within the plate appearance: a double is two bases, here,
    # always. No attribution problem and no probability to estimate.
    #
    # A walk is worth ZERO total bases despite putting the batter on
    # first. The stat counts bases gained on HITS only -- which is exactly
    # why total bases and H+R+RBI reward different hitters and trade as
    # separate markets. A patient singles hitter is good at one and poor
    # at the other.
    pa["total_bases"] = pa["events"].map(BASE_VALUES).fillna(0).astype(int)

    # Home/away: the batter's team bats in the bottom half when it's home.
    if "inning_topbot" in pa.columns:
        pa["is_home"] = (pa["inning_topbot"] == "Bot").astype(int)
    else:
        pa["is_home"] = 0

    for col, default in [("stand", "R"), ("p_throws", "R"),
                         ("home_team", "UNK"), ("away_team", "UNK")]:
        if col not in pa.columns:
            pa[col] = default
        pa[col] = pa[col].fillna(default)

    # Platoon edge: a batter facing an opposite-handed pitcher sees the
    # ball longer and hits meaningfully better. One of the few genuinely
    # large, genuinely stable effects in baseball, and it's free here.
    pa["platoon_edge"] = (pa["stand"] != pa["p_throws"]).astype(int)

    sort_cols = [c for c in ["batter", "game_date", "game_pk", "at_bat_number"]
                 if c in pa.columns]
    pa = pa.sort_values(sort_cols).reset_index(drop=True)

    keep = [c for c in PA_COLUMNS if c in pa.columns]
    return pa[keep].copy()


def build_game_totals(pa_table: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the PA table back to one row per game.

    This is NOT what the model trains on -- it's what the model is scored
    against. The engine predicts a distribution from per-PA rates; this is
    the observed reality it gets compared to.
    """
    totals = pa_table.groupby(["batter", "game_pk", "game_date"]).agg(
        pa=("is_hr", "size"),
        hr=("is_hr", "sum"),
        hits=("is_hit", "sum"),
        walks=("is_walk", "sum"),
        strikeouts=("is_k", "sum"),
        rbi=("rbi", "sum"),
        total_bases=("total_bases", "sum"),
        reached=("reached", "sum"),
        reached_nonhr=("reached_nonhr", "sum"),
        hrr_certain=("hrr_certain", "sum"),
        is_home=("is_home", "first"),
        home_team=("home_team", "first"),
    ).reset_index()

    totals["got_hr"] = (totals["hr"] > 0).astype(int)
    totals["got_hit"] = (totals["hits"] > 0).astype(int)
    totals["got_walk"] = (totals["walks"] > 0).astype(int)
    return totals.sort_values(["batter", "game_date"]).reset_index(drop=True)


def per_pa_contribution_distribution(pa_table: pd.DataFrame,
                                     value_col: str = "hrr_certain",
                                     max_value: int = 6,
                                     score_prob: float = None) -> np.ndarray:
    """
    The population-wide distribution of how much ONE plate appearance
    contributes to a counting stat.

    For hits+runs+RBI the answer is mostly 0 (an out), often 1 (a single
    that strands, or an RBI groundout), occasionally 2-5. The compounding
    step needs this whole shape, because "expected 2.2 per game" tells you
    nothing about P(more than 1.5) on its own -- two very different
    distributions can share a mean.

    FOLDING IN THE RUN
    ------------------
    Pass `score_prob` and the run the batter scores LATER is mixed in here
    rather than attributed to a guessed plate appearance. Every row splits
    into two branches:

        reached base, didn't homer -> value     with probability 1 - q
                                   -> value + 1 with probability q
        everything else            -> value     (already exact: an out
                                      contributes nothing, a home run
                                      already counts its own run)

    This is arithmetic on the mixture, not sampling. There is no random
    seed, the answer is identical every run, and the mean comes out right
    by construction: adding q to each of the N non-home-run times on base
    adds exactly q*N runs, which is what q was estimated to mean.

    The alternative -- picking which plate appearances "scored" and
    crediting them a whole run -- needs a random assignment, because
    Statcast records the scorer only as free text in the play description.
    It would put the same total mass in the same places on average while
    adding noise and a seed to argue about.

    Returned as a probability vector indexed 0..max_value.
    """
    values = pa_table[value_col].clip(0, max_value).astype(int).to_numpy()

    if score_prob is None or "reached_nonhr" not in pa_table.columns:
        counts = np.bincount(values, minlength=max_value + 1).astype(float)
        return counts / counts.sum()

    q = float(np.clip(score_prob, 0.0, 1.0))
    on_base = pa_table["reached_nonhr"].to_numpy().astype(bool)

    dist = np.zeros(max_value + 1, dtype=float)
    np.add.at(dist, values[~on_base], 1.0)
    np.add.at(dist, values[on_base], 1.0 - q)
    np.add.at(dist, np.clip(values[on_base] + 1, 0, max_value), q)
    return dist / dist.sum()
