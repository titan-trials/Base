"""
Modelling RBI properly -- the last component of H+RBI+BB that was a
constant.

THE PROBLEM WITH RBI, AND WHY IT WAS LEFT AS A CONSTANT
--------------------------------------------------------
Hits, walks and strikeouts are things a hitter does. A run batted in is
something a hitter does WITH HELP: you cannot drive in runners who aren't
there. V2 dropped RBI as a target entirely after AUC never moved off 0.50,
and V5 fell back to using each batter's flat historical RBI-per-plate-
appearance rate.

That constant is the weakest link in the H+RBI+BB prop. Hits and walks are
modelled per plate appearance against tonight's actual pitcher; RBI was a
season average that never moved.

THE STRUCTURE (the user's idea, made precise)
---------------------------------------------
"Take the on-base chance of the hitters ahead of him" is exactly right,
and it factors into two estimable pieces:

    RBI per PA  =  how many runners are on when he bats
                   x  how well he drives them in

Both halves are learnable, and neither is learnable as one lump:

  STAGE 1   E[runners on base | lineup slot, on-base skill of the hitters
            batting ahead of him]

            The lineup wraps around -- the leadoff hitter bats after the
            8th and 9th hitters in every inning except the first, so "the
            three ahead of him" means slots 7, 8, 9. Treating the order as
            a line rather than a circle would systematically understate
            the top of the lineup.

  STAGE 2   E[RBI | runners on, outs, this batter's power]

            A home run drives in everyone; a single scores a runner from
            second. So the batter's extra-base ability multiplies whatever
            stage 1 supplies, and it interacts: power matters far more
            with runners on than with the bases empty.

Two small regressions chained, rather than one model asked to learn a
product it has no way to express.

WHY NOT USE THE ACTUAL BASE STATE
---------------------------------
Statcast carries `on_1b`, `on_2b`, `on_3b` and `outs_when_up` on every
plate appearance, so the true base-out state is known for all ~500,000
historical rows. Stage 2 uses it directly, which is why it's the easy half.

Stage 1 exists because at prediction time the base state is exactly what
you don't know -- it's the future. Hence projecting it from the lineup,
which IS knowable before first pitch.

HONEST LIMITS
-------------
- Runners are erased by double plays and by innings ending. The
  projection learns this implicitly from history rather than modelling
  base-running, so it's a fitted relationship, not a simulation.
- The on-base skill of hitters ahead is only as good as pool coverage. A
  hitter batting behind two players not in the pool gets league-average
  stand-ins for them, which flattens his estimate toward the middle.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# How many hitters ahead in the order actually matter for who's on base.
# Three is roughly one time through the front of an inning; beyond that,
# outs have usually cleared the bases.
LOOKBACK_SLOTS = 3

RUNNER_FEATURES = ["lineup_slot", "obp_ahead", "is_home"]
RBI_FEATURES = [
    "runners_on", "runners_in_scoring", "outs_when_up",
    "bat_is_hr_career", "bat_is_hit_career",
]


def add_base_state(pa_table: pd.DataFrame) -> pd.DataFrame:
    """
    Turn Statcast's runner-id columns into base-state features.

    `on_1b` / `on_2b` / `on_3b` hold the MLBAM id of the runner on that
    base, or null for empty. So "is there a runner" is simply "is this
    field populated" -- an easy thing to get subtly wrong by testing for
    zero instead of null.
    """
    df = pa_table.copy()

    for base in ("on_1b", "on_2b", "on_3b"):
        df[f"runner_{base[-2:]}"] = (
            df[base].notna().astype(int) if base in df.columns else 0
        )

    df["runners_on"] = df["runner_1b"] + df["runner_2b"] + df["runner_3b"]
    # Second and third are "scoring position": a single plausibly scores
    # them, a runner on first usually needs an extra-base hit.
    df["runners_in_scoring"] = df["runner_2b"] + df["runner_3b"]

    if "outs_when_up" not in df.columns:
        df["outs_when_up"] = 1  # neutral if the column is absent
    df["outs_when_up"] = pd.to_numeric(df["outs_when_up"], errors="coerce").fillna(1)

    return df


def add_lineup_obp_context(game_totals: pd.DataFrame,
                           batter_obp: pd.Series,
                           league_obp: float) -> pd.DataFrame:
    """
    For each (game, hitter): the mean on-base rate of the LOOKBACK_SLOTS
    hitters batting ahead of him, wrapping around the order.

    Requires `lineup_slot` on the frame. Hitters whose predecessors aren't
    in the pool get the league rate for those slots, which is the honest
    thing to do -- it says "we don't know who bats in front of him" rather
    than pretending the gap is zero.
    """
    df = game_totals.copy()
    if "lineup_slot" not in df.columns:
        df["obp_ahead"] = league_obp
        return df

    # NOT "_obp": pandas' itertuples() renames any column starting with an
    # underscore to a positional placeholder, because attribute names can't
    # begin with one. The rename is silent and the attribute lookup then
    # fails at runtime.
    df["obp_self"] = df["batter"].map(batter_obp).fillna(league_obp)
    ahead_values = []

    for _, game in df.groupby("game_pk", sort=False):
        by_slot = {}
        for row in game.itertuples():
            slot = row.lineup_slot
            if pd.notna(slot):
                by_slot[int(slot)] = row.obp_self

        for row in game.itertuples():
            slot = row.lineup_slot
            if pd.isna(slot):
                ahead_values.append((row.Index, league_obp))
                continue
            slot = int(slot)
            # Wrap: slot 1 is preceded by 9, 8, 7 -- the order is a circle,
            # not a line. ((s - k - 1) % 9) + 1 walks backwards through it.
            preceding = [((slot - k - 1) % 9) + 1 for k in range(1, LOOKBACK_SLOTS + 1)]
            values = [by_slot.get(s, league_obp) for s in preceding]
            ahead_values.append((row.Index, float(np.mean(values))))

    lookup = dict(ahead_values)
    df["obp_ahead"] = [lookup.get(i, league_obp) for i in df.index]
    return df.drop(columns=["obp_self"])


def train_runner_model(game_pa: pd.DataFrame):
    """
    STAGE 1: expected runners on base, from lineup position and the
    on-base skill of the hitters ahead.

    Ridge rather than plain least squares because `lineup_slot` and
    `obp_ahead` are correlated (good hitters bat in the middle, and the
    middle bats behind the good top of the order), and ridge keeps that
    collinearity from producing two large offsetting coefficients that
    swing wildly on new data.
    """
    frame = game_pa.dropna(subset=RUNNER_FEATURES + ["runners_on"])
    if len(frame) < 500:
        return None
    model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    model.fit(frame[RUNNER_FEATURES], frame["runners_on"])
    return model


def train_rbi_model(pa_with_state: pd.DataFrame):
    """
    STAGE 2: expected RBI in a plate appearance, given who's on base, how
    many outs, and how much power the batter has.

    Trained on the ACTUAL base state, which is known for every historical
    plate appearance -- this is the half with real signal and real sample
    size.
    """
    frame = pa_with_state.dropna(subset=RBI_FEATURES + ["rbi"])
    if len(frame) < 1000:
        return None
    model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    model.fit(frame[RBI_FEATURES], frame["rbi"])
    return model


def predict_rbi_per_pa(runner_model, rbi_model, frame: pd.DataFrame,
                       league_rbi: float) -> np.ndarray:
    """
    Chain the two stages: project runners on base, then expected RBI.

    Falls back to the league rate whenever either model is missing, so a
    caller without lineup data still gets a number rather than a crash --
    the same behaviour as before this file existed.
    """
    if runner_model is None or rbi_model is None:
        return np.full(len(frame), league_rbi)

    work = frame.copy()
    missing = [c for c in RUNNER_FEATURES if c not in work.columns]
    if missing:
        return np.full(len(frame), league_rbi)

    work["runners_on"] = np.clip(
        runner_model.predict(work[RUNNER_FEATURES].fillna(0)), 0.0, 3.0
    )
    # Scoring position historically runs a bit under half of all runners;
    # splitting the projection this way keeps stage 2's two runner inputs
    # consistent with each other instead of leaving one at zero.
    work["runners_in_scoring"] = work["runners_on"] * 0.45
    if "outs_when_up" not in work.columns:
        work["outs_when_up"] = 1.0

    for col in ("bat_is_hr_career", "bat_is_hit_career"):
        if col not in work.columns:
            work[col] = 0.0

    predicted = rbi_model.predict(work[RBI_FEATURES].fillna(0))
    # RBI per plate appearance cannot be negative, and the ceiling is a
    # grand slam. Clipping stops a linear model extrapolating nonsense for
    # an extreme feature combination.
    return np.clip(predicted, 0.0, 4.0)


def runner_summary(game_pa: pd.DataFrame) -> pd.DataFrame:
    """Average runners on base by lineup slot -- the effect stage 1
    exploits, printed plainly enough to check by eye."""
    if "lineup_slot" not in game_pa.columns:
        return pd.DataFrame()
    return game_pa.groupby("lineup_slot").agg(
        plate_appearances=("runners_on", "size"),
        mean_runners_on=("runners_on", "mean"),
        mean_in_scoring=("runners_in_scoring", "mean"),
        mean_rbi=("rbi", "mean"),
    ).reset_index()
