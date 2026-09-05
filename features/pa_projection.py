"""
Projecting how many plate appearances a hitter gets tonight.

WHY THIS IS THE HIGHEST-VALUE PIECE IN THE PROJECT
---------------------------------------------------
The V5 leverage diagnostic measured what knowing the exact plate-appearance
count would be worth:

    gets a hit    +0.0026  ->  +0.0485   (worth +0.0459)
    gets a walk   +0.0272  ->  +0.0540   (worth +0.0268)
    gets a homer  +0.0016  ->  +0.0172   (worth +0.0156)

For comparison, every batter-skill feature built across V1-V5 combined is
worth roughly +0.004. Exposure is not a detail around the edge of the
model; it is the largest single term, and it had never been modelled.

The reason it went unnoticed for so long is instructive: every model
silently conditions on exposure. "Chance he homers tonight" always meant
"chance he homers in however many chances he happens to get," and that
second half was never examined.

WHAT'S KNOWABLE AND WHAT ISN'T
------------------------------
Not all of that +0.046 is reachable. Plate appearances depend on:

  KNOWABLE before first pitch
    lineup slot        the dominant term -- a leadoff hitter is
                       guaranteed a first-inning turn and comes up every
                       ninth batter thereafter
    home or away       a home team leading after eight innings does not
                       bat in the ninth, costing roughly a third of a
                       plate appearance on average
    team quality       more baserunners means more turns through the
                       order for everyone behind them

  NOT knowable
    game length        extra innings add turns
    blowouts           a manager rests starters in a 12-2 game
    in-game removal    injury, ejection, pinch-hitting

So the honest target is to capture the knowable part. The diagnostic in
train_props_v5.py reports both the model's projection and the
actual-PA upper bound, so the gap between them stays visible instead of
being quietly claimed as progress.

WHY A DISTRIBUTION, NOT A NUMBER
--------------------------------
model/compound.py already marginalises over a plate-appearance
distribution, because P(at least one home run) is concave in PA -- plugging
in an average overstates it. So this model predicts P(PA = 3), P(PA = 4),
P(PA = 5)... rather than an expected value, using multinomial logistic
regression. The output drops straight into `marginalise_over_pa`.

Multinomial rather than Poisson because the support here is tiny and
bounded (essentially 3 to 6) and the shape is not Poisson at all -- it's
sharply peaked with a hard floor. A distribution over a handful of
categories is both simpler and more honest than forcing a parametric form
that doesn't fit.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Plate appearances outside this range are essentially always a player
# leaving early or a very long extra-inning game -- neither is predictable,
# and both would distort the model. Clipped rather than dropped, so the
# probability mass stays where it belongs.
MIN_PA, MAX_PA = 1, 6

PA_FEATURES = [
    "lineup_slot", "lineup_slot_sq", "is_home",
    "batter_pa_mean_20", "team_pa_mean_20",
]


def add_pa_history_features(game_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling context for the plate-appearance model.

    `batter_pa_mean_20` is a rolling average of the hitter's own recent PA
    per game -- it captures lineup slot indirectly, and stands in
    completely on games where the real slot is missing.

    `team_pa_mean_20` is the average PA of ALL hitters in the pool who
    played that team's games recently, which proxies team offensive
    quality: a lineup that gets on base turns over more often, giving
    everyone more chances. Built from the pool rather than from full team
    data, so it's only as good as the pool's coverage of that team.

    Both shifted by one game, same lookahead rule as everywhere else.
    """
    df = game_totals.sort_values(["batter", "game_date"]).copy()

    df["batter_pa_mean_20"] = df.groupby("batter", sort=False)["pa"].transform(
        lambda s: s.rolling(20, min_periods=3).mean().shift(1)
    )

    if "home_team" in df.columns:
        by_team = df.sort_values("game_date").groupby("home_team", sort=False)["pa"]
        df["team_pa_mean_20"] = by_team.transform(
            lambda s: s.rolling(50, min_periods=5).mean().shift(1)
        )
    else:
        df["team_pa_mean_20"] = df["batter_pa_mean_20"]

    return df


def prepare_pa_training_frame(game_totals_with_slots: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and shape the frame the PA model trains on.

    Restricted to STARTERS. A pinch-hitter's one plate appearance is real
    but unpredictable before the game -- including those rows would teach
    the model that the 5-slot sometimes yields a single PA, which is true
    and useless, since you only ever project PA for players you know are
    starting.
    """
    df = add_pa_history_features(game_totals_with_slots)

    if "is_starter" in df.columns:
        df = df[df["is_starter"] == 1]
    if "lineup_slot" in df.columns:
        df = df[df["lineup_slot"].between(1, 9)]

    # `team_pa_mean_20` is NaN early in a team's history, before its
    # 50-game window fills. Fall back to the batter's own average rather
    # than dropping the row -- the team term is a mild refinement, and
    # losing an otherwise complete row over it is a bad trade.
    if "team_pa_mean_20" in df.columns:
        df["team_pa_mean_20"] = df["team_pa_mean_20"].fillna(df["batter_pa_mean_20"])

    # Slot's effect on PA is not linear -- the gap between batting 1st and
    # 2nd is smaller than between 8th and 9th, because the 9-slot is the one
    # that most often loses a turn. A squared term lets a linear model bend
    # to that. Built before the dropna so it can be checked for NaN too.
    df["lineup_slot_sq"] = df["lineup_slot"] ** 2
    df = df.dropna(subset=[c for c in PA_FEATURES if c in df.columns]
                          + ["lineup_slot", "batter_pa_mean_20"]).copy()
    df["pa_clipped"] = df["pa"].clip(MIN_PA, MAX_PA).astype(int)
    return df


def train_pa_model(train_df: pd.DataFrame, feature_cols=PA_FEATURES):
    """Multinomial logistic regression over clipped PA counts."""
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=1.0)),
    ])
    model.fit(train_df[feature_cols], train_df["pa_clipped"])
    return model


def predict_pa_distributions(model, frame: pd.DataFrame,
                             feature_cols=PA_FEATURES) -> list:
    """
    A {pa_count: probability} dict per row, ready for
    model.compound.marginalise_over_pa.
    """
    probabilities = model.predict_proba(frame[feature_cols])
    classes = model.named_steps["clf"].classes_
    return [
        {int(c): float(p) for c, p in zip(classes, row)}
        for row in probabilities
    ]


# ---------------------------------------------------------------------
# The empirical table -- what replaced the regression on 2026-09-05
# ---------------------------------------------------------------------
#
# The multinomial regression above had two problems that PAVA could not
# reach:
#
#   1. TRAIN/SERVE SKEW. At fit time `batter_pa_mean_20` was a 20-game
#      rolling mean and `team_pa_mean_20` a 50-row rolling mean grouped by
#      home_team (the VENUE's team -- for an away batter, the opponent's).
#      At prediction time predict_slate fed the hitter's all-time mean for
#      both. Two of five features meant different things on the two sides
#      of the model.
#
#   2. THE PARABOLA. `lineup_slot_sq` let the fitted curve over-separate
#      the order: 4.55 PA at leadoff against a measured 4.29, a spread of
#      1.14 against 0.94. PAVA fixed the ORDERING (V8c); the level stayed
#      wrong, and V8c said as much.
#
# There is no regression to get wrong here. P(PA = k | slot, home/away) is
# read straight off the history: eighteen cells, thousands of player-games
# in each, and the answer is a histogram. It is monotone in slot by
# construction whenever the cells are full, it has no serving path that
# can drift from its fitting path, and a hitter with an unknown slot falls
# back to the same table marginalised over slot.
#
# What it deliberately does NOT do is condition on team quality. The old
# `team_pa_mean_20` was meant to, and was noise. If a team term earns its
# place later, the right way in is to tilt a cell's distribution to a
# team-level mean (see _tilt_distribution_to_mean), not to fit harder.
PA_TABLE_SINCE = "2024-01-01"     # current era; PA/game drifts slowly
PA_TABLE_MIN_CELL = 300           # below this, a cell borrows its slot's pooled shape


class EmpiricalPAModel:
    """
    P(PA = k | lineup slot, home/away), counted from starters' games.
    """

    def __init__(self, cells: dict, by_slot: dict, pooled: dict, n_rows: int):
        self.cells = cells        # {(slot, is_home): {k: p}}
        self.by_slot = by_slot    # {slot: {k: p}}
        self.pooled = pooled      # {k: p}
        self.n_rows = n_rows

    @staticmethod
    def _hist(values: pd.Series) -> dict:
        counts = values.clip(MIN_PA, MAX_PA).astype(int).value_counts()
        total = float(counts.sum())
        return {int(k): float(v) / total for k, v in counts.sort_index().items()}

    @classmethod
    def fit(cls, game_totals: pd.DataFrame, since: str = PA_TABLE_SINCE,
            min_cell: int = PA_TABLE_MIN_CELL, verbose: bool = True):
        df = game_totals
        if "lineup_slot" not in df.columns:
            raise ValueError("game_totals has no lineup_slot column")
        if "is_starter" in df.columns:
            df = df[df["is_starter"] == 1]
        df = df[pd.to_numeric(df["lineup_slot"], errors="coerce").between(1, 9)]
        if since is not None and "game_date" in df.columns:
            recent = df[pd.to_datetime(df["game_date"]) >= pd.Timestamp(since)]
            # A pool rebuilt from a short cache may have too little recent
            # history; fall back to everything rather than to nothing.
            if len(recent) >= 18 * min_cell:
                df = recent
        if df.empty:
            raise ValueError("no starter games with a lineup slot")
        df = df.copy()
        df["lineup_slot"] = df["lineup_slot"].astype(int)
        df["is_home"] = df["is_home"].fillna(0).astype(int) if "is_home" in df.columns else 0

        pooled = cls._hist(df["pa"])
        by_slot = {int(s): cls._hist(g["pa"]) for s, g in df.groupby("lineup_slot")}
        cells = {}
        for (slot, home), g in df.groupby(["lineup_slot", "is_home"]):
            if len(g) >= min_cell:
                cells[(int(slot), int(home))] = cls._hist(g["pa"])

        model = cls(cells, by_slot, pooled, len(df))
        if verbose:
            means = model.slot_means()
            print(f"  PA table: {len(df):,} starter games since "
                  f"{pd.to_datetime(df['game_date']).min().date() if 'game_date' in df.columns else '?'}, "
                  f"{len(cells)} of 18 slot x home cells filled.")
            print("    expected PA by slot: " + "  ".join(
                f"{s}:{means[s]:.2f}" for s in sorted(means)))
        return model

    def distribution(self, slot, is_home) -> dict:
        try:
            slot = int(slot)
        except (TypeError, ValueError):
            return dict(self.pooled)
        try:
            home = int(is_home)
        except (TypeError, ValueError):
            home = None
        if home is not None and (slot, home) in self.cells:
            return dict(self.cells[(slot, home)])
        return dict(self.by_slot.get(slot, self.pooled))

    def slot_means(self) -> dict:
        return {s: sum(k * v for k, v in d.items()) for s, d in self.by_slot.items()}

    def predict(self, frame: pd.DataFrame) -> list:
        slots = pd.to_numeric(frame["lineup_slot"], errors="coerce")
        homes = frame["is_home"] if "is_home" in frame.columns else pd.Series(np.nan, index=frame.index)
        return [self.distribution(s, h) if pd.notna(s) else dict(self.pooled)
                for s, h in zip(slots, homes)]


def evaluate_pa_model(model, test_df: pd.DataFrame,
                      feature_cols=PA_FEATURES) -> dict:
    """
    How good is the PA projection, in units that mean something?

    Reported as mean absolute error in plate appearances against two
    baselines -- the pool average, and the hitter's own recent average.
    Beating the second one is the real test: "he usually gets 4.3" is
    already a decent guess, and lineup slot has to add something beyond it.
    """
    distributions = predict_pa_distributions(model, test_df, feature_cols)
    expected = np.array([
        sum(k * v for k, v in d.items()) for d in distributions
    ])
    actual = test_df["pa_clipped"].to_numpy()

    pool_mean = float(test_df["pa_clipped"].mean())
    own_mean = test_df["batter_pa_mean_20"].clip(MIN_PA, MAX_PA).to_numpy()

    return {
        "n": int(len(test_df)),
        "mae_model": float(np.abs(expected - actual).mean()),
        "mae_pool_mean": float(np.abs(pool_mean - actual).mean()),
        "mae_own_recent": float(np.abs(own_mean - actual).mean()),
        "corr": float(np.corrcoef(expected, actual)[0, 1]),
    }


def print_pa_evaluation(result: dict):
    print("\n--- Plate-appearance projection ---")
    print(f"  Test games                       : {result['n']:,}")
    print(f"  Mean abs error, this model       : {result['mae_model']:.4f} PA")
    print(f"  Mean abs error, pool average     : {result['mae_pool_mean']:.4f} PA")
    print(f"  Mean abs error, own recent avg   : {result['mae_own_recent']:.4f} PA")
    print(f"  Correlation with actual PA       : {result['corr']:+.4f}")
    gain = result["mae_own_recent"] - result["mae_model"]
    if gain > 0.005:
        print(f"  -> lineup slot beats 'his usual' by {gain:.4f} PA per game.")
    else:
        print("  -> lineup slot adds little over the hitter's own recent")
        print("     average, which already encodes his usual slot.")


def slot_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Average PA by lineup slot -- the effect this whole file exists for,
    stated plainly enough to sanity-check by eye."""
    return frame.groupby("lineup_slot").agg(
        games=("pa", "size"),
        mean_pa=("pa", "mean"),
        pct_5plus_pa=("pa", lambda s: (s >= 5).mean()),
    ).reset_index()


# ---------------------------------------------------------------------
# The constraint the model cannot learn
# ---------------------------------------------------------------------
#
# A batting order is a SEQUENCE, not nine independent players. Slot 1 bats
# ahead of slot 2 in every inning, so across a game he gets at least as
# many plate appearances. The gap can be zero. It cannot be negative.
#
# predict_pa_distributions() scores each hitter on his own feature row and
# has no way to know that. Worse, PA_FEATURES carries `lineup_slot_sq`, so
# the fitted curve is a PARABOLA in slot -- and a parabola turns over. On
# the 2026-08-18 slate, with all 15 lineups confirmed, the vertex landed
# near the top of the order and the model projected:
#
#     leadoff 4.270 PA  <  second 4.341 PA        (impossible)
#     reality 4.294 PA  >  second 4.206 PA        (13,853 games)
#
# The quadratic is not the villain and removing it is not the fix: the
# real decline genuinely accelerates down the order (-0.088 PA per slot at
# the top, -0.15 at the bottom), which is curvature a linear term cannot
# represent. The problem is that the curvature is UNCONSTRAINED.
#
# So the shape is kept and the constraint is imposed afterwards, by
# projecting each team's nine expected values onto the nearest
# non-increasing sequence. That projection is exactly isotonic regression,
# it is the minimum change that satisfies the constraint, and it leaves an
# already-monotone lineup untouched.
#
# Same species of fix as forcing project_lineup() to emit a permutation of
# 1-9: a fact obvious to a person, invisible to a regression, and cheap to
# assert once you notice it is missing.
MONOTONICITY_MIN_HITTERS = 4


def _tilt_distribution_to_mean(dist: dict, target: float) -> dict:
    """
    Reshape a {pa_count: probability} dict so its mean becomes `target`,
    changing the shape as little as possible.

    Exponential tilting: every count's probability is multiplied by
    exp(theta * count) and renormalised, with theta solved by bisection.
    This is the maximum-entropy way to hit a mean -- of all distributions
    with the required mean, it is the closest to the one we started with,
    so the model's opinion about SPREAD survives while its opinion about
    LEVEL is corrected.

    Rescaling the counts instead would move probability mass onto plate
    appearance totals the model never predicted, and clipping would pile
    mass on an endpoint.
    """
    counts = np.array(sorted(dist), dtype=float)
    probs = np.array([dist[c] for c in sorted(dist)], dtype=float)
    total = probs.sum()
    if total <= 0 or len(counts) < 2:
        return dict(dist)
    probs = probs / total

    # A mean outside the support is unreachable by tilting; the closest
    # attainable answer is the endpoint itself.
    target = float(np.clip(target, counts.min(), counts.max()))
    if abs(float((probs * counts).sum()) - target) < 1e-9:
        return {int(c): float(p) for c, p in zip(counts, probs)}

    low, high = -10.0, 10.0
    for _ in range(80):
        theta = 0.5 * (low + high)
        weights = probs * np.exp(theta * counts)
        weights /= weights.sum()
        if float((weights * counts).sum()) < target:
            low = theta
        else:
            high = theta

    weights = probs * np.exp(0.5 * (low + high) * counts)
    weights /= weights.sum()
    return {int(c): float(p) for c, p in zip(counts, weights)}


def _non_increasing(values: np.ndarray) -> np.ndarray:
    """
    Nearest non-increasing sequence, by pool-adjacent-violators.

    Written out rather than imported from sklearn so the constraint is
    visible and so this file does not depend on IsotonicRegression's
    handling of ties and duplicate x values. Standard PAVA: walk left to
    right, and whenever a value rises above the block before it, merge the
    two into one block at their common mean and re-check backwards.
    """
    blocks = []   # [sum, count]
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and (blocks[-2][0] / blocks[-2][1]) < (blocks[-1][0] / blocks[-1][1]):
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return np.array(out)


def enforce_slot_monotonicity(frame: pd.DataFrame, pa_dists: list,
                              verbose: bool = True) -> list:
    """
    Force expected plate appearances to be non-increasing down each team's
    batting order, adjusting the distributions to match.

    `frame` needs `game_pk`, `lineup_slot`, and a team identifier. Rows
    without a slot are left alone -- an unknown slot has no position in the
    sequence, so there is nothing to constrain it against.

    Returns a NEW list of distributions; the input is not mutated.
    """
    adjusted = [dict(d) for d in pa_dists]
    if "lineup_slot" not in frame.columns or "game_pk" not in frame.columns:
        return adjusted

    team_col = next((c for c in ("team", "team_id") if c in frame.columns), None)
    if team_col is None:
        return adjusted

    means = np.array([sum(k * v for k, v in d.items()) for d in adjusted])
    slots = pd.to_numeric(frame["lineup_slot"], errors="coerce").to_numpy()
    positions = np.arange(len(frame))

    changed, worst, groups_fixed = 0, 0.0, 0
    for _, group in frame.groupby(["game_pk", team_col], sort=False):
        rows = positions[frame.index.get_indexer(group.index)]
        valid = rows[~np.isnan(slots[rows])]
        if len(valid) < MONOTONICITY_MIN_HITTERS:
            continue

        order = valid[np.argsort(slots[valid], kind="stable")]
        current = means[order]
        target = _non_increasing(current)

        if np.allclose(current, target, atol=1e-9):
            continue
        groups_fixed += 1
        for row, new_mean in zip(order, target):
            if abs(new_mean - means[row]) < 1e-9:
                continue
            changed += 1
            worst = max(worst, abs(new_mean - means[row]))
            adjusted[row] = _tilt_distribution_to_mean(adjusted[row], float(new_mean))

    if verbose:
        if changed:
            print(f"  PA monotonicity: {groups_fixed} lineups had a slot "
                  f"projected above the one ahead of it, which the batting "
                  f"order forbids.")
            print(f"    {changed} hitters adjusted, largest change "
                  f"{worst:.3f} PA.")
        else:
            print("  PA monotonicity: every lineup already non-increasing "
                  "down the order.")
    return adjusted
