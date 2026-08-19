"""
Context and "superstition" features -- the park a hitter is standing in,
the day of the week, day vs night, home vs road, and the weather -- tested
honestly instead of either believed or dismissed.

THE PROBLEM WITH RAW SPLITS
---------------------------
"He has 5 home runs on Sundays" and "he hits more home runs at this park"
are the kind of claims that fill prop-betting cards. Some of them describe
something real (park dimensions ARE different; wind IS blowing out) and
some are pure sampling noise dressed up as a pattern.

The failure mode is always the same. Take a hitter with a 5% per-plate-
appearance home run rate who has played 6 games at some park and homered
in 2 of them. The raw split says 33% -- almost 7x his real rate. Feed that
number to a model as a feature and the model learns to trust a coin that
landed heads twice.

THE FIX: SHRINKAGE
------------------
Every split here is shrunk toward the player's OWN baseline rate using an
empirical-Bayes weighting:

    shrunk = (successes_in_split + k * baseline) / (games_in_split + k)

k is a prior strength in units of games. With k = 30:

    2 HR in 6 games at a park, baseline 25%   ->  (2 + 7.5)/(6 + 30) = 26.4%
    18 HR in 60 games at a park, baseline 25% ->  (18 + 15)/(60 + 30) = 36.7%

The 6-game split barely moves off baseline. The 60-game split is allowed
to say something. That is exactly the behaviour you want: a split has to
EARN its deviation with sample size.

The feature handed to the model is the DELTA (shrunk - baseline), not the
raw rate, so it reads as "how much does this context move this hitter off
his own norm" and it collapses to ~0 when the evidence is thin. A
coefficient near zero on these columns then means something clean --
"after shrinkage, this context carries no signal" -- rather than being
confounded with noisy small-sample splits.

LOOKAHEAD GUARD
---------------
Both the split counts and the baseline are EXPANDING and shifted by one
game within each player. A game's venue feature is built only from that
player's earlier games at that venue. Without the shift, a player who
homered today would have today's home run inside "his rate at this park,"
which would leak the label directly into the feature and produce a
spectacular, entirely fake AUC.

WEATHER
-------
Temperature and wind come from data/game_context.py (MLB's own Stats API,
the same source public sites display). These are physics, not folklore:
warm thin air carries a fly ball further, and a wind blowing out adds to
carry while a wind blowing in subtracts. They are included with no
shrinkage because they're measured conditions, not estimated splits.
"""
import numpy as np
import pandas as pd

# Prior strength, in games. 30 games is roughly a fifth of a season -- weak
# enough that a genuine multi-season park effect can still express itself,
# strong enough that a 6-game sample stays pinned near baseline.
DEFAULT_PRIOR_GAMES = 30.0

CALENDAR_SPLIT_COLS = [
    "venue_edge", "dow_edge", "home_edge", "daynight_edge",
]
WEATHER_COLS = [
    "temp_f", "wind_out_mph", "wind_pull_mph",
]


def add_shrunk_split_edge(
    df: pd.DataFrame,
    split_col: str,
    out_col: str,
    target_col: str = "hit_hr",
    player_col: str = "batter",
    date_col: str = "game_date",
    prior_games: float = DEFAULT_PRIOR_GAMES,
) -> pd.DataFrame:
    """
    Build one shrunk-split feature.

    For each row, using ONLY that player's strictly earlier games:
      baseline  = his overall rate of `target_col` so far
      split     = his rate of `target_col` so far within this same
                  `split_col` value (this venue / this weekday / etc.)
      out_col   = shrunk(split) - baseline

    Also emits `{out_col}_n` -- how many prior games back the split. Not
    fed to the model; useful when eyeballing whether a big edge is real or
    a 4-game fluke.
    """
    out = df.sort_values([player_col, date_col]).copy()

    # Player baseline: expanding mean over all prior games.
    grouped_player = out.groupby(player_col, sort=False)[target_col]
    baseline = grouped_player.transform(
        lambda s: s.expanding(min_periods=1).mean().shift(1)
    )

    # Split-specific counts: expanding within (player, split value).
    grouped_split = out.groupby([player_col, split_col], sort=False)[target_col]
    split_successes = grouped_split.transform(
        lambda s: s.expanding(min_periods=1).sum().shift(1)
    )
    split_games = grouped_split.transform(
        lambda s: s.expanding(min_periods=1).count().shift(1)
    )

    split_successes = split_successes.fillna(0.0)
    split_games = split_games.fillna(0.0)

    shrunk = (split_successes + prior_games * baseline) / (split_games + prior_games)

    out[out_col] = shrunk - baseline
    out[f"{out_col}_n"] = split_games

    # A player's very first game has no baseline at all -- no edge to claim.
    out[out_col] = out[out_col].where(baseline.notna(), 0.0)
    return out


def add_context_features(
    features_df: pd.DataFrame,
    statcast_df: pd.DataFrame,
    prior_games: float = DEFAULT_PRIOR_GAMES,
) -> pd.DataFrame:
    """
    Attach handedness, home/away, venue, calendar and weather context to a
    per-(batter, game) feature frame, then build the shrunk split edges.

    Expects `features_df` to already carry the MLB Stats API context
    columns from data.game_context.attach_game_context (venue_id,
    day_night, temp_f, wind_out_mph, wind_toward_lf). Missing columns are
    tolerated -- the corresponding features come back as 0 rather than
    crashing, so this can run before the (slow) first weather pull.
    """
    df = features_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # --- Handedness and home/away, straight from the Statcast pull -------
    # `stand` is the batter's side for that plate appearance (switch
    # hitters change it by pitcher, so it's taken per game, not per career).
    # `inning_topbot` == 'Bot' means the batter's team is batting in the
    # bottom half, i.e. they are the HOME team.
    per_game = statcast_df.copy()
    per_game["game_date"] = pd.to_datetime(per_game["game_date"])
    agg_spec = {}
    if "stand" in per_game.columns:
        agg_spec["stand"] = ("stand", "first")
    if "inning_topbot" in per_game.columns:
        agg_spec["_topbot"] = ("inning_topbot", "first")
    if "home_team" in per_game.columns:
        agg_spec["home_team"] = ("home_team", "first")

    if agg_spec:
        meta = per_game.groupby(["batter", "game_pk"]).agg(**agg_spec).reset_index()
        df = df.merge(meta, on=["batter", "game_pk"], how="left", suffixes=("", "_meta"))

    if "_topbot" in df.columns:
        df["is_home"] = (df["_topbot"] == "Bot").astype(int)
        df = df.drop(columns=["_topbot"])
    else:
        df["is_home"] = 0

    if "stand" not in df.columns:
        df["stand"] = "R"
    df["stand"] = df["stand"].fillna("R")

    # --- Venue key -------------------------------------------------------
    # Prefer the Stats API venue_id (stable, handles teams changing parks);
    # fall back to Statcast's home_team abbreviation.
    if "venue_id" in df.columns and df["venue_id"].notna().any():
        df["venue_key"] = df["venue_id"].fillna(-1).astype("int64").astype(str)
    elif "home_team" in df.columns:
        df["venue_key"] = df["home_team"].fillna("UNK").astype(str)
    else:
        df["venue_key"] = "UNK"

    # --- Calendar keys ---------------------------------------------------
    df["day_of_week"] = df["game_date"].dt.dayofweek        # 0 = Monday
    if "day_night" not in df.columns:
        df["day_night"] = "unknown"
    df["day_night"] = df["day_night"].fillna("unknown").astype(str)

    # --- Shrunk split edges ---------------------------------------------
    df = add_shrunk_split_edge(df, "venue_key", "venue_edge", prior_games=prior_games)
    df = add_shrunk_split_edge(df, "day_of_week", "dow_edge", prior_games=prior_games)
    df = add_shrunk_split_edge(df, "is_home", "home_edge", prior_games=prior_games)
    df = add_shrunk_split_edge(df, "day_night", "daynight_edge", prior_games=prior_games)

    # --- Weather ---------------------------------------------------------
    if "temp_f" not in df.columns:
        df["temp_f"] = np.nan
    df["temp_f"] = pd.to_numeric(df["temp_f"], errors="coerce")

    # Sanity floor. MLB's feed occasionally reports 0 F for a game -- found
    # on real data at Tropicana Field and Minute Maid Park, both domes.
    # Baseball is not played at 0 F, so that's a missing reading encoded as
    # a number, and left alone it would drag the coefficient toward a
    # relationship that doesn't exist. Anything below freezing is treated
    # as absent.
    df.loc[df["temp_f"] <= 32, "temp_f"] = np.nan

    # Median-fill rather than drop: a missing weather reading shouldn't
    # cost the whole game row, and a median-filled temp is a neutral
    # assumption rather than an informative one.
    df["temp_f"] = df["temp_f"].fillna(df["temp_f"].median())

    if "wind_out_mph" not in df.columns:
        df["wind_out_mph"] = 0.0
    df["wind_out_mph"] = pd.to_numeric(df["wind_out_mph"], errors="coerce").fillna(0.0)

    # Pull-side wind: `wind_toward_lf` is positive toward LEFT field, which
    # is the pull field for a RIGHT-handed batter and the opposite field
    # for a lefty. Flipping by handedness turns a field-absolute number
    # into a batter-relative one ("is the wind blowing where I hit it").
    if "wind_toward_lf" not in df.columns:
        df["wind_toward_lf"] = 0.0
    df["wind_toward_lf"] = pd.to_numeric(df["wind_toward_lf"], errors="coerce").fillna(0.0)
    handed_sign = np.where(df["stand"].eq("L"), -1.0, 1.0)
    df["wind_pull_mph"] = df["wind_toward_lf"] * handed_sign

    return df.sort_values("game_date").reset_index(drop=True)


def context_feature_cols(include_weather: bool = True) -> list:
    cols = list(CALENDAR_SPLIT_COLS)
    if include_weather:
        cols += list(WEATHER_COLS)
    return cols
