"""
Contact-quality features -- HOW a batter is hitting the ball, not whether
the ball happened to leave the yard.

WHY THIS IS THE MOST PROMISING FEATURE GROUP IN THE PROJECT
-----------------------------------------------------------
The single clearest finding in CONTEXT.md is that CONSISTENT INDIVIDUAL
SKILLS predict better than RARE BURSTY EVENTS. Walk rate (a stable
behavioural tendency) reached AUC ~0.59; HR rate (a rare outcome) sat near
a coin flip.

A home run is the rare event. But the SKILL underneath a home run is not:

    hr_rate_10          ~2-4 home runs inside a 10-game window.
                        Tiny sample, enormous variance -- mostly noise.

    sweet_spot_rate_10  ~25-35 batted balls inside the same 10 games.
                        An order of magnitude more observations of the
                        same underlying ability.

Same skill, roughly 10x the sample. That is the whole idea here: measure
the repeatable process (does this hitter square balls up right now)
instead of the lottery-ticket outcome (did one of them clear a fence).

DEFINITIONS
-----------
sweet_spot   Launch angle between 8 and 32 degrees. Statcast's own
             published definition -- the angle band where batted balls do
             the most damage. Rate is per batted ball.

barrel       Statcast's own classification. Preferred source is the
             `launch_speed_angle` column (6 == barrel) when present in the
             cached pull. Fallback is the standard public approximation:
             exit velocity >= 98 mph, with the qualifying launch-angle
             band widening by 1 degree in each direction per mph above 98.

hard_hit     Exit velocity >= 95 mph. Statcast's own threshold.

fb_ld_ev     Average exit velocity on FLY BALLS AND LINE DRIVES only
             (launch angle >= 10). Overall average EV is polluted by
             ground balls -- a hitter can raise it by scalding grounders,
             which produce zero home runs. This isolates EV on the batted
             balls that can actually leave the park.

pull_air_rate Share of air balls (LA >= 10) hit to the pull field. Nearly
             all home runs are pulled or hit dead centre; an oppo-field
             air-ball hitter converts power into home runs less often.
             Uses Statcast's `hc_x`/`hc_y` hit coordinates plus the
             batter's own handedness (`stand`).

LOOKAHEAD GUARD
---------------
Every rolling and expanding column is `.shift(1)` inside a per-batter
groupby, matching the rule used everywhere else in this project: a game's
features never see that game's own outcome, and one player's history
never bleeds into another's window.
"""
import numpy as np
import pandas as pd

SWEET_SPOT_MIN_LA = 8.0
SWEET_SPOT_MAX_LA = 32.0
HARD_HIT_MIN_EV = 95.0
BARREL_MIN_EV = 98.0
AIR_BALL_MIN_LA = 10.0

# Statcast's hit-coordinate system: home plate sits near (125.42, 203.5),
# y grows DOWNWARD (screen coordinates), so outfield is at smaller y.
HOME_PLATE_X = 125.42
HOME_PLATE_Y = 203.5

DEFAULT_WINDOWS = (10, 20, 40)


def _is_barrel_formula(exit_velo, launch_angle) -> bool:
    """
    Public approximation of Statcast's barrel definition, used only when
    the `launch_speed_angle` column is absent from the cached pull.

    At 98 mph the qualifying launch-angle band is 26-30 degrees; each
    additional mph widens it by 1 degree in both directions.
    """
    if pd.isna(exit_velo) or pd.isna(launch_angle) or exit_velo < BARREL_MIN_EV:
        return False
    widening = exit_velo - BARREL_MIN_EV
    lower = max(SWEET_SPOT_MIN_LA, 26.0 - widening)
    upper = min(50.0, 30.0 + widening)
    return lower <= launch_angle <= upper


def _spray_angle_deg(hc_x, hc_y):
    """
    Horizontal spray angle in degrees: 0 = dead centre field,
    negative = toward left field, positive = toward right field.
    NaN when hit coordinates are missing (common on older or untracked
    batted balls) -- callers must handle that.
    """
    dx = hc_x - HOME_PLATE_X
    dy = HOME_PLATE_Y - hc_y  # flip so "toward the outfield" is positive
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arctan2(dx, dy))


def build_contact_quality_log(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (batter, game_pk, game_date): batted-ball counts and quality
    counts for one game. Counts rather than rates, because the rolling
    step below sums counts across a window and divides once -- dividing
    per game first and averaging the rates would weight a 1-batted-ball
    game the same as a 5-batted-ball one.
    """
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # A batted ball is any pitch with a measured launch angle.
    batted = df[df["launch_angle"].notna()].copy()
    if batted.empty:
        raise ValueError(
            "No batted balls found (no non-null launch_angle). Statcast "
            "batted-ball tracking is reliable from 2015 on -- check the "
            "date range and that the cached pull isn't truncated."
        )

    batted["is_sweet_spot"] = (
        batted["launch_angle"].between(SWEET_SPOT_MIN_LA, SWEET_SPOT_MAX_LA)
    ).astype(int)
    batted["is_hard_hit"] = (batted["launch_speed"] >= HARD_HIT_MIN_EV).astype(int)

    if "launch_speed_angle" in batted.columns and batted["launch_speed_angle"].notna().any():
        # Statcast's own batted-ball classification: 6 == barrel.
        batted["is_barrel"] = (batted["launch_speed_angle"] == 6).astype(int)
    else:
        batted["is_barrel"] = [
            int(_is_barrel_formula(ev, la))
            for ev, la in zip(batted["launch_speed"], batted["launch_angle"])
        ]

    batted["is_air"] = (batted["launch_angle"] >= AIR_BALL_MIN_LA).astype(int)
    batted["air_ev"] = batted["launch_speed"].where(batted["is_air"] == 1)

    # Pull-side air balls. spray angle sign is field-absolute, so it has to
    # be flipped by handedness: a lefty pulls to RIGHT field (positive
    # angle), a righty pulls to LEFT field (negative angle).
    if {"hc_x", "hc_y", "stand"}.issubset(batted.columns):
        spray = _spray_angle_deg(batted["hc_x"], batted["hc_y"])
        pull_direction = np.where(batted["stand"].eq("L"), 1.0, -1.0)
        pull_angle = spray * pull_direction  # positive = toward pull field
        batted["is_pull_air"] = (
            (batted["is_air"] == 1) & (pull_angle > 10.0)
        ).astype(int)
    else:
        batted["is_pull_air"] = 0

    grouped = batted.groupby(["batter", "game_pk", "game_date"]).agg(
        batted_balls=("launch_angle", "size"),
        sweet_spots=("is_sweet_spot", "sum"),
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        air_balls=("is_air", "sum"),
        pull_airs=("is_pull_air", "sum"),
        game_air_ev_sum=("air_ev", "sum"),
        game_air_ev_count=("air_ev", "count"),
        game_max_ev=("launch_speed", "max"),
    ).reset_index()

    return grouped.sort_values(["batter", "game_date"]).reset_index(drop=True)


def add_contact_quality_rolling(
    contact_log: pd.DataFrame, windows=DEFAULT_WINDOWS
) -> pd.DataFrame:
    """
    Rolling contact-quality RATES, computed as (sum of events) / (sum of
    opportunities) across the window, then shifted 1 game.

    min_periods is set to 3 games so an early-career player produces NaN
    rather than a rate built on two batted balls.
    """
    df = contact_log.copy()

    def _add(group):
        group = group.copy().sort_values("game_date")
        for w in windows:
            batted = group["batted_balls"].rolling(w, min_periods=3).sum().shift(1)
            air = group["air_balls"].rolling(w, min_periods=3).sum().shift(1)

            safe_batted = batted.replace(0, np.nan)
            safe_air = air.replace(0, np.nan)

            group[f"sweet_spot_rate_{w}"] = (
                group["sweet_spots"].rolling(w, min_periods=3).sum().shift(1) / safe_batted
            )
            group[f"barrel_rate_{w}"] = (
                group["barrels"].rolling(w, min_periods=3).sum().shift(1) / safe_batted
            )
            group[f"hard_hit_rate_{w}"] = (
                group["hard_hits"].rolling(w, min_periods=3).sum().shift(1) / safe_batted
            )
            group[f"air_rate_{w}"] = air / safe_batted
            group[f"pull_air_rate_{w}"] = (
                group["pull_airs"].rolling(w, min_periods=3).sum().shift(1) / safe_air
            )
            group[f"fb_ld_ev_{w}"] = (
                group["game_air_ev_sum"].rolling(w, min_periods=3).sum().shift(1)
                / group["game_air_ev_count"].rolling(w, min_periods=3).sum().shift(1).replace(0, np.nan)
            )
            group[f"max_ev_{w}"] = (
                group["game_max_ev"].rolling(w, min_periods=3).max().shift(1)
            )
            group[f"batted_balls_{w}"] = batted

        # Career baselines -- expanding, so they use everything before today.
        career_batted = group["batted_balls"].expanding(min_periods=5).sum().shift(1)
        group["career_sweet_spot_rate"] = (
            group["sweet_spots"].expanding(min_periods=5).sum().shift(1)
            / career_batted.replace(0, np.nan)
        )
        group["career_barrel_rate"] = (
            group["barrels"].expanding(min_periods=5).sum().shift(1)
            / career_batted.replace(0, np.nan)
        )
        group["career_hard_hit_rate"] = (
            group["hard_hits"].expanding(min_periods=5).sum().shift(1)
            / career_batted.replace(0, np.nan)
        )
        return group

    # Explicit loop rather than groupby().apply(): pandas 2.2 deprecated
    # apply's handling of the grouping column and pandas 3.0 changes it,
    # and this project pins pandas<3.0 as a workaround for a separate
    # pybaseball bug. A loop behaves identically across all three versions
    # and costs nothing at this data size (a few dozen players).
    pieces = [_add(group) for _, group in df.groupby("batter", sort=False)]
    return pd.concat(pieces, ignore_index=True)


def contact_quality_feature_cols(windows=DEFAULT_WINDOWS, primary_window: int = 20) -> list:
    """
    The subset worth handing to a linear model. Deliberately NOT every
    window of every stat -- sweet_spot_rate_10 and sweet_spot_rate_20 are
    almost the same number, and feeding both to a logistic regression just
    splits one coefficient across two collinear columns.

    Two windows are kept for the headline rates (short = current form,
    long = established level), one for the rest.
    """
    short = min(windows)
    long = max(windows)
    return [
        f"sweet_spot_rate_{short}", f"sweet_spot_rate_{long}",
        f"barrel_rate_{primary_window}", f"barrel_rate_{long}",
        f"hard_hit_rate_{primary_window}",
        f"fb_ld_ev_{primary_window}",
        f"air_rate_{primary_window}",
        f"pull_air_rate_{long}",
        f"max_ev_{long}",
        "career_barrel_rate",
    ]
