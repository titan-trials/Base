"""
Additional prop targets beyond a single game's HR.

RBI_over(threshold): binary target, RBI (approximated -- see
multi_player_features.py) at or above a threshold for that game.

hr_next_N: a LOOSENED version of the HR target -- instead of "does this
exact game include a HR" (noisy, single-event), "does at least one of the
next N games include a HR". This is a forward-looking TARGET, not a
feature -- using future games as a label is fine (that's what we're trying
to predict), the lookahead-bias rule only applies to FEATURES peeking at
the future, which this doesn't do.
"""
import pandas as pd


def add_rbi_target(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    col_name = f"rbi_over_{str(threshold).replace('.', '_')}"
    df[col_name] = (df["rbi_approx"] >= threshold).astype(int)
    return df, col_name


def add_forward_hr_target(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Adds hr_next_{window}: whether hit_hr == 1 in any of the next
    `window` games (NOT including the current game), computed per player
    so one player's future games never leak into another's target."""
    df = df.copy()
    col_name = f"hr_next_{window}"

    def _add(group):
        reversed_roll = group["hit_hr"][::-1].rolling(window, min_periods=1).max()[::-1]
        group[col_name] = reversed_roll.shift(-1)
        return group

    df = df.groupby("batter", group_keys=False).apply(_add)
    df = df.dropna(subset=[col_name]).reset_index(drop=True)
    df[col_name] = df[col_name].astype(int)
    return df, col_name