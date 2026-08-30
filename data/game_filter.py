"""
Keep regular-season baseball, drop everything else.

WHY THIS FILE EXISTS
--------------------
Statcast tags every pitch with `game_type`:

    R  regular season
    S  spring training
    E  exhibition
    D  division series      L  league championship
    W  world series         F  wild card

Every cached pull in this project carries that column, and until now
nothing looked at it. About 11% of the wide-pool cache -- 66,000 of
605,799 plate appearances -- is therefore not regular-season baseball.

HOW MUCH IT ACTUALLY MATTERS, MEASURED
---------------------------------------
Less than a first look suggested, and it is worth writing the real
numbers down because an early estimate taken from three files of the old
nine-slugger cache was wildly wrong (it claimed home runs were 25% off;
they are not).

Across all 388 wide-pool files, pooled rate versus the regular-season
truth:

    home run   0.03233 vs 0.03205    +0.85%
    hit        0.22517 vs 0.22406    +0.50%
    walk       0.09513 vs 0.09377    +1.45%
    strikeout  0.21142 vs 0.21147    -0.02%

So on a season-long average this is a small correction, and anyone
expecting it to move tonight's numbers will be disappointed.

THE SEASONAL AVERAGE HIDES THE REAL PROBLEM
--------------------------------------------
Spring training is not spread through the year. Share of 2026 plate
appearances that are NOT regular season, by month:

    February   100.0%
    March       69.4%
    April        0.0%
    May onward   0.0%

In August this filter changes almost nothing. In April, a trailing
150-plate-appearance window reaches back into March and is substantially
spring training -- pitchers building arm strength against minor-league
hitters. That is when the model is quietly training on a different sport,
and that is what this prevents.

The effect on the pitcher workload model was much larger and immediate,
because outing LENGTH is exactly what spring training distorts: 1.5
batters faced per start, which was masking a 6% error in the opposite
direction.
"""
import pandas as pd

REGULAR_SEASON = "R"


def regular_season_only(df: pd.DataFrame, verbose: bool = False,
                        label: str = "") -> pd.DataFrame:
    """
    Rows from regular-season games only.

    A frame with no `game_type` column is returned untouched rather than
    emptied. Older cached pulls predate the column, and silently
    discarding every row of one would be a far worse failure than
    including a little spring training.
    """
    if "game_type" not in df.columns:
        return df
    keep = df["game_type"] == REGULAR_SEASON
    dropped = int((~keep).sum())
    if verbose and dropped:
        where = f" from {label}" if label else ""
        print(f"  Dropped {dropped:,} non-regular-season rows{where} "
              f"({dropped / max(len(df), 1):.1%}) -- spring training and "
              f"postseason.")
    return df[keep]
