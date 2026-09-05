"""
Modelling changes that are BUILT but not yet EARNED.

Every flag here defaults to the behaviour the backtests and the live
scoring log were produced with. Turning one on changes the numbers the
model prints, and until a rolling-origin comparison says it is better the
running total should keep describing the model it has been describing.

    python flag_lab.py          # rolling-origin comparison, each flag on vs off

The review of 2026-09-05 lists why each of these is expected to help.
None of them is a new "level" feature -- the class that has failed five
times in this project -- they change what the model can EXPRESS (an
interaction, the link scale, a per-hitter shape) or what it is shrunk
TOWARD.

Read at call time by the modules that implement them, so a lab can flip
them in-process without re-importing anything.
"""

# 1. Shrinkage target and pitcher window ---------------------------------
#
# `prior_mean` in add_batter_rolling_rates is the mean over EVERY plate
# appearance in the cache, 2022-2026. A 150-PA window in September 2026 is
# shrunk toward a league rate that includes 2022. The pitcher K model
# measured all-history at +6.8% versus the trailing twelve months. With
# this set, the shrinkage target for each plate appearance is the league
# rate over the trailing N days as of that plate appearance's date.
LEAGUE_PRIOR_TRAILING_DAYS = None       # e.g. 365

# Pitcher rates allowed (pit_{target}_allowed) are career-expanding at
# training time and career-total at serving time. With this set, both
# sides use a trailing window of N days instead. Applies to the pool-based
# rates in add_pitcher_rolling_rates AND to build_pitcher_rates, so train
# and serve stay consistent.
PITCHER_RATE_WINDOW_DAYS = None         # e.g. 365

# 2. Link scale -------------------------------------------------------
#
# The per-PA logistic regression takes bat_*, pit_*_allowed and matchup_*
# on the PROBABILITY scale. Per-PA HR rate spans roughly 1%-8%; the model
# has to bend a linear-in-p feature onto a logit link. With this set,
# every rate column is fed as logit(p) instead. Same model, one transform.
LOGIT_FEATURES = False

# 3. Platoon as two indicators instead of one -------------------------
#
# `platoon_edge` is a single 0/1 with one coefficient. The left-on-left
# penalty is materially larger than right-on-right, for strikeouts and
# for power, and one coefficient cannot express that. With this set the
# feature list carries `lhb_vs_lhp` and `rhb_vs_rhp` (same-hand
# indicators by batter side) in place of `platoon_edge`.
PLATOON_SPLIT = False

# 4. Per-hitter contribution shape ------------------------------------
#
# Total bases and H+R+RBI use ONE population per-PA contribution
# distribution, scaled to each hitter's mean by the "frequency" method.
# A slugger and a slap hitter with the same expected total bases per PA
# get the same P(4 bases | something happened). With this set, the total
# bases shape is built per hitter from his shrunk single/double/triple/HR
# mix (bases_features already estimates it) with p_is_hr and p_is_hit as
# the anchors.
PER_HITTER_SHAPE = False


def rate_feature_transform(values):
    """Identity or logit, per LOGIT_FEATURES. Applied to rate columns only."""
    import numpy as np
    if not LOGIT_FEATURES:
        return values
    p = np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1.0 - p))
