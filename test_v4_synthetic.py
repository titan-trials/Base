"""
Synthetic-data smoke test for the V4 feature pipeline.

Builds a fake Statcast-shaped dataframe with a KNOWN embedded signal, then
runs every new V4 feature function over it. Catches shape errors, KeyError
regressions, and -- most importantly -- lookahead leakage, before burning
time on a real cached run.

    python test_v4_synthetic.py

This does NOT hit the network. Weather columns are injected directly, so
data/game_context.py's HTTP path is exercised separately (see the
parse_wind assertions below, which cover the part most likely to be wrong:
the direction sign).
"""
import numpy as np
import pandas as pd

from data.game_context import parse_wind
from features.multi_player_features import build_multi_game_log, add_rolling_features_multi
from features.fatigue_features import add_fatigue_features
from features.contact_quality import (
    build_contact_quality_log, add_contact_quality_rolling,
    contact_quality_feature_cols, DEFAULT_WINDOWS, _is_barrel_formula,
)
from features.context_features import add_context_features, add_shrunk_split_edge
from model.hr_v4 import (
    chronological_split, fit_calibrated, predict_calibrated, evaluate,
    coefficient_report, permutation_test,
)

RNG = np.random.default_rng(7)
N_PLAYERS = 10
N_GAMES = 300
PITCHES_PER_GAME = 18

# Planted effects. The test asserts the pipeline recovers these -- if it
# can't find a signal this blatant, it won't find a subtle real one.
FORM_PERSISTENCE = 0.92     # AR(1) coefficient on a player's hot/cold state
TEMP_EFFECT = 0.004         # extra HR probability per degree F above 70
VENUE_BOOST = {"COL": 0.10}  # one genuinely hitter-friendly park


def make_synthetic_statcast() -> pd.DataFrame:
    """
    Fake pitch-level data with three REAL embedded effects and one
    deliberate non-effect:

      REAL   a persistent per-player form state (AR(1)) that drives BOTH
             launch angle / exit velocity AND home run probability. This
             is the thing contact-quality features are supposed to track:
             if sweet_spot_rate can see form, the pipeline works.
      REAL   temperature -- warmer games get more home runs.
      REAL   one hitter-friendly venue.
      NOT    day of week. Nothing depends on it, so a correct permutation
             test must return a non-significant p-value for dow_edge. A
             test that "finds" a weekday effect here is finding noise, and
             would mean the shrinkage or the split logic is leaking.
    """
    rows = []
    start = pd.Timestamp("2023-04-01")
    venues = ["NYY", "BOS", "COL", "SEA", "MIA"]

    for player in range(N_PLAYERS):
        batter_id = 600000 + player
        power = 0.14 + 0.022 * player         # underlying skill
        stand = "L" if player % 2 == 0 else "R"
        date = start
        form = 0.0

        for game in range(N_GAMES):
            date = date + pd.Timedelta(days=int(RNG.integers(1, 3)))
            game_pk = 700000 + player * 10000 + game
            venue = venues[(game + player) % len(venues)]
            home = RNG.random() < 0.5

            # Persistent hot/cold state -- the thing rolling features exist
            # to detect. Without persistence, no rolling window could ever
            # help, and the whole feature family would be untestable.
            form = FORM_PERSISTENCE * form + RNG.normal(0, 1.0)
            effective_power = float(np.clip(power + 0.055 * form, 0.02, 0.75))

            temp_f = float(RNG.normal(72, 13))
            game_hr_prob = np.clip(
                effective_power
                + TEMP_EFFECT * (temp_f - 70.0)
                + VENUE_BOOST.get(venue, 0.0),
                0.01, 0.85,
            )
            hits_hr = RNG.random() < game_hr_prob

            for pitch in range(PITCHES_PER_GAME):
                is_batted = RNG.random() < 0.40
                launch_angle = (
                    RNG.normal(14 + 22 * effective_power, 15) if is_batted else np.nan
                )
                launch_speed = (
                    RNG.normal(84 + 34 * effective_power, 8) if is_batted else np.nan
                )
                event = None
                if hits_hr and pitch == PITCHES_PER_GAME - 1:
                    # Force the batted-ball flag on so the planted HR rate
                    # isn't silently diluted by the is_batted draw.
                    is_batted = True
                    event = "home_run"
                    launch_angle, launch_speed = 27.0, 105.0
                elif is_batted:
                    event = RNG.choice(
                        ["single", "double", "field_out", "field_out"], p=[.2, .1, .35, .35]
                    )
                elif RNG.random() < 0.12:
                    event = RNG.choice(["walk", "strikeout"])

                rows.append({
                    "batter": batter_id,
                    "pitcher": 500000 + int(RNG.integers(0, 40)),
                    "game_pk": game_pk,
                    "game_date": date,
                    "events": event,
                    "description": RNG.choice(
                        ["hit_into_play", "ball", "called_strike", "swinging_strike", "foul"]
                    ),
                    "zone": int(RNG.integers(1, 15)),
                    "launch_angle": launch_angle,
                    "launch_speed": launch_speed,
                    "launch_speed_angle": (
                        6 if (is_batted and launch_speed and launch_speed > 100
                              and 24 < (launch_angle or 0) < 33) else 3
                    ),
                    "pitch_type": RNG.choice(["FF", "SL", "CH", "SI", "CU"]),
                    "stand": stand,
                    "home_team": venue,
                    "away_team": "XXX",
                    "inning_topbot": "Bot" if home else "Top",
                    "hc_x": RNG.normal(125, 45) if is_batted else np.nan,
                    "hc_y": RNG.normal(120, 40) if is_batted else np.nan,
                    "at_bat_number": pitch // 4 + 1,
                    "pitch_number": pitch % 4 + 1,
                    # Carried on the pitch rows purely so the test can pull
                    # the SAME temperature that generated the outcome,
                    # rather than injecting an unrelated random one.
                    "_synthetic_temp_f": temp_f,
                })

    return pd.DataFrame(rows)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def main():
    print("=" * 70)
    print("1. Wind parsing (the sign is the easiest thing to get backwards)")
    print("=" * 70)
    speed, key, out_mph, lf = parse_wind("11 mph, R To L")
    check("'R To L' speed parsed", speed == 11.0, f"got {speed}")
    check("'R To L' blows TOWARD left field (positive)", lf > 0, f"got {lf}")
    check("'R To L' has no out/in component", out_mph == 0.0, f"got {out_mph}")

    _, _, out_mph, lf = parse_wind("8 mph, Out To CF")
    check("'Out To CF' is positive out-component", out_mph == 8.0, f"got {out_mph}")
    _, _, out_mph, _ = parse_wind("14 mph, In From CF")
    check("'In From CF' is negative out-component", out_mph == -14.0, f"got {out_mph}")
    _, _, out_mph, lf = parse_wind("Calm")
    check("'Calm' is all zero", out_mph == 0.0 and lf == 0.0)
    _, _, out_mph, lf = parse_wind(None)
    check("missing wind is zero, not NaN", out_mph == 0.0 and lf == 0.0)

    print("\n" + "=" * 70)
    print("2. Barrel formula fallback")
    print("=" * 70)
    check("98 mph at 28deg is a barrel", _is_barrel_formula(98, 28))
    check("98 mph at 40deg is not", not _is_barrel_formula(98, 40))
    check("108 mph at 20deg is (band widens with EV)", _is_barrel_formula(108, 20))
    check("90 mph is never a barrel", not _is_barrel_formula(90, 28))

    print("\n" + "=" * 70)
    print("3. Building the synthetic dataset")
    print("=" * 70)
    statcast = make_synthetic_statcast()
    print(f"  {len(statcast):,} synthetic pitches, {statcast['batter'].nunique()} batters, "
          f"{statcast['game_pk'].nunique():,} games")

    game_log = build_multi_game_log(statcast)
    features_df = add_rolling_features_multi(game_log)
    features_df = add_fatigue_features(features_df)
    check("rolling form features built", len(features_df) > 500, f"{len(features_df)} rows")

    print("\n" + "=" * 70)
    print("4. Contact-quality features")
    print("=" * 70)
    contact_log = build_contact_quality_log(statcast)
    contact_rates = add_contact_quality_rolling(contact_log)
    cq_cols = contact_quality_feature_cols(DEFAULT_WINDOWS, primary_window=20)
    missing = [c for c in cq_cols if c not in contact_rates.columns]
    check("all declared contact-quality columns exist", not missing, f"missing {missing}")

    # Lookahead guard: a game's own batted balls must not be inside its own
    # rolling feature. Verified directly rather than trusted.
    one = contact_rates[contact_rates["batter"] == contact_rates["batter"].iloc[0]].reset_index(drop=True)
    manual = one["sweet_spots"].iloc[:10].sum() / one["batted_balls"].iloc[:10].sum()
    reported = one["sweet_spot_rate_10"].iloc[10]
    check("sweet_spot_rate_10 at game 11 equals games 1-10 only (no lookahead)",
          np.isclose(manual, reported, atol=1e-9),
          f"manual {manual:.6f} vs reported {reported:.6f}")

    features_df = features_df.merge(
        contact_rates[["batter", "game_pk", "game_date"] + cq_cols],
        on=["batter", "game_pk", "game_date"], how="left",
    )
    check("contact features merged without row explosion",
          len(features_df) == len(features_df.drop_duplicates(["batter", "game_pk"])),
          f"{len(features_df)} rows")

    print("\n" + "=" * 70)
    print("5. Shrinkage behaviour (the anti-superstition machinery)")
    print("=" * 70)
    toy = pd.DataFrame({
        "batter": [1] * 12,
        "game_date": pd.date_range("2024-04-01", periods=12),
        "hit_hr": [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "venue_key": ["A", "B", "A", "B", "B", "B", "A", "B", "B", "B", "B", "B"],
    })
    shrunk = add_shrunk_split_edge(toy, "venue_key", "venue_edge", prior_games=30.0)
    # Venue A: 3 games, all 3 home runs -> raw split rate 1.00 against a
    # baseline near 0.25, i.e. a raw edge of about +0.75. Shrinkage must
    # crush that almost entirely.
    raw_a = toy[toy["venue_key"] == "A"]["hit_hr"].mean()
    baseline = toy["hit_hr"].mean()
    raw_edge = abs(raw_a - baseline)
    max_edge = shrunk["venue_edge"].abs().max()
    check("shrinkage crushes a 3-game split by >85%",
          max_edge < raw_edge * 0.15,
          f"raw edge {raw_edge:+.3f} -> shrunk max |edge| {max_edge:.4f} "
          f"({100 * (1 - max_edge / raw_edge):.0f}% reduction)")

    big = pd.DataFrame({
        "batter": [1] * 200,
        "game_date": pd.date_range("2024-04-01", periods=200),
        "hit_hr": ([1] * 80) + ([0] * 120),
        "venue_key": (["A"] * 80) + (["B"] * 120),
    })
    shrunk_big = add_shrunk_split_edge(big, "venue_key", "venue_edge", prior_games=30.0)
    late_edge = shrunk_big["venue_edge"].iloc[-1]
    check("large-sample split is allowed to move off baseline",
          abs(late_edge) > 0.05, f"edge = {late_edge:+.4f} on a 119-game split")

    print("\n" + "=" * 70)
    print("6. Full context features (with injected weather)")
    print("=" * 70)
    venue_ids = {v: 3300 + i for i, v in enumerate(statcast["home_team"].unique())}
    game_meta = statcast.groupby("game_pk").agg(
        home_team=("home_team", "first"),
        temp_f=("_synthetic_temp_f", "first"),
    ).reset_index()
    game_meta["venue_id"] = game_meta["home_team"].map(venue_ids)
    game_meta["day_night"] = np.where(game_meta["game_pk"] % 3 == 0, "day", "night")
    game_meta["wind_out_mph"] = RNG.normal(0, 6, len(game_meta))
    game_meta["wind_toward_lf"] = RNG.normal(0, 5, len(game_meta))
    features_df = features_df.merge(
        game_meta.drop(columns="home_team"), on="game_pk", how="left"
    )

    features_df = add_context_features(features_df, statcast)
    for col in ["venue_edge", "dow_edge", "home_edge", "daynight_edge",
                "temp_f", "wind_out_mph", "wind_pull_mph", "is_home", "stand"]:
        check(f"context column '{col}' present", col in features_df.columns)

    lefty = features_df[features_df["stand"] == "L"]
    check("pull-side wind is sign-flipped for left-handed batters",
          np.allclose(lefty["wind_pull_mph"], -lefty["wind_toward_lf"]),
          "a lefty pulls to right field, so toward-LF wind must flip sign")

    print("\n" + "=" * 70)
    print("7. Model, calibration and metrics end to end")
    print("=" * 70)
    all_features = [
        "hr_rate_10", "hr_rate_20", "avg_ev_10", "avg_ev_20",
        "iso_proxy_10", "iso_proxy_20", "career_hr_rate", "pa_20",
        "rest_days", "games_last_7_days",
    ] + cq_cols + ["venue_edge", "dow_edge", "home_edge", "daynight_edge",
                   "temp_f", "wind_out_mph", "wind_pull_mph"]

    model_df = features_df.dropna(subset=all_features + ["hit_hr"]).reset_index(drop=True)
    print(f"  {len(model_df):,} complete rows, HR rate {model_df['hit_hr'].mean():.3f}")

    train_df, calib_df, test_df = chronological_split(model_df)
    check("three-way split sums to the whole",
          len(train_df) + len(calib_df) + len(test_df) == len(model_df))
    check("splits are chronologically ordered",
          train_df["game_date"].max() <= calib_df["game_date"].min()
          and calib_df["game_date"].max() <= test_df["game_date"].min())

    model, calibrator, method = fit_calibrated(train_df, calib_df, all_features)
    probs = predict_calibrated(model, calibrator, test_df[all_features])
    check("calibrated probabilities are valid", ((probs >= 0) & (probs <= 1)).all())
    print(f"  Calibration method chosen: {method}")

    result = evaluate(test_df["hit_hr"], probs, "synthetic")
    print(f"  AUC {result['auc']:.4f} | Brier {result['brier']:.5f} | "
          f"Brier skill {result['brier_skill']:+.4f}")
    check("synthetic embedded signal is detected (AUC well above 0.5)",
          result["auc"] > 0.60,
          f"got {result['auc']:.3f} -- signal was deliberately planted, so a "
          f"coin-flip result here would mean the pipeline is broken")

    coefs = coefficient_report(model, all_features)
    print("\n  Top 5 standardised coefficients:")
    print(coefs.head(5).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    temp_coef = float(coefs.loc[coefs["feature"] == "temp_f", "coef"].iloc[0])
    check("planted temperature effect is recovered with the right sign",
          temp_coef > 0,
          f"coef {temp_coef:+.4f} -- warmer air was planted to INCREASE home "
          f"runs, so a negative coefficient would mean a sign error somewhere")

    from model.hr_v4 import reliability_by_quantile
    rel = reliability_by_quantile(test_df["hit_hr"], probs, n_bins=6)
    check("reliability table builds with usable bucket sizes",
          rel["count"].min() >= 20, f"smallest bucket {int(rel['count'].min())} games")
    print(f"  Largest calibration gap across buckets: {rel['gap'].abs().max():.3f}")

    print("\n" + "=" * 70)
    print("8. Permutation test (small run, just checking it executes)")
    print("=" * 70)
    perm = permutation_test(train_df, calib_df, test_df, all_features,
                            ["dow_edge"], n_permutations=15)
    check("permutation test returns a usable p-value",
          0.0 < perm["p_value"] <= 1.0, f"p = {perm['p_value']:.3f}")
    print(f"  day-of-week p-value on synthetic data (no real weekday effect "
          f"planted): {perm['p_value']:.3f}")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
