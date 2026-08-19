"""
Does the two-stage RBI model actually beat the flat constant it replaces?

    python eval_rbi.py

V5 used each batter's season RBI-per-plate-appearance as a constant. This
compares that against the lineup-aware model on held-out games, with a
chronological split.

The comparison is deliberately unflattering to the new model: the constant
is a genuinely strong baseline, because a hitter's RBI rate really is
mostly a property of him and his usual lineup position. If the lineup-aware
version can't beat it, that is the answer and it goes in CONTEXT.md next to
the RBI and chase-rate findings rather than getting quietly shipped.

Metrics are mean absolute error and R-squared on RBI per plate appearance,
plus the downstream number that actually matters -- whether the
hits+RBI+walks over/under lines calibrate better.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import START_DATE, END_DATE
from data.cache import cache_path
from data.lineup_slots import attach_lineup_slots
from data.park_factors import get_park_factor
from features.pa_table import build_pa_table, build_game_totals
from features.rate_features import add_batter_rolling_rates
from features.rbi_features import (
    add_base_state, add_lineup_obp_context, train_runner_model,
    train_rbi_model, predict_rbi_per_pa, runner_summary,
    RUNNER_FEATURES, RBI_FEATURES,
)
from train_props_v5 import load_raw_statcast

TRAIN_FRAC = 0.75


def main():
    combined = load_raw_statcast()

    print("\nBuilding plate-appearance table with base-out state...")
    # build_pa_table now carries on_1b/on_2b/on_3b/outs_when_up through
    # natively, so the base state comes straight off it. (It didn't
    # originally, which forced a fragile positional re-attachment here --
    # exactly the kind of index-alignment code that silently mismatches
    # rows if either sort ever changes.)
    pa = build_pa_table(combined)
    pa = add_base_state(pa)
    pa = add_batter_rolling_rates(pa, targets=("is_hr", "is_hit"))
    print(f"  {len(pa):,} plate appearances")

    print("\n  Base state by lineup slot is only visible once slots are joined.")
    game_totals = build_game_totals(pa)
    try:
        game_totals = attach_lineup_slots(game_totals, verbose=False)
    except Exception:
        game_totals["lineup_slot"] = np.nan

    matched = int(game_totals["lineup_slot"].notna().sum())
    print(f"  {matched:,} of {len(game_totals):,} player-games have a lineup slot.")
    if matched < 1000:
        raise SystemExit("Not enough lineup data. Run build_wide_pool.py first.")

    # Per-game runner context, joined back down to plate appearances.
    per_game_runners = pa.groupby(["batter", "game_pk"]).agg(
        runners_on=("runners_on", "mean"),
        runners_in_scoring=("runners_in_scoring", "mean"),
        rbi=("rbi", "sum"),
        pa_count=("rbi", "size"),
    ).reset_index()
    game_frame = game_totals.merge(
        per_game_runners.drop(columns=["rbi"]), on=["batter", "game_pk"], how="inner"
    )

    obp = (pa.groupby("batter")["is_hit"].mean()
           + pa.groupby("batter")["is_walk"].mean())
    league_obp = float(pa["is_hit"].mean() + pa["is_walk"].mean())
    game_frame = add_lineup_obp_context(game_frame, obp, league_obp)

    print("\n--- The effect stage 1 exploits ---")
    summary = runner_summary(
        game_frame.assign(rbi=game_frame["rbi"])
        if "rbi" in game_frame else game_frame
    )
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- Chronological split ------------------------------------------
    game_frame = game_frame.sort_values("game_date").reset_index(drop=True)
    split_date = game_frame["game_date"].quantile(TRAIN_FRAC)
    train_games = game_frame[game_frame["game_date"] <= split_date]
    test_games = game_frame[game_frame["game_date"] > split_date]

    pa = pa.sort_values("game_date").reset_index(drop=True)
    train_pa = pa[pa["game_date"] <= split_date]
    test_pa = pa[pa["game_date"] > split_date]

    print(f"\nSplit at {split_date.date()}: {len(train_pa):,} train PA | "
          f"{len(test_pa):,} test PA")

    runner_model = train_runner_model(train_games)
    rbi_model = train_rbi_model(train_pa)
    if runner_model is None or rbi_model is None:
        raise SystemExit("Not enough data to fit both stages.")

    print("\n--- Stage 1: expected runners on base ---")
    for name, coef in zip(RUNNER_FEATURES, runner_model.named_steps["reg"].coef_):
        print(f"    {name:<14} {coef:+.4f}  (per 1 s.d.)")
    print("\n--- Stage 2: expected RBI given base state ---")
    for name, coef in zip(RBI_FEATURES, rbi_model.named_steps["reg"].coef_):
        print(f"    {name:<22} {coef:+.4f}")

    # ---- The comparison ------------------------------------------------
    league_rbi = float(train_pa["rbi"].mean())
    batter_constant = train_pa.groupby("batter")["rbi"].mean()

    test = test_games.copy()
    test["bat_is_hr_career"] = test["batter"].map(
        train_pa.groupby("batter")["is_hr"].mean()).fillna(0)
    test["bat_is_hit_career"] = test["batter"].map(
        train_pa.groupby("batter")["is_hit"].mean()).fillna(0)
    test["outs_when_up"] = 1.0

    actual = test["rbi"] / test["pa_count"].replace(0, np.nan)
    actual = actual.fillna(0)

    predictions = {
        "league constant": np.full(len(test), league_rbi),
        "per-batter constant (V5)": test["batter"].map(batter_constant)
                                        .fillna(league_rbi).to_numpy(),
        "two-stage lineup model": predict_rbi_per_pa(
            runner_model, rbi_model, test, league_rbi),
    }

    print("\n" + "=" * 68)
    print("RBI PER PLATE APPEARANCE -- held-out games")
    print("=" * 68)
    print(f"{'method':<28} {'MAE':>9} {'R2':>9} {'mean pred':>11}")
    baseline_var = float(((actual - actual.mean()) ** 2).mean())
    for label, pred in predictions.items():
        mae = float(np.abs(pred - actual).mean())
        mse = float(((pred - actual) ** 2).mean())
        r2 = 1.0 - mse / baseline_var if baseline_var > 0 else float("nan")
        print(f"{label:<28} {mae:>9.4f} {r2:>+9.4f} {np.mean(pred):>11.4f}")
    print(f"{'actual':<28} {'':>9} {'':>9} {actual.mean():>11.4f}")

    print("\nR2 here is against 'always predict the average', same idea as")
    print("Brier skill. Positive means the method beats that; negative means")
    print("it is worse than a single number for everybody.")
    print("\nIf the two-stage model does not beat the per-batter constant,")
    print("that is a real negative result -- record it and keep the constant.")


if __name__ == "__main__":
    main()
