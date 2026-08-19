"""
Dry run of the ACTUAL train_hr_v4.py pipeline against synthetic data.

test_v4_synthetic.py checks each new feature function in isolation. This
one checks the real script end to end -- build_dataset(), the ablation
ladder, calibration, the reliability table and the permutation tests --
with only the three network-touching loaders stubbed out:

    load_batter_statcast_cached  -> synthetic Statcast frame
    build_pitcher_mix_table      -> synthetic pitch mixes
    attach_game_context          -> synthetic weather / venue / day-night

Everything else is the production code path. The point is to catch a
KeyError or a merge that explodes row counts BEFORE spending an hour on
the real weather fetch.

    python test_v4_dryrun.py
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

import test_v4_synthetic as synth

SYNTHETIC = synth.make_synthetic_statcast()
_BATTERS = sorted(SYNTHETIC["batter"].unique())


def fake_load_batter_statcast_cached(first, last, start_dt, end_dt):
    """One synthetic player per config.PLAYERS entry, cycling through the
    synthetic batter ids."""
    index = fake_load_batter_statcast_cached.counter % len(_BATTERS)
    fake_load_batter_statcast_cached.counter += 1
    return SYNTHETIC[SYNTHETIC["batter"] == _BATTERS[index]].copy()


fake_load_batter_statcast_cached.counter = 0


def fake_build_pitcher_mix_table(pairs):
    rows = []
    rng = np.random.default_rng(3)
    for pid, season in pairs:
        fastball = float(rng.uniform(0.35, 0.60))
        breaking = float(rng.uniform(0.20, 0.40))
        rows.append({
            "opp_pitcher_id": pid,
            "season": season,
            "fastball_pct": fastball,
            "breaking_pct": breaking,
            "offspeed_pct": max(0.0, 1.0 - fastball - breaking),
        })
    return pd.DataFrame(rows)


def fake_attach_game_context(features_df, verbose=True):
    meta = SYNTHETIC.groupby("game_pk").agg(
        home_team=("home_team", "first"),
        temp_f=("_synthetic_temp_f", "first"),
    ).reset_index()
    venue_ids = {v: 3300 + i for i, v in enumerate(meta["home_team"].unique())}
    meta["venue_id"] = meta["home_team"].map(venue_ids)
    meta["venue_name"] = meta["home_team"] + " Park"
    meta["day_night"] = np.where(meta["game_pk"] % 3 == 0, "day", "night")
    rng = np.random.default_rng(11)
    meta["wind_mph"] = rng.uniform(0, 18, len(meta))
    meta["wind_out_mph"] = rng.normal(0, 6, len(meta))
    meta["wind_toward_lf"] = rng.normal(0, 5, len(meta))
    meta["sky_condition"] = "Clear"
    meta["wind_dir_raw"] = "out to cf"
    return features_df.merge(meta.drop(columns="home_team"), on="game_pk", how="left")


def main():
    import train_hr_v4
    import features.pitch_mix_features  # noqa: F401

    # Patch the three network-touching entry points, in the namespace
    # train_hr_v4 actually calls them from.
    train_hr_v4.load_batter_statcast_cached = fake_load_batter_statcast_cached
    train_hr_v4.build_pitcher_mix_table = fake_build_pitcher_mix_table
    train_hr_v4.attach_game_context = fake_attach_game_context

    # Keep the dry run quick -- the real script defaults to 200.
    train_hr_v4.N_PERMUTATIONS = 30
    train_hr_v4.PLAYERS = [(f"Synth{i}", f"Player{i}") for i in range(len(_BATTERS))]

    print("=" * 70)
    print("DRY RUN of train_hr_v4.main() on synthetic data")
    print("(loaders stubbed; every other line is production code)")
    print("=" * 70)
    train_hr_v4.main()

    print("\n" + "=" * 70)
    print("DRY RUN COMPLETE -- train_hr_v4.py executes end to end.")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
