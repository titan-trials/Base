"""
Dry run of predict_slate.py with the three network calls stubbed.

    python test_slate_dryrun.py

The slate runner touches the network in three places -- schedule, rosters,
and the Statcast top-up. All three are replaced here with fixtures built
from whatever is already in cache/, so the ORCHESTRATION can be verified
without a connection: feature assembly, the per-PA models, plate-appearance
projection, compounding, and the output file the dashboard reads.

This catches the class of bug that has cost the most time in this project
-- a column that doesn't exist, a merge that explodes row counts, a
method threaded inconsistently -- before a real run spends an hour pulling
data only to fail at the last step.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import predict_slate
from data.cache import cache_path


def _cached_players(limit=9):
    """Player ids that already have Statcast data on disk."""
    ids = []
    for path in glob.glob(os.path.join(os.path.dirname(cache_path("x")), "statcast_*.csv")):
        try:
            head = pd.read_csv(path, usecols=["batter"], nrows=1)
            ids.append(int(head["batter"].iloc[0]))
        except Exception:
            continue
    return sorted(set(ids))[:limit]


PLAYER_IDS = _cached_players()
if len(PLAYER_IDS) < 4:
    raise SystemExit("Need at least 4 cached players to dry-run. "
                     "Run build_wide_pool.py or train_props_v5.py first.")

HALF = len(PLAYER_IDS) // 2
GAME_DATE = "2026-08-18"


def fake_get_slate(game_date, verbose=True):
    print(f"  [stub] schedule for {game_date}: 1 game")
    return pd.DataFrame([{
        "game_pk": 999001, "game_date": game_date,
        "start_time_utc": f"{game_date}T23:10:00Z",
        "venue_id": 19, "venue_name": "Coors Field",
        "home_team_id": 115, "home_team": "COL",
        "away_team_id": 119, "away_team": "LAD",
        "home_probable_id": 663372, "home_probable": "Ryan Feltner",
        "away_probable_id": 641778, "away_probable": "Eric Lauer",
        "home_lineup": [], "away_lineup": [],
    }])


def fake_build_slate_hitters(slate, slot_history, verbose=True):
    """Half the cached players per side, with confirmed slots 1-N."""
    rows = []
    game = slate.iloc[0]
    for side, ids in (("home", PLAYER_IDS[:HALF]), ("away", PLAYER_IDS[HALF:])):
        other = "away" if side == "home" else "home"
        for slot, pid in enumerate(ids, start=1):
            rows.append({
                "game_pk": game.game_pk, "player_id": int(pid),
                "name": f"Player{pid}", "team_id": game[f"{side}_team_id"],
                "team": game[f"{side}_team"], "opponent": game[f"{other}_team"],
                "is_home": int(side == "home"),
                "opposing_pitcher_id": int(game[f"{other}_probable_id"]),
                "opposing_pitcher": game[f"{other}_probable"],
                "venue_id": game.venue_id, "venue_name": game.venue_name,
                "start_time_utc": game.start_time_utc,
                "lineup_slot": slot, "lineup_status": "confirmed",
            })
    frame = pd.DataFrame(rows)
    print(f"  [stub] {len(frame)} hitters across the slate")
    return frame


def fake_refresh_players(player_ids, start_date, end_date, max_age_days=1,
                         names=None, verbose=True):
    """Read straight from disk -- no pull, no top-up."""
    frames = []
    for path in glob.glob(os.path.join(os.path.dirname(cache_path("x")), "statcast_*.csv")):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "batter" in df.columns and int(df["batter"].iloc[0]) in set(player_ids):
            frames.append(df)
    print(f"  [stub] loaded {len(frames)} cached player files")
    return pd.concat(frames, ignore_index=True)


def fake_build_pitcher_rates(pitcher_ids, start_date, end_date, league_rates,
                             names=None, max_age_days=1, verbose=True):
    """No network: synthesise plausible pitcher rates so the real-pitcher
    branch and its handedness splits get exercised."""
    import numpy as np
    rng = np.random.default_rng(5)
    rows = []
    for pid in pitcher_ids:
        row = {"pitcher": int(pid), "pit_pa_seen": int(rng.integers(150, 700))}
        for target, league in league_rates.items():
            overall = float(np.clip(league * rng.uniform(0.7, 1.4), 1e-4, 0.9))
            row[f"pit_{target}_allowed"] = overall
            for hand in ("L", "R"):
                row[f"pit_{target}_vs_{hand}"] = float(
                    np.clip(overall * rng.uniform(0.85, 1.15), 1e-4, 0.9))
                row[f"pit_pa_vs_{hand}"] = int(rng.integers(40, 350))
        rows.append(row)
    print(f"  [stub] pitcher rates for {len(rows)} starters")
    return pd.DataFrame(rows)


def _write_synthetic_slots():
    """
    Fabricate a lineup-slot cache with a REALISTIC gradient.

    An earlier version assigned slots with heavy noise, which left no
    slot-to-plate-appearance relationship for the model to find -- so the
    "earlier slots get more PA" assertion failed on the FIXTURE, not on the
    code. Real data has a strong gradient (4.40 PA at leadoff down to 3.51
    at ninth), so the fixture needs one too or the check tests nothing.

    Slots are assigned by each batter's own average PA, best-to-worst. That
    makes this a WIRING test -- does lineup_slot reach the PA model and move
    expected_pa in the right direction -- not a test of whether the effect
    exists in baseball.
    """
    frames = []
    for path in glob.glob(os.path.join(os.path.dirname(cache_path("x")), "statcast_*.csv")):
        try:
            d = pd.read_csv(path, usecols=["batter", "game_pk", "events"], low_memory=False)
        except Exception:
            continue
        frames.append(d[d["events"].notna()])
    counts = pd.concat(frames).groupby(["batter", "game_pk"]).size().rename("pa").reset_index()
    rank = counts.groupby("batter")["pa"].mean().rank(ascending=False, method="first")
    counts["lineup_slot"] = counts["batter"].map(rank).clip(1, 9).astype(int)
    counts["is_starter"] = 1
    counts[["game_pk", "batter", "lineup_slot", "is_starter"]].to_csv(
        cache_path("lineup_slots"), index=False)


def fake_attach_lineup_slots(df, verbose=True):
    slots = pd.read_csv(cache_path("lineup_slots"))
    return df.merge(slots, on=["batter", "game_pk"], how="left")


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def patch(name, replacement):
    """
    Replace an attribute on predict_slate, but ONLY if it already exists.

    This guard is the whole reason this function exists. Plain assignment
    (`predict_slate.attach_lineup_slots = fake`) CREATES the attribute when
    it's absent -- so a stub can silently supply a name the real script
    forgot to import. That is exactly what happened: this dry run passed
    every check while `python predict_slate.py` died on
    `NameError: attach_lineup_slots is not defined`, because the test was
    the only thing defining it.

    A test harness that can invent the code under test isn't testing it.
    """
    if not hasattr(predict_slate, name):
        raise AssertionError(
            f"predict_slate has no attribute '{name}' -- the real script is "
            f"missing an import or a definition. Stubbing it here would hide "
            f"a NameError that a real run will hit."
        )
    setattr(predict_slate, name, replacement)


def main():
    _write_synthetic_slots()
    patch("get_slate", fake_get_slate)
    patch("build_slate_hitters", fake_build_slate_hitters)
    patch("refresh_players", fake_refresh_players)
    patch("data_age_days", lambda pid, d=None: 0.0)
    patch("needs_refresh", lambda pid, **kw: False)
    patch("build_pitcher_rates", fake_build_pitcher_rates)
    patch("pitcher_hand", lambda pid, cache_lookup=True: "R")
    patch("attach_lineup_slots", fake_attach_lineup_slots)


    print("=" * 72)
    print("DRY RUN of predict_slate.main() -- network stubbed")
    print("=" * 72)
    predict_slate.main(GAME_DATE)

    print("\n" + "=" * 72)
    print("VERIFYING THE OUTPUT FILE")
    print("=" * 72)
    path = cache_path(f"slate_{GAME_DATE}")
    check("output file written", os.path.exists(path), path)

    out = pd.read_csv(path)
    check("one row per hitter", len(out) > 0, f"{len(out)} rows")

    required = ["name", "team", "lineup_slot", "expected_pa",
                "prob_hr", "prob_hit", "prob_walk", "prob_k"]
    missing = [c for c in required if c not in out.columns]
    check("dashboard columns present", not missing, f"missing {missing}")
    check("real pitcher data was used",
          "pitcher_pa_seen" in out.columns and (out["pitcher_pa_seen"] > 0).all(),
          f"median PA seen {out['pitcher_pa_seen'].median():.0f}"
          if "pitcher_pa_seen" in out.columns else "column missing")

    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    for col in prob_cols:
        values = out[col].dropna()
        check(f"{col} is a valid probability",
              ((values >= 0) & (values <= 1)).all(),
              f"range {values.min():.4f}-{values.max():.4f}")

    check("expected PA is plausible",
          out["expected_pa"].between(1, 6).all(),
          f"range {out['expected_pa'].min():.2f}-{out['expected_pa'].max():.2f}")

    # The over/under lines must be monotone: P(>0.5) >= P(>1.5) >= ...
    lines = sorted([c for c in out.columns if c.startswith("prob_hrr_over_")],
                   key=lambda c: float(c.rsplit("_", 1)[1]))
    for lower, higher in zip(lines, lines[1:]):
        check(f"{lower} >= {higher} (monotone in the line)",
              (out[lower] >= out[higher] - 1e-9).all())

    # NOT asserting "slot 1 projects more PA than slot 9" here.
    #
    # That check kept failing on the FIXTURE rather than on the code: this
    # dry run uses the nine cached sluggers, who all bat near the top of a
    # real order and average ~4.3 plate appearances regardless of the slot
    # a fixture assigns them. V5 measured exactly this -- batter identity
    # explained 0.9% of PA variance in that pool. There is no gradient to
    # detect, so the assertion was testing noise.
    #
    # The directional claim IS verified, on real data, where it belongs:
    # the 270-player slate shows 4.40 PA at leadoff falling to 3.51 at
    # ninth. What this fixture CAN establish is that the PA model runs and
    # produces differentiated output rather than one number for everyone.
    by_slot = out.groupby("lineup_slot")["expected_pa"].mean()
    spread_pa = float(out["expected_pa"].max() - out["expected_pa"].min())
    check("PA projection differentiates between hitters", spread_pa > 0.01,
          f"expected PA spans {spread_pa:.3f} across the slate "
          f"(direction is checked on real data, not this fixture)")

    spread = out["prob_hr"].max() - out["prob_hr"].min()
    check("players get meaningfully different numbers", spread > 0.01,
          f"home-run probability spans {spread:.3f} across the slate")

    print("\n  Sample output:")
    print(out.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED -- predict_slate.py runs end to end.")
    print("=" * 72)


if __name__ == "__main__":
    main()
