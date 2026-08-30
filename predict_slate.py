"""
Run the whole engine for one day's games.

    python predict_slate.py              # tomorrow
    python predict_slate.py 2026-08-18   # a specific date

WHAT IT DOES, IN ORDER
----------------------
 1. Pull the day's schedule: games, venues, probable starters, and the
    official lineups if they're posted yet.
 2. Work out who is batting -- confirmed lineup where available, otherwise
    projected from each team's recent boxscores.
 3. Top up Statcast for every one of those hitters, pulling ONLY games
    since each player's last cached date.
 4. Rebuild per-plate-appearance rates over everyone in the cache.
 5. Predict tonight's per-PA probabilities for each slate hitter against
    his actual opposing starter, in his actual park.
 6. Project his plate appearances from his lineup slot.
 7. Compound into every prop: home run, hit, walk, strikeout, and the
    hits+runs+RBI over/under lines.
 8. Write cache/slate_{date}.csv for the dashboard.

Then:   streamlit run dashboard.py

WHY THE NUMBERS SPREAD OUT NOW
------------------------------
A weak hitter and a strong one should not get similar numbers, and until
the pool widened they did -- nine elite sluggers gave the model nothing to
separate. With a full slate the inputs genuinely differ: shrunk career
rates differ, lineup slots differ (4.40 plate appearances at leadoff
versus 3.51 at ninth), parks differ, and handedness matchups differ. A
number-nine hitter in a pitcher's park facing a tough right-hander SHOULD
come out near the floor.

WHAT THIS IS AND ISN'T
----------------------
These are calibrated probabilities, not confident calls. On held-out data
the engine beats "just quote the base rate" by a Brier skill of roughly
+0.018 on the hits+runs+RBI line and +0.028 on home runs. That is a real
edge and a small one. The reliability tables say the printed percentages
can be read at face value; they do not say any single one will be right.

FIRST RUN IS SLOW. Every hitter on the slate who isn't cached needs a full
Statcast pull. After that, top-ups are seconds.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import START_DATE
from data.schedule import get_slate, tomorrow
from data.roster import build_slate_hitters
from data.refresh import refresh_players, data_age_days, needs_refresh
from data.pitcher_data import (
    build_pitcher_rates, pitcher_hand, measure_bullpen_rates,
    blend_with_bullpen,
)
from data.lineup_slots import get_lineup_slots, attach_lineup_slots
from data.park_factors import get_park_factor
from features.pa_table import build_pa_table, build_game_totals, per_pa_contribution_distribution
from features.rate_features import (
    add_batter_rolling_rates, add_pitcher_rolling_rates, add_matchup_rates,
    rate_feature_cols, RATE_TARGETS, log5,
)
from features.form_features import add_form_deviation
from features.rbi_features import (
    add_base_state, add_lineup_obp_context, train_runner_model,
    train_rbi_model, predict_rbi_per_pa,
)
from features.run_features import (
    RunScoringModel, add_lineup_obp_behind, build_run_training_frame,
)
from data.batting_lines import get_batting_lines
from features.bases_features import (
    measure_bases_per_hit, expected_total_bases_per_pa,
)
from features.pa_projection import (
    prepare_pa_training_frame, train_pa_model, predict_pa_distributions, PA_FEATURES,
    enforce_slot_monotonicity,
)
from model.compound import (
    prob_at_least_one, prob_over_line, marginalise_over_pa,
    scale_contribution_distribution, fit_game_dispersion,
    compound_count_distribution_overdispersed, empirical_pa_distribution,
)
from train_props_v5 import (
    train_rate_model, MODEL_C, HRR_LINES, TB_LINES, HIT_LINES,
)
from data.cache import cache_path

MAX_DATA_AGE_DAYS = 0   # 0 = re-check once per calendar day
# Pull real Statcast for the ~30 probable starters on this slate. Each call
# returns EVERY batter that pitcher faced, not just the ones in our hitter
# pool -- roughly 600 plate appearances instead of ~20. Cached per pitcher,
# so the library grows a slate at a time rather than needing a bulk pull.
USE_REAL_PITCHER_DATA = True
SHAPE_METHOD = "frequency"   # chosen by select_tilt_method on the backtest


def preserve_committed_rows(fresh, existing_path, now_utc, force):
    """
    Keep rows for games that have already started; take the rest from `fresh`.

    A prediction for a game already in progress is evidence -- it was made
    without knowing the outcome. A prediction made NOW for that same game is
    not, because the rolling rates behind it may already contain the game's
    own results. So on a re-run the old rows win for games underway, and the
    new rows win for everything still to come.

    Games in the previous file that are missing from `fresh` are kept too. A
    postponement or a lineup that stopped being published should not silently
    delete a prediction that was committed to hours earlier.
    """
    if force or not os.path.exists(existing_path):
        return fresh
    try:
        previous = pd.read_csv(existing_path)
    except Exception as exc:
        print(f"\n  Could not read the previous slate file ({exc}).")
        print(f"  Writing fresh predictions for every game.")
        return fresh
    if previous.empty or "start_time_utc" not in previous.columns:
        return fresh

    # pd.Timestamp.utcnow() already carries UTC, so tz_localize on it raises
    # -- and it would raise at SAVE time, after the whole ten-minute run.
    # Normalise instead of assuming which of the two it is.
    now = pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")

    started = pd.to_datetime(previous["start_time_utc"], utc=True,
                             errors="coerce") <= now
    committed = previous[started.fillna(False)]
    if committed.empty:
        print(f"\n  No game in the previous file had started yet -- "
              f"refreshing all {fresh['game_pk'].nunique()} games.")
        return fresh

    # A column set that does not match means the two halves of this file were
    # produced by different versions of the code. Concatenating them anyway
    # yields a slate where some games have prop columns and others have NaN,
    # which every downstream reader will treat as "no prediction" rather than
    # "written by older code". Say so rather than letting it pass.
    missing = set(fresh.columns) - set(committed.columns)
    extra = set(committed.columns) - set(fresh.columns)
    if missing or extra:
        print(f"\n  NOTE: the preserved rows were written by a different "
              f"version of this script.")
        if missing:
            print(f"  They lack: {', '.join(sorted(missing))}")
        if extra:
            print(f"  They carry extra: {', '.join(sorted(extra))}")
        print(f"  Those games keep their original predictions and will score "
              f"on the props they have.")

    keep_pks = set(committed["game_pk"].unique())
    refreshed = fresh[~fresh["game_pk"].isin(keep_pks)]
    combined = pd.concat([committed, refreshed], ignore_index=True)

    print(f"\n  Preserved {len(committed)} hitters across {len(keep_pks)} game(s) "
          f"already underway,")
    print(f"  written earlier and still clean. Refreshed "
          f"{refreshed['game_pk'].nunique()} game(s) not yet started.")
    print(f"  Use --force to overwrite the preserved rows as well.")
    return combined


def main(game_date: str = None):
    game_date = game_date or tomorrow()
    print("=" * 72)
    print(f"SLATE: {game_date}")
    print("=" * 72)

    # ---- 0. Protect a prediction that has already earned its keep ----
    #
    # A slate file written BEFORE first pitch is a scientific record: it is
    # the only evidence that the model committed to these numbers without
    # knowing the outcomes. Regenerating it destroys that, and the damage
    # is silent -- the new file looks identical in every respect except
    # that its predictions were made with hindsight.
    #
    # The old version of this only printed a warning, only for past dates,
    # and then carried on. That is backwards. The dangerous moment is
    # TONIGHT, a few minutes after a good pre-game run, when re-running to
    # try out a code change quietly overwrites the record. By tomorrow --
    # when the date finally counts as "past" -- it is already gone.
    #
    # So: refuse, rather than warn. `--force` is there for when overwriting
    # is genuinely what you want.
    # Re-running is NORMAL, not an accident to be prevented.
    #
    # Lineups are posted a couple of hours before each game, so on a fifteen
    # game slate the cards trickle in across the afternoon. Getting confirmed
    # lineups for the late games means re-running, and the old behaviour --
    # refuse outright, or overwrite everything with --force -- made that a
    # choice between stale lineups and destroying the morning's clean
    # predictions for games already being played.
    #
    # Neither is necessary. A prediction is only spoiled for games that have
    # already started; games still to come are untouched. So a re-run now
    # PRESERVES rows for games underway and refreshes only the rest. Running
    # it five times an afternoon accumulates clean coverage instead of
    # trading it away.
    #
    # --force still exists and now means what it says: throw away the
    # previous file entirely, including its committed rows.
    force = "--force" in sys.argv
    existing_path = cache_path(f"slate_{game_date}")

    # A past date is a trap worth naming even when no file exists yet.
    # Re-running for a date whose games are finished pulls those results
    # into the player caches FIRST, so the rolling rates would include the
    # very outcomes being predicted. The numbers would look excellent and
    # mean nothing.
    if pd.Timestamp(game_date).normalize() < pd.Timestamp.today().normalize():
        print("\n  WARNING: this date is in the past.")
        print("  The rolling rates now include these games' outcomes, so any")
        print("  prediction made here has seen the answers. Not scoreable.")
        print("  Press Ctrl+C now unless you specifically want that.\n")

    # ---- 1. Schedule -------------------------------------------------
    slate = get_slate(game_date)
    if slate.empty:
        print("Nothing to predict.")
        return

    # ---- 2. Who's batting --------------------------------------------
    # Recent lineup slots, used to project a lineup where none is posted.
    print("\nLoading recent lineup history (for projecting unposted lineups)...")
    slot_history = pd.DataFrame()
    try:
        cached_slots = pd.read_csv(cache_path("lineup_slots"))
        slot_history = cached_slots[cached_slots["lineup_slot"] > 0]
    except Exception:
        print("  No cached lineup history -- projections will be limited.")

    print("\nBuilding slate hitter list...")
    hitters = build_slate_hitters(slate, slot_history)
    if hitters.empty:
        print("No hitters resolved. Rosters may be unavailable.")
        return

    # ---- 3. Refresh player data --------------------------------------
    player_ids = sorted(hitters["player_id"].unique())
    names = dict(zip(hitters["player_id"], hitters["name"]))
    # Staleness is "have we CHECKED today", not "how old is his newest
    # game" -- see data/refresh.py. Checked against today, never against
    # the slate date, which is in the future and can never be satisfied.
    today = pd.Timestamp.today().normalize()
    stale = [p for p in player_ids
             if needs_refresh(p, as_of=today, max_age_days=MAX_DATA_AGE_DAYS)]
    print(f"\nRefreshing Statcast for {len(player_ids)} hitters "
          f"({len(stale)} not yet checked today)...")
    if stale:
        print("  First run for a new player is a full history pull and is slow.")
        print("  Re-running later today costs zero network calls.")
    combined = refresh_players(player_ids, START_DATE, game_date,
                               max_age_days=MAX_DATA_AGE_DAYS, names=names)

    # ---- 3b. Real rates for tonight's starters -----------------------
    pitcher_rates_real = pd.DataFrame()
    starters = pd.DataFrame(columns=["pid", "pname"])
    if USE_REAL_PITCHER_DATA:
        starters = pd.concat([
            slate[["home_probable_id", "home_probable"]].rename(
                columns={"home_probable_id": "pid", "home_probable": "pname"}),
            slate[["away_probable_id", "away_probable"]].rename(
                columns={"away_probable_id": "pid", "away_probable": "pname"}),
        ]).dropna(subset=["pid"])
        starters = starters[starters["pname"] != "TBD"]
        if not starters.empty:
            print(f"\nPulling Statcast for {len(starters)} probable starters...")
            print("  (each call covers every batter he faced, not just our pool)")

    # ---- 4. Rebuild rates over everything we know --------------------
    print("\nBuilding plate-appearance table and rates...")
    pa = build_pa_table(combined)
    print(f"  {len(pa):,} plate appearances, {pa['batter'].nunique()} batters, "
          f"through {pa['game_date'].max().date()}")
    pa = add_batter_rolling_rates(pa)
    pa = add_pitcher_rolling_rates(pa)
    pa = add_matchup_rates(pa)

    # Form deviation -- how far a hitter is running from his OWN baseline.
    #
    # This is a marker, not a feature. It is deliberately NOT added to
    # rate_feature_cols(), and that omission is load-bearing rather than an
    # oversight: `needed` below is built from rate_feature_cols(), and it
    # drives both `train_pa.dropna(subset=needed)` and
    # `frame.dropna(subset=needed)`. Putting form columns in there would
    # silently DELETE from the slate every hitter without enough history to
    # score one -- a display marker quietly removing predictions.
    #
    # Keeping it out of that one function is therefore what guarantees the
    # marker cannot reach the model, enforced by the code rather than by
    # anyone remembering.
    pa = add_form_deviation(pa)
    pa["park_hr_factor"] = pa["home_team"].apply(get_park_factor) / 100.0

    game_totals = build_game_totals(pa)

    # Attach historical batting-order slots. This was MISSING, and its
    # absence silently disabled two models at once: the plate-appearance
    # projection (which needs slots to train) and the two-stage RBI model
    # (which needs them for lineup context). Both had `except: fall back`
    # handlers, so the run completed and printed plausible numbers while
    # quietly using the weaker path for everything.
    print("Attaching historical lineup slots...")
    try:
        game_totals = attach_lineup_slots(game_totals)
        matched = int(game_totals["lineup_slot"].notna().sum())
        print(f"  {matched:,} of {len(game_totals):,} player-games have a slot "
              f"({matched / max(len(game_totals), 1):.0%}).")
        if matched < 0.3 * len(game_totals):
            print("  WARNING: low coverage. The PA and RBI models will fall")
            print("  back to weaker estimates for most hitters.")
    except Exception as e:
        print(f"  Lineup slots unavailable ({type(e).__name__}: {e}).")
        print("  The PA projection and RBI model will both use fallbacks.")
        game_totals["lineup_slot"] = np.nan
        game_totals["is_starter"] = 0

    feature_groups = rate_feature_cols()
    needed = sorted({c for cols in feature_groups.values() for c in cols})
    train_pa = pa.dropna(subset=needed).reset_index(drop=True)

    print(f"  Training per-PA models on {len(train_pa):,} plate appearances...")
    models = {t: (train_rate_model(train_pa, feature_groups[t], t), feature_groups[t])
              for t in RATE_TARGETS}

    # ---- 5. Tonight's feature row per hitter -------------------------
    # Start from each batter's most recent plate appearance -- that row
    # carries his current shrunk rolling rates -- then overwrite the
    # context columns with tonight's opponent, park and handedness.
    print("\nAssembling tonight's feature rows...")
    latest = pa.sort_values("game_date").groupby("batter").tail(1).set_index("batter")

    pitcher_rates = {}
    for target in RATE_TARGETS:
        col = f"pit_{target}_allowed"
        if col in pa.columns:
            pitcher_rates[target] = pa.groupby("pitcher")[col].last()

    league = {t: float(pa[t].mean()) for t in RATE_TARGETS}

    # A starter faces a hitter ~2.3 of his ~4.3 plate appearances. Applying
    # his rate to all of them charges him for ~47% he never throws, which
    # is near-harmless for an average starter and worth up to 5 percentage
    # points of home-run probability against an extreme one. Measured from
    # this pool rather than assumed.
    starter_share, bullpen_rates = measure_bullpen_rates(pa)
    print(f"  Bullpen exposure: starter throws {starter_share:.1%} of a "
          f"hitter's plate appearances; the rest use league reliever rates.")

    if USE_REAL_PITCHER_DATA and not starters.empty:
        pitcher_rates_real = build_pitcher_rates(
            starters["pid"].astype(int).tolist(), START_DATE, game_date,
            league_rates=league,
            names=dict(zip(starters["pid"].astype(int), starters["pname"])),
        ).set_index("pitcher")

    rows = []
    for hitter in hitters.itertuples():
        if hitter.player_id not in latest.index:
            continue
        row = latest.loc[hitter.player_id].copy()

        row["is_home"] = hitter.is_home
        row["park_hr_factor"] = get_park_factor(_park_team(hitter)) / 100.0

        # Throwing hand: from the pitcher's own cached data when we have
        # it, otherwise from how he's thrown to our hitter pool.
        pid = hitter.opposing_pitcher_id
        has_real = pid is not None and pid in pitcher_rates_real.index
        p_throws = pitcher_hand(int(pid)) if has_real else _pitcher_hand(pa, pid)
        batter_stand = row.get("stand", "R")
        row["platoon_edge"] = int(batter_stand != p_throws)

        for target in RATE_TARGETS:
            allowed_col = f"pit_{target}_allowed"
            if has_real:
                # Prefer the handedness split -- a left-handed starter is a
                # materially different opponent for a left-handed hitter,
                # and this split is shrunk toward the pitcher's own overall
                # rate rather than toward league, so a thin split degrades
                # to "himself" instead of to "average pitcher".
                split_col = f"pit_{target}_vs_{batter_stand}"
                allowed = float(pitcher_rates_real.loc[pid].get(
                    split_col, pitcher_rates_real.loc[pid][allowed_col]))
                # Weight by how much of the game he actually pitches.
                allowed = blend_with_bullpen(
                    allowed, bullpen_rates.get(target, league[target]),
                    starter_share)
            elif (pid is not None and target in pitcher_rates
                    and pid in pitcher_rates[target].index):
                allowed = blend_with_bullpen(
                    float(pitcher_rates[target].loc[pid]),
                    bullpen_rates.get(target, league[target]), starter_share)
            else:
                # No starter named (TBD). The whole game is effectively
                # "unknown pitching", so league average is already right --
                # blending would double-count the uncertainty.
                allowed = league[target]
            row[allowed_col] = allowed
            row[f"matchup_{target}"] = log5(
                row[f"bat_{target}_career"], allowed, league[target]
            )

        rows.append({**hitter._asdict(), **{c: row.get(c) for c in needed},
                     "stand": row.get("stand", "R"),
                     # Carried explicitly, never via `needed` -- see the
                     # add_form_deviation call above for why that matters.
                     "form_z": row.get("form_z"),
                     "form_state": row.get("form_state", "Unknown"),
                     "pitcher_pa_seen": (int(pitcher_rates_real.loc[pid, "pit_pa_seen"])
                                         if has_real else 0)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No slate hitters had usable history. Nothing to predict.")
        return
    frame = frame.dropna(subset=needed).reset_index(drop=True)
    print(f"  {len(frame)} hitters have enough history to score.")

    if "form_state" in frame.columns:
        counts = frame["form_state"].value_counts()
        hot, cold = int(counts.get("Hot", 0)), int(counts.get("Cold", 0))
        unknown = int(counts.get("Unknown", 0))
        print(f"  Form marker: {hot} hot, {cold} cold, "
              f"{len(frame) - hot - cold - unknown} normal, {unknown} unknown.")
        # About a dozen of these are expected from noise alone on a
        # 270-hitter slate -- the threshold was set from a simulation of
        # players whose true rates never move. A night with 60 flagged is
        # not a hot slate, it is a bug.
        if hot + cold > 0.25 * max(len(frame), 1):
            print(f"  WARNING: {hot + cold} of {len(frame)} flagged. That is "
                  f"far above the ~5% noise floor -- check form_features.")

    # ---- 6. Per-PA probabilities and PA projection -------------------
    for target in RATE_TARGETS:
        model, cols = models[target]
        frame[f"p_{target}"] = model.predict_proba(frame[cols])[:, 1]

    pa_dists = _project_pa(game_totals, frame)
    # The batting order is a sequence, and the model scores each hitter
    # alone. See features/pa_projection.enforce_slot_monotonicity.
    pa_dists = enforce_slot_monotonicity(frame, pa_dists)

    # ---- 7. Compound -------------------------------------------------
    print("Compounding to game-level probabilities...")

    # --- Runs scored: official boxscore lines, then the scoring model ---
    #
    # Runs cannot be derived from Statcast. The only trace of who crossed
    # the plate is free text in the play description, which names the
    # runner but not his player id, so the boxscore is the source. See
    # data/batting_lines.py.
    print("  Loading official batting lines (hits / runs / RBI)...")
    obp_series = (pa.groupby("batter")["is_hit"].mean()
                  + pa.groupby("batter")["is_walk"].mean())
    league_obp = float(pa["is_hit"].mean() + pa["is_walk"].mean())

    run_model = RunScoringModel(league_obp=league_obp)
    train_games = game_totals.copy()
    try:
        lines = get_batting_lines(game_totals["game_pk"].unique(), verbose=True)
        if lines.empty:
            raise ValueError("no batting lines cached -- run fetch_batting_lines.py")
        train_games = build_run_training_frame(game_totals, lines)
        coverage = float(train_games["runs_official"].notna().mean())
        print(f"  Official lines matched for {coverage:.1%} of player-games.")
        if coverage < 0.9:
            print("  WARNING: the rest have no runs at all, which understates")
            print("  every H+R+RBI line. Run: python fetch_batting_lines.py")
        train_games = add_lineup_obp_behind(train_games, obp_series, league_obp)
        run_model = RunScoringModel.fit(train_games, obp_series, league_obp)
    except Exception as e:
        print(f"  Runs: official lines unavailable ({type(e).__name__}: {e}).")
        print("  Falling back to measured league constants. Fix this before")
        print("  trusting the H+R+RBI lines -- run fetch_batting_lines.py.")

    # The game-level target the dispersion is fitted against must be the
    # OFFICIAL total, not a Statcast reconstruction, or the two halves of
    # the calibration are measuring different quantities.
    if "hrr" not in train_games.columns or train_games["hrr"].isna().all():
        train_games["hrr"] = train_games["hrr_certain"]
    train_games = train_games.dropna(subset=["hrr"])

    # The per-PA shape, with the run folded in analytically at the league
    # scoring rate -- see per_pa_contribution_distribution.
    base_dist = per_pa_contribution_distribution(
        pa, "hrr_certain", score_prob=run_model.league_prob)
    dispersion_sd = fit_game_dispersion(train_games, base_dist, value_col="hrr")

    # --- RBI: two-stage lineup model, falling back to the old constant --
    # Measured on held-out games (eval_rbi.py): the per-batter constant
    # over-predicts RBI by 6.4% (0.1267 vs 0.1191 actual), which is ~0.032
    # RBI per game of pure bias flowing into every H+R+RBI line. The
    # two-stage model lands at 0.1194 -- essentially unbiased.
    league_rbi = float(pa["rbi"].mean())
    rbi_rate = pa.groupby("batter")["rbi"].mean()
    frame["p_rbi"] = frame["player_id"].map(rbi_rate).fillna(league_rbi)

    try:
        pa_state = add_base_state(pa)
        obp = (pa.groupby("batter")["is_hit"].mean()
               + pa.groupby("batter")["is_walk"].mean())
        league_obp = float(pa["is_hit"].mean() + pa["is_walk"].mean())

        runner_context = pa_state.groupby(["batter", "game_pk"]).agg(
            runners_on=("runners_on", "mean")).reset_index()
        game_ctx = game_totals.merge(runner_context, on=["batter", "game_pk"],
                                     how="inner")
        game_ctx = add_lineup_obp_context(game_ctx, obp, league_obp)

        pa_state["bat_is_hr_career"] = pa_state["batter"].map(
            pa.groupby("batter")["is_hr"].mean())
        pa_state["bat_is_hit_career"] = pa_state["batter"].map(
            pa.groupby("batter")["is_hit"].mean())

        runner_model = train_runner_model(game_ctx)
        rbi_model = train_rbi_model(pa_state)

        if runner_model is not None and rbi_model is not None:
            slate_ctx = frame.copy()
            slate_ctx["batter"] = slate_ctx["player_id"]
            slate_ctx["game_pk"] = slate_ctx["game_pk"]
            slate_ctx = add_lineup_obp_context(slate_ctx, obp, league_obp)
            slate_ctx["bat_is_hr_career"] = slate_ctx["player_id"].map(
                pa.groupby("batter")["is_hr"].mean()).fillna(0.0)
            slate_ctx["bat_is_hit_career"] = slate_ctx["player_id"].map(
                pa.groupby("batter")["is_hit"].mean()).fillna(0.0)
            slate_ctx["lineup_slot"] = pd.to_numeric(
                slate_ctx["lineup_slot"], errors="coerce")

            modelled = predict_rbi_per_pa(runner_model, rbi_model,
                                          slate_ctx, league_rbi)
            known_slot = slate_ctx["lineup_slot"].notna().to_numpy()

            # Capture the constant BEFORE overwriting it. The first version
            # printed frame["p_rbi"].mean() after the assignment, so it
            # compared the new value against itself and always reported the
            # two as identical -- a message that could never show a change.
            constant_mean = float(frame.loc[known_slot, "p_rbi"].mean())
            frame.loc[known_slot, "p_rbi"] = modelled[known_slot]
            print(f"  RBI: two-stage lineup model used for "
                  f"{int(known_slot.sum())} of {len(frame)} hitters "
                  f"({modelled[known_slot].mean():.4f}/PA vs "
                  f"{constant_mean:.4f} from the per-batter constant, "
                  f"{modelled[known_slot].mean() - constant_mean:+.4f}).")
        else:
            print("  RBI: not enough data for the two-stage model -- "
                  "using per-batter constants.")
    except Exception as e:
        # Report the actual reason. An earlier version printed only the
        # exception TYPE, which turned "lineup_slot column is missing" into
        # an uninformative "KeyError" and cost a debugging round trip.
        print(f"  RBI: two-stage model unavailable ({type(e).__name__}: {e}) -- "
              f"using per-batter constants.")
    # --- Runs per PA ----------------------------------------------------
    #
    # Routed through REACHING BASE rather than estimated directly, which is
    # what makes it respond to tonight's matchup: a hitter facing a starter
    # who suppresses his on-base rate scores less, automatically, because
    # p_reach falls. A per-batter runs-per-game constant would sit there
    # unchanged no matter who was pitching.
    frame_ctx = frame.copy()
    frame_ctx["batter"] = frame_ctx["player_id"]
    frame_ctx["lineup_slot"] = pd.to_numeric(frame_ctx["lineup_slot"],
                                             errors="coerce")
    frame_ctx = add_lineup_obp_behind(frame_ctx, obp_series, league_obp)

    frame["p_run"] = run_model.runs_per_pa(
        frame_ctx,
        p_hr=frame["p_is_hr"].to_numpy(),
        p_reach=(frame["p_is_hit"] + frame["p_is_walk"]).to_numpy(),
    )
    known_slot = frame_ctx["lineup_slot"].notna().to_numpy()
    print(f"  Runs: {frame['p_run'].mean():.4f} expected per plate appearance "
          f"({int(known_slot.sum())} of {len(frame)} hitters have a slot).")

    # HITS + RUNS + RBI. Walks are NOT in this prop -- a walk contributes
    # only by putting the batter in position to score, which is already
    # counted inside p_run via p_reach.
    frame["expected_hrr_per_pa"] = (
        frame["p_is_hit"] + frame["p_run"] + frame["p_rbi"]
    )

    for target, label in [("is_hr", "hr"), ("is_hit", "hit"),
                          ("is_walk", "walk"), ("is_k", "k")]:
        frame[f"prob_{label}"] = [
            marginalise_over_pa(dist, lambda n, p=prob: float(prob_at_least_one(p, n)))
            for dist, prob in zip(pa_dists, frame[f"p_{target}"])
        ]

    for line in HRR_LINES:
        values = []
        for dist, mean in zip(pa_dists, frame["expected_hrr_per_pa"]):
            shaped = scale_contribution_distribution(base_dist, float(mean),
                                                     method=SHAPE_METHOD)
            values.append(marginalise_over_pa(
                dist,
                lambda n, d=shaped: prob_over_line(
                    compound_count_distribution_overdispersed(
                        d, int(n), dispersion_sd, method=SHAPE_METHOD), line),
            ))
        frame[f"prob_hrr_over_{line}"] = values

    # ---- TOTAL BASES and HITS ----------------------------------------
    #
    # Same engine, different contribution distribution. Both are fully
    # settled inside the plate appearance -- a double is two bases in the
    # at bat that produced it -- so neither needs the run-attribution
    # machinery H+R+RBI required. That is the whole reason these two props
    # cost a few lines rather than a file.
    league_bph, bases_per_hit, _ = measure_bases_per_hit(pa)
    frame["bases_per_hit"] = (frame["player_id"].map(bases_per_hit)
                              .fillna(league_bph))
    frame["expected_tb_per_pa"] = expected_total_bases_per_pa(
        frame["p_is_hr"].to_numpy(), frame["p_is_hit"].to_numpy(),
        frame["bases_per_hit"].to_numpy(),
    )

    tb_dist = per_pa_contribution_distribution(pa, "total_bases")
    tb_dispersion = fit_game_dispersion(train_games, tb_dist,
                                        value_col="total_bases")
    hit_dist = per_pa_contribution_distribution(pa, "is_hit")
    hit_dispersion = fit_game_dispersion(train_games, hit_dist,
                                         value_col="hits")
    print(f"  Total bases: {frame['expected_tb_per_pa'].mean():.4f} expected "
          f"per plate appearance (league shape mean "
          f"{float((tb_dist * np.arange(len(tb_dist))).sum()):.4f}).")

    for label, dist, disp, means, lines in [
        ("tb", tb_dist, tb_dispersion, frame["expected_tb_per_pa"], TB_LINES),
        ("hits", hit_dist, hit_dispersion, frame["p_is_hit"], HIT_LINES),
    ]:
        for line in lines:
            values = []
            for pa_dist, mean in zip(pa_dists, means):
                shaped = scale_contribution_distribution(
                    dist, float(mean), method=SHAPE_METHOD)
                values.append(marginalise_over_pa(
                    pa_dist,
                    lambda n, d=shaped, s=disp: prob_over_line(
                        compound_count_distribution_overdispersed(
                            d, int(n), s, method=SHAPE_METHOD), line),
                ))
            frame[f"prob_{label}_over_{line}"] = values

    # Consistency check, free and worth printing. P(hits over 0.5) is
    # computed here by convolving a per-PA hit distribution; prob_hit is
    # computed by an entirely different route, 1-(1-p)^n. They answer the
    # same question, so a gap between them means one of the two paths is
    # wrong -- and a silent disagreement is exactly the kind of thing this
    # project keeps finding after the fact.
    if "prob_hits_over_0.5" in frame.columns:
        gap = float((frame["prob_hits_over_0.5"] - frame["prob_hit"]).abs().mean())
        note = "consistent" if gap < 0.02 else "*** THESE SHOULD AGREE ***"
        print(f"  Cross-check: P(hits>0.5) vs P(at least 1 hit) differ by "
              f"{gap:.4f} on average -- {note}")

    frame["expected_pa"] = [sum(k * v for k, v in d.items()) for d in pa_dists]

    # How much history backs each HITTER.
    #
    # `pitcher_pa_seen` already ships so the dashboard can flag a matchup
    # where the pitcher is unknown. Nothing said the same about the batter,
    # and the asymmetry showed: a hitter with 48 career plate appearances
    # rendered exactly as confidently as one with 3,705.
    #
    # The rates themselves are fine -- empirical-Bayes shrinkage does its
    # job, and the thin hitters on a real slate come out LOWER than the
    # rest (median 0.076 against 0.110 on home runs), not wilder. But
    # "correctly regressed toward league average" and "we know this about
    # him" are different claims, and only one of them was visible.
    batter_pa_counts = pa.groupby("batter").size()
    frame["batter_pa_seen"] = (frame["player_id"].map(batter_pa_counts)
                               .fillna(0).astype(int))
    thin = frame["batter_pa_seen"] < 150
    if thin.any():
        print(f"  {int(thin.sum())} of {len(frame)} hitters have under 150 "
              f"plate appearances of history (min "
              f"{int(frame['batter_pa_seen'].min())}). Their rates are shrunk "
              f"hard toward league average -- read those as 'a typical "
              f"hitter in this spot', not as a read on the player.")

    # Print expected PA by lineup slot. This is the single largest term in
    # every probability below it, and it silently ran on a weaker fallback
    # for several versions. Historical reference from the backtest: 4.40 PA
    # at leadoff falling to 3.51 at ninth. If this table is flat, the PA
    # model is not doing its job whatever the log above says.
    slot_pa = frame.dropna(subset=["lineup_slot"]).groupby("lineup_slot")[
        "expected_pa"].agg(["size", "mean"])
    if len(slot_pa) >= 3:
        print("\n  Expected plate appearances by lineup slot:")
        for slot, row in slot_pa.iterrows():
            print(f"    slot {int(slot)}: {row['mean']:.2f} PA  "
                  f"({int(row['size'])} hitters)")

    # ---- 8. Save ------------------------------------------------------
    out_cols = [
        # player_id is exported so score_slate.py can join predictions to
        # actual outcomes without name matching, which breaks on accents
        # and on two players sharing a name.
        "game_pk", "player_id", "name", "team", "opponent", "is_home", "lineup_slot",
        "lineup_status", "opposing_pitcher", "venue_name", "start_time_utc",
        "expected_pa", "batter_pa_seen", "pitcher_pa_seen",
        # Form marker. Written to the slate so it can be graded later, and
        # read by nothing that produces the probabilities beside it.
        "form_z", "form_state",
        "prob_hr", "prob_hit", "prob_walk", "prob_k",
    ] + [f"prob_hrr_over_{l}" for l in HRR_LINES] \
      + [f"prob_tb_over_{l}" for l in TB_LINES] \
      + [f"prob_hits_over_{l}" for l in HIT_LINES]
    # When this prediction was written. score_slate.py compares it against
    # first pitch: a file written AFTER the games is not an out-of-sample
    # prediction, it's a memory, and it must not be scored as though it
    # were. Stored in UTC so it compares cleanly to the schedule feed.
    # Stamped per ROW, not per file. After a re-run the rows in this file
    # come from different moments -- the noon games from the morning pass,
    # the night games from this one -- and score_slate.py decides row by row
    # whether each was written before its own first pitch. One timestamp for
    # the whole file cannot express that.
    now_utc = pd.Timestamp.utcnow()
    frame["predicted_at_utc"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_cols.append("predicted_at_utc")

    out = frame[[c for c in out_cols if c in frame.columns]].copy()
    out = preserve_committed_rows(out, existing_path, now_utc, force)
    out = out.sort_values(["game_pk", "lineup_slot"], na_position="last")

    path = cache_path(f"slate_{game_date}")
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} predictions to cache/slate_{game_date}.csv")

    print("\nHighest home-run probabilities on the slate:")
    top = out.nlargest(10, "prob_hr")[
        ["name", "team", "opponent", "lineup_slot", "lineup_status",
         "opposing_pitcher", "prob_hr", "prob_hit"]
    ]
    print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nLowest home-run probabilities (the spread a wide pool buys you):")
    print(out.nsmallest(5, "prob_hr")[["name", "team", "lineup_slot", "prob_hr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n  streamlit run dashboard.py")


# Baseball-Reference and MLB's API disagree on a few abbreviations. The
# park-factor table uses the first spelling; the schedule feed may return
# either.
TEAM_ALIASES = {
    "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
    "TBR": "TB", "WSN": "WSH", "ATH": "OAK", "AZ": "ARI",
}


def _park_team(hitter) -> str:
    """
    Which park is this game in?

    The answer is simply the HOME team's park, and the slate already knows
    who is home -- so this reads it off directly rather than trying to
    recognise a venue name.

    That matters: the first version of this matched park-factor keys as
    substrings of the venue name, which quietly almost never fired. "COL"
    is not a substring of "Coors Field", so every game would have fallen
    back to a neutral park factor of 100 while looking like it worked.
    A silent default is worse than a crash, because nothing in the output
    says the feature stopped doing anything.
    """
    home_team = hitter.team if hitter.is_home else hitter.opponent
    if not isinstance(home_team, str):
        return "UNK"
    return TEAM_ALIASES.get(home_team.upper(), home_team.upper())


def _pitcher_hand(pa: pd.DataFrame, pitcher_id):
    """Throwing hand, from any plate appearance this pitcher has against
    the pool. Defaults to right-handed, which is ~70% of starters."""
    if pitcher_id is None or "p_throws" not in pa.columns:
        return "R"
    rows = pa[pa["pitcher"] == pitcher_id]
    if rows.empty:
        return "R"
    return rows["p_throws"].mode().iloc[0]


def _project_pa(game_totals: pd.DataFrame, frame: pd.DataFrame) -> list:
    """
    Plate-appearance distribution per slate hitter.

    Uses the lineup-slot model where a slot is known, and each hitter's own
    historical distribution otherwise. Both feed the same compounding step,
    so a hitter with an unknown slot still gets a number -- just a vaguer
    one.
    """
    # Failures here are REPORTED, not swallowed. A silent fallback to
    # historical PA histograms looks identical in the output to the real
    # lineup-slot model, and the difference is roughly 0.9 plate
    # appearances between the leadoff and ninth slots -- the largest single
    # term in every probability this script prints.
    model = None
    try:
        with_slots = game_totals.copy()
        if "lineup_slot" not in with_slots.columns:
            raise ValueError("game_totals has no lineup_slot column")
        training = prepare_pa_training_frame(with_slots)
        if len(training) < 500:
            print(f"  PA model: only {len(training)} usable rows (need 500) -- "
                  f"using historical PA histograms instead.")
        else:
            model = train_pa_model(training)
            print(f"  PA model trained on {len(training):,} player-games.")
    except Exception as e:
        print(f"  PA model unavailable ({type(e).__name__}: {e}) -- "
              f"using historical PA histograms instead.")

    fallback = {
        pid: empirical_pa_distribution(game_totals, pid)
        for pid in frame["player_id"].unique()
    }

    if model is None:
        return [fallback[pid] for pid in frame["player_id"]]

    rows = frame.copy()
    batter_mean = game_totals.groupby("batter")["pa"].mean()
    rows["batter_pa_mean_20"] = rows["player_id"].map(batter_mean).fillna(
        float(game_totals["pa"].mean()))
    rows["team_pa_mean_20"] = rows["batter_pa_mean_20"]
    rows["lineup_slot"] = pd.to_numeric(rows["lineup_slot"], errors="coerce")
    rows["lineup_slot_sq"] = rows["lineup_slot"] ** 2

    known = rows["lineup_slot"].notna()
    distributions = [fallback[pid] for pid in rows["player_id"]]
    if known.any():
        predicted = predict_pa_distributions(model, rows[known], PA_FEATURES)
        for position, dist in zip(np.flatnonzero(known.to_numpy()), predicted):
            distributions[position] = dist
    return distributions


if __name__ == "__main__":
    # Filter flags out before reading the date, or `predict_slate.py
    # --force` takes "--force" as the game date and fails somewhere much
    # less obvious than here.
    date_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(date_args[0] if date_args else None)
