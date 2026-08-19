"""
V5: one engine, every prop.

    python train_props_v5.py

WHAT THIS REPLACES
------------------
V1-V4 built a separate game-level classifier per question: one for home
runs, one for walks, one abandoned for RBI. Each fought the same problem
-- a game outcome is 4-5 plate appearances stacked up, and stacking is
where almost all the variance comes from. Measured on this data, 98.5% of
the game-to-game variation in "did he homer" is binomial noise from
getting only ~4 tries.

V5 splits the problem where it naturally breaks:

    ESTIMATE   per-PA probability     features/rate_features.py   (statistics + ML)
    COMPOUND   over tonight's PA      model/compound.py           (exact arithmetic)

Consequences worth being explicit about:

  - The models train on ~30,000 plate appearances instead of ~6,700 games.
    Same data pull, 4.5x the rows, and the target is now the repeatable
    event rather than a noisy summary of several.
  - Every prop comes from one engine. "Chance of a home run," "chance of a
    walk," "over 1.5 hits+runs+RBI" are three questions to the same
    distribution, not three models.
  - Over/under lines become answerable AT ALL. A yes/no classifier cannot
    produce P(more than 1.5); a count distribution gives every line at
    once.

WHAT THIS SCRIPT PROVES OR DISPROVES
------------------------------------
For each target it reports, on held-out games:

  PA-LEVEL     how good the rate estimate is, where the sample lives
  GAME-LEVEL   how good the compounded answer is, against the base rate
  HEAD-TO-HEAD the compounded answer versus a direct game-level classifier
               trained on the same information

That last comparison is the honest test of the whole idea. If compounding
doesn't beat direct classification, the restructuring bought nothing and
should be reported as such.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import PLAYERS, START_DATE, END_DATE
from data.loader import load_batter_statcast_cached
from data.park_factors import get_park_factor
from features.pa_table import (
    build_pa_table, build_game_totals, per_pa_contribution_distribution,
)
from features.rate_features import (
    add_batter_rolling_rates, add_pitcher_rolling_rates, add_matchup_rates,
    rate_feature_cols, RATE_TARGETS,
)
from model.compound import (
    prob_at_least_one, compound_count_distribution, prob_over_line,
    empirical_pa_distribution, scale_contribution_distribution,
    marginalise_over_pa, pa_leverage_diagnostic,
    fit_game_dispersion, compound_count_distribution_overdispersed,
    select_tilt_method,
)
from model.hr_v4 import evaluate, bootstrap_brier_skill, print_bootstrap
from data.lineup_slots import attach_lineup_slots
from data.batting_lines import get_batting_lines
from features.run_features import (
    RunScoringModel, add_lineup_obp_behind, build_run_training_frame,
)
from features.pa_projection import (
    prepare_pa_training_frame, train_pa_model, predict_pa_distributions,
    evaluate_pa_model, print_pa_evaluation, slot_summary,
)

warnings.filterwarnings("ignore")

TRAIN_FRAC = 0.75
MODEL_C = 0.1
HRR_LINES = (0.5, 1.5, 2.5, 3.5)
# Total bases and hits. Both settle inside the plate appearance, so they
# reuse the compound engine with a different contribution distribution
# and need none of the run-attribution machinery H+R+RBI required.
TB_LINES = (0.5, 1.5, 2.5, 3.5)
# 0.5 is included deliberately even though prob_hit already answers it by
# a different route -- the two are cross-checked against each other in
# predict_slate.py, and a disagreement means one path is broken.
HIT_LINES = (0.5, 1.5, 2.5)

# --- Data source --------------------------------------------------------
# False = the original 9-slugger list in config.PLAYERS.
# True  = the stratified wide pool built by build_wide_pool.py. Run that
#         script first; this flag only chooses which cache to read.
USE_WIDE_POOL = True
POOL_SEASON, POOL_SIZE = 2025, 60

# --- Plate-appearance projection ---------------------------------------
# Requires cache/lineup_slots.csv (built by build_wide_pool.py). When
# lineup data is missing this falls back to each batter's historical PA
# histogram, which is what V5 originally used.
USE_PA_MODEL = True

# Which game-level question each per-PA target answers.
BINARY_TARGETS = {
    "is_hr": ("got_hr", "at least 1 home run"),
    "is_hit": ("got_hit", "at least 1 hit"),
    "is_walk": ("got_walk", "at least 1 walk"),
}


def load_raw_statcast() -> pd.DataFrame:
    """Either the original nine-player list or the wide stratified pool."""
    if not USE_WIDE_POOL:
        print("Loading the 9-player pool from config.PLAYERS...")
        frames = []
        for first, last in PLAYERS:
            try:
                frames.append(load_batter_statcast_cached(first, last, START_DATE, END_DATE))
            except Exception as e:
                print(f"  Skipping {first} {last}: {e}")
        return pd.concat(frames, ignore_index=True)

    from data.player_pool import get_player_pool, load_pool_statcast
    print(f"Loading the wide stratified pool ({POOL_SIZE} hitters)...")
    pool = get_player_pool(POOL_SEASON, n_players=POOL_SIZE)
    combined = load_pool_statcast(pool, START_DATE, END_DATE)
    combined.attrs["pool"] = pool
    return combined


def build_pa_dataset() -> tuple:
    combined = load_raw_statcast()

    print("Building plate-appearance table...")
    pa = build_pa_table(combined)
    print(f"  {len(pa):,} plate appearances, {pa['batter'].nunique()} batters")

    print("Estimating shrunk batter rates (windows in PA, not games)...")
    pa = add_batter_rolling_rates(pa)
    priors = pa.attrs.get("rate_priors", {})
    for target, info in priors.items():
        print(f"  {target:9s} league rate {info['prior_mean']:.4f}  "
              f"shrinkage strength k = {info['k']:.0f} PA")

    print("Estimating shrunk pitcher-allowed rates...")
    pa = add_pitcher_rolling_rates(pa)

    print("Building log5 matchup rates...")
    pa = add_matchup_rates(pa)

    pa["park_hr_factor"] = pa["home_team"].apply(get_park_factor) / 100.0

    game_totals = build_game_totals(pa)

    if USE_PA_MODEL:
        print("Attaching batting-order slots (cached; run build_wide_pool.py "
              "first if this is empty)...")
        try:
            game_totals = attach_lineup_slots(game_totals)
            matched = int(game_totals["lineup_slot"].notna().sum())
            print(f"  {matched:,} of {len(game_totals):,} player-games have a slot.")
        except Exception as e:
            print(f"  Lineup slots unavailable ({type(e).__name__}) -- "
                  f"falling back to historical PA histograms.")
            game_totals["lineup_slot"] = np.nan
            game_totals["is_starter"] = 0

    # ---- Official hits / runs / RBI -----------------------------------
    # Runs scored are not derivable from Statcast, so the target for the
    # H+R+RBI line comes from MLB's own boxscores. Without them there is no
    # honest way to score this prop at all, so the failure is loud.
    print("Attaching official boxscore lines (hits / runs / RBI)...")
    obp_series = (pa.groupby("batter")["is_hit"].mean()
                  + pa.groupby("batter")["is_walk"].mean())
    league_obp = float(pa["is_hit"].mean() + pa["is_walk"].mean())
    try:
        lines = get_batting_lines(game_totals["game_pk"].unique(), verbose=True)
        if lines.empty:
            raise ValueError("cache/batting_lines.csv is empty")
        game_totals = build_run_training_frame(game_totals, lines)
        game_totals = add_lineup_obp_behind(game_totals, obp_series, league_obp)
        coverage = float(game_totals["runs_official"].notna().mean())
        print(f"  Official lines matched for {coverage:.1%} of player-games.")
        game_totals = game_totals.dropna(subset=["hrr"])
    except Exception as e:
        raise SystemExit(
            f"Official batting lines unavailable ({type(e).__name__}: {e}).\n"
            f"The H+R+RBI prop needs RUNS SCORED, which Statcast does not\n"
            f"provide. Run:  python fetch_batting_lines.py"
        )

    return pa, game_totals, obp_series, league_obp


def build_pa_projection(game_totals: pd.DataFrame, split_date):
    """
    Train the plate-appearance model, or return None if lineup data is
    missing. Returns (model, frame_with_features, evaluation_dict).
    """
    if "lineup_slot" not in game_totals.columns or game_totals["lineup_slot"].isna().all():
        return None, None, None

    frame = prepare_pa_training_frame(game_totals)
    if len(frame) < 500:
        print(f"  Only {len(frame)} usable rows for the PA model -- skipping.")
        return None, None, None

    train = frame[frame["game_date"] <= split_date]
    test = frame[frame["game_date"] > split_date]
    if len(train) < 300 or len(test) < 100:
        return None, None, None

    model = train_pa_model(train)
    return model, frame, evaluate_pa_model(model, test)


def train_rate_model(train_pa, cols, target):
    """The per-PLATE-APPEARANCE outcome model (home run / hit / walk / K).

    Named apart from features.pa_projection.train_pa_model deliberately:
    that one predicts HOW MANY plate appearances, this one predicts WHAT
    HAPPENS in one. Two very different jobs that both wanted the same
    name.
    """
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=MODEL_C)),
    ])
    model.fit(train_pa[cols], train_pa[target])
    return model


def main():
    pa, game_totals, obp_series, league_obp = build_pa_dataset()

    feature_groups = rate_feature_cols()
    needed = sorted({c for cols in feature_groups.values() for c in cols})
    pa = pa.dropna(subset=needed).reset_index(drop=True)
    pa = pa.sort_values("game_date").reset_index(drop=True)

    split_date = pa["game_date"].quantile(TRAIN_FRAC)
    train_pa = pa[pa["game_date"] <= split_date]
    test_pa = pa[pa["game_date"] > split_date]
    test_games = game_totals[game_totals["game_date"] > split_date].copy()

    print(f"\nSplit at {split_date.date()}: "
          f"{len(train_pa):,} train PA | {len(test_pa):,} test PA "
          f"| {len(test_games):,} test games")

    # ---------- 1. Per-PA models ---------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1 -- PER-PLATE-APPEARANCE MODELS")
    print("This is where the sample size is. A modest edge here compounds")
    print("into the game-level answer; a nonexistent one cannot.")
    print("=" * 72)

    pa_models, pa_probs = {}, {}
    for target in RATE_TARGETS:
        cols = feature_groups[target]
        model = train_rate_model(train_pa, cols, target)
        probs = model.predict_proba(test_pa[cols])[:, 1]
        pa_models[target] = (model, cols)
        pa_probs[target] = probs

        result = evaluate(test_pa[target], probs, label=target)
        print(f"\n  {target:9s} n={result['n']:,}  rate {result['base_rate']:.4f}")
        print(f"    AUC {result['auc']:.4f} | Brier {result['brier']:.5f} | "
              f"Brier skill {result['brier_skill']:+.5f}")

    # ---------- 2. Compound to game level ------------------------------
    print("\n" + "=" * 72)
    print("STEP 2 -- COMPOUND TO GAME LEVEL, AND COMPARE TO DIRECT MODELLING")
    print("'compound' = per-PA rate raised over the PA count.")
    print("'direct'   = a game-level classifier on the same information.")
    print("=" * 72)

    test_pa = test_pa.copy()
    for target in RATE_TARGETS:
        test_pa[f"p_{target}"] = pa_probs[target]

    # Average the per-PA probability within each game. Each game's PAs
    # share a batter, a park and (mostly) a starting pitcher, so their
    # rates are near-identical -- the mean is an accurate summary, not a
    # crude one.
    per_game = test_pa.groupby(["batter", "game_pk"]).agg(
        **{f"p_{t}": (f"p_{t}", "mean") for t in RATE_TARGETS},
        pa_observed=("is_hr", "size"),
    ).reset_index()
    scored = test_games.merge(per_game, on=["batter", "game_pk"], how="inner")
    # row_dists is built positionally below, and Step 3 indexes into it via
    # itertuples().Index -- so the index must be a clean 0..n-1 range.
    scored = scored.reset_index(drop=True)

    # --- Plate-appearance distributions --------------------------------
    # Two sources, in order of preference:
    #   1. the PA model (lineup slot + home/away + recent history)
    #   2. the batter's own historical PA histogram
    # Whichever is used, `pa_dists_by_row` holds one distribution per
    # scored game so the compounding step is identical either way.
    hist_dists = {
        batter: empirical_pa_distribution(
            game_totals[game_totals["game_date"] <= split_date], batter
        )
        for batter in scored["batter"].unique()
    }
    pa_dists = hist_dists  # kept for the leverage diagnostic below

    pa_model, pa_frame, pa_eval = build_pa_projection(game_totals, split_date)
    row_dists = None
    if pa_model is not None:
        print("\n" + "=" * 72)
        print("PLATE-APPEARANCE PROJECTION")
        print("The V5 diagnostic said exposure was worth ~12x every skill")
        print("feature combined. This is the attempt to capture the")
        print("knowable part of it: lineup slot, home/away, recent usage.")
        print("=" * 72)
        print_pa_evaluation(pa_eval)
        print("\n  Average PA by lineup slot (the effect being exploited):")
        print(slot_summary(pa_frame).to_string(index=False,
                                               float_format=lambda v: f"{v:.3f}"))

        slot_lookup = pa_frame.set_index(["batter", "game_pk"])
        joinable = scored.set_index(["batter", "game_pk"]).index
        available = slot_lookup.index.intersection(joinable)
        if len(available) > 0.5 * len(scored):
            aligned = slot_lookup.loc[available]
            projected = dict(zip(available, predict_pa_distributions(pa_model, aligned)))
            row_dists = [
                projected.get((b, g), hist_dists[b])
                for b, g in zip(scored["batter"], scored["game_pk"])
            ]
            covered = sum(1 for b, g in zip(scored["batter"], scored["game_pk"])
                          if (b, g) in projected)
            print(f"\n  Using the PA model for {covered:,} of {len(scored):,} "
                  f"scored games; the rest fall back to history.")
        else:
            print("\n  Too few scored games have a lineup slot -- using "
                  "historical PA histograms throughout.")

    if row_dists is None:
        row_dists = [hist_dists[b] for b in scored["batter"]]

    summary = []
    for target, (game_col, description) in BINARY_TARGETS.items():
        compounded = np.array([
            marginalise_over_pa(dist,
                                lambda n, p=prob: float(prob_at_least_one(p, n)))
            for dist, prob in zip(row_dists, scored[f"p_{target}"])
        ])
        compound_result = evaluate(scored[game_col], compounded,
                                   label=f"{description} (compound)")

        # Direct comparison: same features, averaged per game, trained
        # straight on the game-level binary outcome.
        cols = feature_groups[target]
        train_game_x = train_pa.groupby(["batter", "game_pk"])[cols].mean().reset_index()
        train_game_y = game_totals[game_totals["game_date"] <= split_date]
        train_game = train_game_x.merge(
            train_game_y[["batter", "game_pk", game_col]], on=["batter", "game_pk"]
        )
        test_game_x = test_pa.groupby(["batter", "game_pk"])[cols].mean().reset_index()
        test_game = test_game_x.merge(
            scored[["batter", "game_pk", game_col]], on=["batter", "game_pk"]
        )

        direct = Pipeline([("scale", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=3000, C=MODEL_C))])
        direct.fit(train_game[cols], train_game[game_col])
        direct_probs = direct.predict_proba(test_game[cols])[:, 1]
        direct_result = evaluate(test_game[game_col], direct_probs,
                                 label=f"{description} (direct)")

        print(f"\n  --- {description} ---")
        print(f"    {'':10s} {'AUC':>8} {'Brier':>10} {'BrierSkill':>12}")
        print(f"    {'compound':10s} {compound_result['auc']:>8.4f} "
              f"{compound_result['brier']:>10.5f} {compound_result['brier_skill']:>+12.5f}")
        print(f"    {'direct':10s} {direct_result['auc']:>8.4f} "
              f"{direct_result['brier']:>10.5f} {direct_result['brier_skill']:>+12.5f}")
        gain = compound_result["brier_skill"] - direct_result["brier_skill"]
        print(f"    compounding is worth {gain:+.5f} Brier skill here")

        summary.append({"prop": description, "method": "compound",
                        **{k: compound_result[k] for k in ("auc", "brier_skill")}})
        summary.append({"prop": description, "method": "direct",
                        **{k: direct_result[k] for k in ("auc", "brier_skill")}})

    # ---------- 3. The counting prop -----------------------------------
    print("\n" + "=" * 72)
    print("STEP 3 -- HITS + RUNS + RBI OVER/UNDER")
    print("This is the prop a yes/no classifier structurally cannot answer.")
    print("A count distribution answers every line at once.")
    print("=" * 72)

    train_pa_full = pa[pa["game_date"] <= split_date]
    # The run-scoring model is fitted on TRAIN ONLY. It reads the outcome
    # it is trying to predict, so fitting it on everything would quietly
    # hand the test set its own answer.
    train_games = game_totals[game_totals["game_date"] <= split_date]
    run_model = RunScoringModel.fit(train_games, obp_series, league_obp)

    base_dist = per_pa_contribution_distribution(
        train_pa_full, "hrr_certain", score_prob=run_model.league_prob)

    # Games are MORE variable than independent plate appearances imply: a
    # dominant pitcher suppresses every trip, a bad one gets hit by
    # everyone. Ignoring that leaves too little mass on zero, which shows
    # up as over-predicting the low lines. Fitted on TRAIN only, against
    # the OFFICIAL game total.
    dispersion_sd = fit_game_dispersion(train_games, base_dist)
    observed_ratio = train_games["hrr"].var() / train_games["hrr"].mean()
    print(f"\n  Game-level dispersion: observed variance/mean = "
          f"{observed_ratio:.3f} (1.0 would mean independent plate")
    print(f"  appearances). Fitted multiplier s.d. = {dispersion_sd:.3f}"
          f"{' -- no excess, using plain convolution.' if dispersion_sd <= 0 else ''}")
    print("\n  Population per-PA contribution (hits + runs + RBI):")
    for value, prob in enumerate(base_dist):
        if prob > 0.001:
            print(f"    {value} -> {prob:.4f}")

    # Per-PA expected contribution for each batter: rebuild from the
    # component probabilities the models already produced. RBI is not
    # modelled directly (it depends on teammates being on base, which V2
    # established is outside a batter's control) so its per-PA rate is
    # taken from the batter's own shrunk history.
    rbi_rate = train_pa_full.groupby("batter")["rbi"].mean()
    league_rbi = float(train_pa_full["rbi"].mean())
    scored["p_rbi"] = scored["batter"].map(rbi_rate).fillna(league_rbi)

    # RUNS. Routed through reaching base rather than estimated directly, so
    # it responds to the matchup: the same hitter against a pitcher who
    # suppresses his on-base rate scores less, automatically. A per-batter
    # runs-per-PA constant would not move at all.
    #
    # Note what is NOT in this sum: walks. The prop is hits + runs + RBI. A
    # walk still matters, but only through p_reach below -- it puts the
    # batter in position to score, and that is the whole of its
    # contribution.
    scored["p_run"] = run_model.runs_per_pa(
        scored,
        p_hr=scored["p_is_hr"].to_numpy(),
        p_reach=(scored["p_is_hit"] + scored["p_is_walk"]).to_numpy(),
    )
    scored["expected_hrr_per_pa"] = (
        scored["p_is_hit"] + scored["p_run"] + scored["p_rbi"]
    )
    print(f"\n  Expected per plate appearance, averaged over test games:")
    print(f"    hits {scored['p_is_hit'].mean():.4f} + runs "
          f"{scored['p_run'].mean():.4f} + RBI {scored['p_rbi'].mean():.4f} "
          f"= {scored['expected_hrr_per_pa'].mean():.4f}")
    print(f"    actual H+R+RBI per PA on those games: "
          f"{scored['hrr'].sum() / scored['pa'].sum():.4f}")

    # Which per-PA shape method fits this data -- exponential tilting or
    # frequency scaling? Both are zero-parameter structural choices, so
    # selecting on train is safe (unlike calibrator selection, which needs
    # a held-out half). Measured, not argued.
    # Each batter's observed H+R+RBI per plate appearance. The certain part
    # is read off the PA table; the run is added as league scoring rate
    # times his non-home-run times on base, matching how base_dist folds it
    # in. Using hrr_certain alone here would understate every hitter by
    # about 0.09 per plate appearance and make the shape selector compare
    # two methods on the wrong target.
    certain = train_pa_full.groupby("batter")["hrr_certain"].mean()
    on_base = train_pa_full.groupby("batter")["reached_nonhr"].mean()
    train_batter_rate = certain + run_model.league_prob * on_base
    league_rate = float(train_pa_full["hrr_certain"].mean()
                        + run_model.league_prob
                        * train_pa_full["reached_nonhr"].mean())
    train_means = train_games["batter"].map(train_batter_rate).fillna(
        league_rate).to_numpy()
    shape_method = select_tilt_method(
        train_games, base_dist, train_means, dispersion_sd, HRR_LINES
    )
    print(f"  Using '{shape_method}' scaling.")

    line_results = []
    for line in HRR_LINES:
        predictions = []
        for row in scored.itertuples():
            tilted = scale_contribution_distribution(
                base_dist, float(row.expected_hrr_per_pa), method=shape_method
            )
            predictions.append(marginalise_over_pa(
                row_dists[row.Index],
                lambda n, d=tilted: prob_over_line(
                    compound_count_distribution_overdispersed(
                        d, int(n), dispersion_sd, method=shape_method), line
                ),
            ))
        predictions = np.array(predictions)
        actual = (scored["hrr"] > line).astype(int)
        result = evaluate(actual, predictions, label=f"H+R+RBI over {line}")
        line_results.append({"line": line, **{k: result[k] for k in
                             ("n", "base_rate", "auc", "brier", "brier_skill", "mean_pred")}})
        print(f"\n  Over {line}: actual {result['base_rate']:.4f}  "
              f"predicted {result['mean_pred']:.4f}")
        print(f"    AUC {result['auc']:.4f} | Brier skill {result['brier_skill']:+.5f}")
        if line == 1.5:
            print_bootstrap(bootstrap_brier_skill(actual, predictions))

    # ---------- 4. Where the remaining leverage actually is ------------
    print("\n" + "=" * 72)
    print("STEP 4 -- WHERE THE REMAINING LEVERAGE IS")
    print("Every model here conditions on how many plate appearances a")
    print("hitter gets, and none of them predicts it. This measures what")
    print("that omission costs.")
    print("=" * 72)

    for target, (game_col, description) in BINARY_TARGETS.items():
        diag = pa_leverage_diagnostic(
            scored, pa_dists, f"p_{target}", game_col, evaluate
        )
        print(f"\n  {description}")
        for key, label in [("historical", "historical PA distribution"),
                           ("mean_pa", "mean PA point estimate"),
                           ("actual_pa", "ACTUAL PA (upper bound only)")]:
            if key in diag:
                print(f"    {label:28s} AUC {diag[key]['auc']:.4f}  "
                      f"skill {diag[key]['brier_skill']:+.5f}")
        if "value_of_knowing_pa" in diag:
            print(f"    -> knowing PA exactly is worth "
                  f"{diag['value_of_knowing_pa']:+.5f} Brier skill")

    train_games = game_totals[game_totals["game_date"] <= split_date]
    batter_mean_pa = train_games.groupby("batter")["pa"].mean()
    residual = scored["pa"] - scored["batter"].map(batter_mean_pa)
    explained = 1.0 - residual.var() / scored["pa"].var()
    print(f"\n  How predictable is PA itself?")
    print(f"    Pool s.d. of PA per game            : {scored['pa'].std():.3f}")
    print(f"    s.d. after removing batter's own mean: {residual.std():.3f}")
    print(f"    Batter identity explains {explained:.1%} of PA variance.")
    print("    The rest is lineup slot (not in Statcast), game length,")
    print("    blowouts and pinch-hitting. Slot and home/away ARE knowable")
    print("    before first pitch from MLB's lineup feed; game length is not.")

    # ---------- 5. Verdict ---------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(pd.DataFrame(line_results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nRead the Brier skill column. Positive means the number beats")
    print("quoting the base rate; the AUC column only says whether the")
    print("ranking is right, which is a weaker claim.")


if __name__ == "__main__":
    main()
