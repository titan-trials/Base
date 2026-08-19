"""
Evaluation and calibration harness for the V4 home-run probability model.

THE REFRAME THIS FILE ENCODES
-----------------------------
V1-V3 asked "can the model correctly call whether this guy goes deep
tonight," and the honest answer was no -- the best AUC the project ever
reached on HR was 0.565. That ceiling is not a bug to engineer around. A
single-game home run is a low-base-rate event whose outcome depends on
pitch location, timing to the millisecond, and where the ball happens to
be caught: an error term far larger than anything the observable features
explain.

So V4 asks a different, answerable question:

    Not "will he homer" (a classification problem with a hard ceiling)
    but "what is the probability that he homers, and is that number
    trustworthy at face value" (a calibration problem).

Those are genuinely different targets, and a model can be excellent at the
second while being mediocre at the first. "23.6%" is a useful, honest,
actionable output. "YES, 60% confident" was never going to be.

WHAT GETS MEASURED, AND WHY
---------------------------
AUC             Ranking only: given one HR game and one non-HR game, how
                often is the HR game ranked higher. Says NOTHING about
                whether the printed percentage is right. Kept because it's
                the project's existing yardstick and comparable to prior
                versions.

Brier score     Mean squared error of the probability itself. Lower is
                better. This is the metric that actually matches the new
                goal -- it punishes a confident wrong number and rewards
                an honest uncertain one.

Brier skill     Brier score compared against the dumbest honest baseline:
                always predicting the overall base rate. This is the
                number that matters most. Positive = the features add real
                information beyond "sluggers homer about X% of the time."
                Zero or negative = the model is elaborate decoration on
                the base rate, no matter how good the AUC looks.

Log loss        Similar to Brier but punishes confident errors far more
                harshly. Reported as a cross-check.

Reliability     The bucket-by-bucket table from model/calibration.py: when
                the model said 25%, did it happen ~25% of the time.

CALIBRATION METHOD
------------------
The data is split THREE ways, always chronologically:

    train (60%)      fit the logistic regression
    calibrate (15%)  fit the raw-prob -> honest-prob mapping
    test (25%)       never touched until the final report

Fitting the calibrator on training predictions would be circular (the
model is overconfident on data it has already seen); fitting it on test
would be leakage. Hence the dedicated middle slice.

WHICH CALIBRATOR -- AND WHY THIS IS NOT A FREE CHOICE
-----------------------------------------------------
The first real run of this file defaulted to isotonic regression whenever
the calibration slice cleared 400 rows. That was WRONG, and wrong in a way
that inverted the project's headline conclusion.

Isotonic is a free-form step function. On a 908-row calibration slice with
a ~22% base rate, it finds regions where every calibration game happened
to be a home run and maps them to exactly 1.000. On the real data it
emitted p = 1.000 for 16 test games (of which 4 were actually home runs)
and p = 0.000 for 10 more. Measured cost: Brier skill went from -0.003
(raw, i.e. roughly break-even against the base rate) to -0.045. The
"nothing beats the base rate" verdict from that run was substantially an
artifact of the calibrator, not a finding about baseball.

The fix is not to hard-code a different calibrator -- it's to stop
guessing. `fit_calibrated` now splits the calibration slice in half,
fits each candidate (none / Platt / isotonic) on the first half, scores
them on the second, and keeps the winner. A calibrator that only helps on
data it has already seen never gets selected. In this regime the honest
answer is usually "none" or "Platt": when a model's raw output already
sits close to the base rate, there is very little miscalibration to fix
and a flexible calibrator can only add variance.

General lesson worth keeping: the more flexible the correction, the more
data it needs before it is a correction rather than a memorisation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MIN_ISOTONIC_SAMPLES = 400

# Regularisation strengths tried when `C=None` is passed. On this problem
# the signal is weak enough that the default C=1.0 overfits noticeably:
# the full 28-feature model gained +0.007 Brier skill purely by moving
# from C=1.0 to C=0.003. Chosen on the calibration slice, never on test.
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def chronological_split(df: pd.DataFrame, train_frac=0.60, calib_frac=0.15,
                        date_col: str = "game_date"):
    """
    Three-way split by time, never shuffled. Baseball has real seasonal and
    career drift (a hitter ages, a park is renovated, the ball itself
    changes), so a shuffled split lets the model peek at the future and
    inflates every metric.
    """
    ordered = df.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:calib_end].copy(),
        ordered.iloc[calib_end:].copy(),
    )


def build_model(C: float = 0.03) -> Pipeline:
    """
    Standardised logistic regression.

    Scaling matters here in a way it didn't in earlier versions: the V4
    feature set mixes columns on wildly different scales (temp_f ~ 85,
    barrel_rate ~ 0.09, venue_edge ~ 0.01). Unscaled, L2 regularisation
    would penalise the small-scale columns far harder than the large-scale
    ones purely because of their units, which has nothing to do with how
    informative they are.

    Default C is 0.03 rather than sklearn's 1.0. With a signal this weak,
    weak regularisation lets the model fit noise in the training window
    that doesn't survive into the test window -- measured on real data,
    C=1.0 cost about 0.007 Brier skill against C=0.003 on the full
    feature set. Pass C=None to select it on the calibration slice.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=C)),
    ])


def _make_calibrator(kind: str, raw, y):
    if kind == "none":
        return None
    if kind == "isotonic":
        if len(raw) < MIN_ISOTONIC_SAMPLES or len(np.unique(y)) < 2:
            return None
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(np.asarray(raw).reshape(-1, 1), y)
    return platt


def _apply_calibrator(calibrator, raw) -> np.ndarray:
    raw = np.asarray(raw)
    if calibrator is None:
        return raw
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(raw), 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def select_calibrator(raw, y, verbose: bool = False):
    """
    Choose between no calibration, Platt, and isotonic by HOLDING OUT half
    the calibration slice.

    Each candidate is fit on the first half and scored on the second. This
    is the whole point: isotonic always looks best on the data it was fit
    to, because it can bend arbitrarily to match it. Scoring on unseen
    rows is what exposes that as memorisation rather than correction.

    Selection uses the ONE-STANDARD-ERROR RULE rather than plain
    argmin. Half a calibration slice is a few hundred rows, so the
    held-out Brier scores carry real noise -- picking the raw winner means
    a flexible calibrator gets selected any time it gets lucky by a
    thousandth. Instead, the SIMPLEST candidate whose score is within one
    standard error of the best is chosen, where simplicity is ordered
    none < Platt < isotonic.

    This matters concretely: without it, isotonic won several rungs of the
    real-data ablation by margins far smaller than their own error bars,
    and cost up to 0.05 Brier skill on the test set each time. Complexity
    has to clear a bar, not just win a coin flip.

    Returns (kind, brier_on_held_out_half).
    """
    raw = np.asarray(raw)
    y = np.asarray(y)
    if len(raw) < 60:
        return "none", float("nan")

    mid = len(raw) // 2
    raw_a, raw_b = raw[:mid], raw[mid:]
    y_a, y_b = y[:mid], y[mid:]
    if len(np.unique(y_a)) < 2 or len(np.unique(y_b)) < 2:
        return "none", float("nan")

    # Per-sample squared errors, so candidates can be compared as a PAIRED
    # difference -- the games are the same, so pairing removes the shared
    # game-to-game variance and gives a far tighter error estimate.
    errors = {}
    for kind in ("none", "platt", "isotonic"):
        try:
            cal = _make_calibrator(kind, raw_a, y_a)
        except ValueError:
            continue
        if kind != "none" and cal is None:
            continue  # isotonic refused (slice too small) -- not a candidate
        errors[kind] = (y_b - _apply_calibrator(cal, raw_b)) ** 2

    scores = {k: float(v.mean()) for k, v in errors.items()}
    best = min(scores, key=scores.get)

    if verbose:
        detail = "  ".join(f"{k}={v:.5f}" for k, v in scores.items())
        print(f"    calibrator selection (held-out half of calib slice): {detail}")

    for kind in ("none", "platt", "isotonic"):
        if kind not in errors:
            continue
        if kind == best:
            return kind, scores[kind]
        diff = errors[kind] - errors[best]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        if scores[kind] <= scores[best] + se:
            if verbose:
                print(f"    -> '{kind}' is within 1 SE ({se:.5f}) of '{best}'; "
                      f"taking the simpler one.")
            return kind, scores[kind]

    return best, scores[best]


def select_C(train_df, calib_df, feature_cols, target_col="hit_hr",
             grid=C_GRID, verbose: bool = False) -> float:
    """Pick regularisation strength by Brier score on the calibration
    slice. Never uses test data."""
    best_C, best_score = grid[0], float("inf")
    for C in grid:
        model = build_model(C=C).fit(train_df[feature_cols], train_df[target_col])
        score = brier_score_loss(
            calib_df[target_col], model.predict_proba(calib_df[feature_cols])[:, 1]
        )
        if verbose:
            print(f"    C={C:<7} calib Brier {score:.5f}")
        if score < best_score:
            best_C, best_score = C, score
    return best_C


def fit_calibrated(train_df, calib_df, feature_cols, target_col="hit_hr",
                   C: float = 0.03, verbose: bool = False):
    """
    Fit the model on `train_df`, then choose and fit a calibrator using
    only `calib_df`.

    C=None selects the regularisation strength on the calibration slice
    too. Returns (model, calibrator, method_name); `calibrator` is None
    when "no calibration" won, which on weak-signal data is common and
    correct rather than a failure.
    """
    if C is None:
        C = select_C(train_df, calib_df, feature_cols, target_col, verbose=verbose)
        if verbose:
            print(f"    selected C = {C}")

    model = build_model(C=C).fit(train_df[feature_cols], train_df[target_col])

    if len(calib_df) == 0:
        return model, None, "none (no calibration slice)"

    raw = model.predict_proba(calib_df[feature_cols])[:, 1]
    y_calib = calib_df[target_col].to_numpy()

    kind, _ = select_calibrator(raw, y_calib, verbose=verbose)
    if kind == "none":
        return model, None, f"none (C={C}; calibration did not help on held-out calib data)"

    calibrator = _make_calibrator(kind, raw, y_calib)
    return model, calibrator, f"{kind} (C={C})"


def predict_calibrated(model, calibrator, X) -> np.ndarray:
    return _apply_calibrator(calibrator, model.predict_proba(X)[:, 1])


def evaluate(y_true, probs, label: str = "") -> dict:
    """
    Every metric in one dict. `brier_skill` is the headline: it compares
    the model against always predicting the observed base rate, which is
    the honest null hypothesis for a probability model.
    """
    y_true = np.asarray(y_true)
    probs = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)

    base_rate = float(y_true.mean())
    base_probs = np.full_like(probs, base_rate)

    brier = brier_score_loss(y_true, probs)
    brier_base = brier_score_loss(y_true, base_probs)

    return {
        "label": label,
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier),
        "brier_base": float(brier_base),
        "brier_skill": float(1.0 - brier / brier_base) if brier_base > 0 else float("nan"),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "log_loss_base": float(log_loss(y_true, base_probs, labels=[0, 1])),
        "mean_pred": float(probs.mean()),
    }


def print_evaluation(result: dict):
    print(f"\n--- {result['label']} ---")
    print(f"  Test games            : {result['n']}")
    print(f"  Actual HR rate        : {result['base_rate']:.3f}")
    print(f"  Mean predicted        : {result['mean_pred']:.3f}  "
          f"(should sit close to the actual rate)")
    print(f"  AUC                   : {result['auc']:.4f}   (ranking ability only)")
    print(f"  Brier score           : {result['brier']:.5f}   (lower is better)")
    print(f"  Brier, base-rate-only : {result['brier_base']:.5f}")
    print(f"  BRIER SKILL SCORE     : {result['brier_skill']:+.4f}   "
          f"<- the number that matters")
    print(f"  Log loss              : {result['log_loss']:.5f}  "
          f"(base rate only: {result['log_loss_base']:.5f})")


def reliability_by_quantile(y_true, probs, n_bins: int = 8) -> pd.DataFrame:
    """
    Reliability table using EQUAL-COUNT buckets instead of equal-width ones.

    model/calibration.py's version cuts the 0-1 range into equal-width
    slices, which is the right choice for raw model output. It goes wrong
    on ISOTONIC output specifically: isotonic is a step function, so its
    predictions pile up on a handful of discrete values and most
    equal-width bins end up holding one or two rows -- rows whose "actual
    rate" is then 0.000 or 1.000 and means nothing.

    Equal-count buckets put a comparable number of games in each row, so
    every observed rate is estimated from a usable sample. Both tables are
    worth printing; they answer the same question with different failure
    modes.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "prob": np.asarray(probs)})
    df["bucket"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop")

    table = df.groupby("bucket", observed=True).agg(
        mean_predicted=("prob", "mean"),
        actual_rate=("y_true", "mean"),
        count=("y_true", "size"),
    ).reset_index()
    table["gap"] = table["actual_rate"] - table["mean_predicted"]
    return table


def print_reliability_by_quantile(y_true, probs, n_bins: int = 8, label: str = "Model"):
    table = reliability_by_quantile(y_true, probs, n_bins=n_bins)
    print(f"\n=== Reliability (equal-count buckets): {label} ===")
    print(f"{'Predicted Range':<26} {'Mean Pred':>10} {'Actual':>9} {'Count':>7} {'Gap':>8}")
    for _, row in table.iterrows():
        print(f"{str(row['bucket']):<26} {row['mean_predicted']:>10.3f} "
              f"{row['actual_rate']:>9.3f} {row['count']:>7.0f} {row['gap']:>+8.3f}")
    worst = table["gap"].abs().max()
    print(f"\nLargest gap in any bucket: {worst:.3f}. Under ~0.05 across every")
    print("bucket means the printed percentage can be read at face value.")


def compare_results(results: list) -> pd.DataFrame:
    """Side-by-side ablation table, ordered as run."""
    table = pd.DataFrame(results)[
        ["label", "n", "auc", "brier", "brier_skill", "log_loss"]
    ]
    return table


def univariate_ranking(df, feature_cols, target_col="hit_hr") -> pd.DataFrame:
    """
    Each feature's AUC on its own, ranked by distance from 0.50.

    Worth printing separately from the model coefficients, because the two
    answer different questions and can disagree sharply. A coefficient
    says "what does this column add GIVEN everything else in the model" --
    so a genuinely informative feature can show a coefficient near zero
    purely because a collinear neighbour already carries the same
    information. Univariate AUC says "does this column know anything at
    all," independent of the company it keeps.

    On the real data this distinction was the whole story for contact
    quality: career_barrel_rate is the single most informative column in
    the entire feature set on its own, yet the contact-quality GROUP made
    the model worse, because barrel rate and career HR rate are two
    measurements of the same underlying power.
    """
    rows = []
    for col in feature_cols:
        series = df[col]
        try:
            auc = roc_auc_score(df[target_col], series)
        except ValueError:
            auc = float("nan")
        rows.append({"feature": col, "auc": auc, "distance_from_coinflip": abs(auc - 0.5)})
    return pd.DataFrame(rows).sort_values(
        "distance_from_coinflip", ascending=False
    ).reset_index(drop=True)


def print_univariate_ranking(table: pd.DataFrame, top: int = 12):
    print("\n--- Each feature on its own (univariate AUC) ---")
    print("Answers 'does this column know anything at all', separately from")
    print("whether it ADDS anything once the others are present.")
    print(table.head(top).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  ... least informative:")
    print(table.tail(4).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def bootstrap_brier_skill(y_true, probs, n_boot: int = 4000,
                          random_state: int = 0) -> dict:
    """
    Confidence interval on Brier skill, by resampling test games.

    A single point estimate of +0.006 is not interpretable on its own --
    it could be a real edge or a lucky test window. Resampling the test
    set with replacement gives the sampling distribution of that number,
    which is the only way to answer "is this bigger than zero."

    Note the CI is about SAMPLING noise within one test window. It does
    not capture the risk that the whole window was unusual; that's what
    rolling_origin_evaluation is for. Both are needed -- a result should
    survive both before it's believed.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    rng = np.random.default_rng(random_state)

    skills = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_sample, p_sample = y_true[idx], probs[idx]
        rate = y_sample.mean()
        if rate in (0.0, 1.0):
            continue
        brier = brier_score_loss(y_sample, p_sample)
        brier_base = brier_score_loss(y_sample, np.full_like(p_sample, rate))
        skills.append(1.0 - brier / brier_base)

    skills = np.array(skills)
    return {
        "point": float(1.0 - brier_score_loss(y_true, probs)
                       / brier_score_loss(y_true, np.full_like(probs, y_true.mean()))),
        "ci_low": float(np.percentile(skills, 2.5)),
        "ci_high": float(np.percentile(skills, 97.5)),
        "p_above_zero": float((skills > 0).mean()),
        "n_boot": int(len(skills)),
    }


def print_bootstrap(result: dict):
    print(f"\n--- Bootstrap CI on Brier skill ({result['n_boot']} resamples) ---")
    print(f"  Point estimate : {result['point']:+.5f}")
    print(f"  95% CI         : [{result['ci_low']:+.5f}, {result['ci_high']:+.5f}]")
    print(f"  P(skill > 0)   : {result['p_above_zero']:.3f}")
    if result["ci_low"] > 0:
        print("  The interval excludes zero: the model beats the base rate.")
    elif result["p_above_zero"] > 0.85:
        print("  The interval straddles zero but leans positive. Not conclusive")
        print("  from one window -- check the rolling-origin table below, which")
        print("  is the stronger evidence when each window is this noisy.")
    else:
        print("  Indistinguishable from just predicting the base rate.")


def rolling_origin_evaluation(df, feature_cols, target_col="hit_hr",
                              fractions=(0.55, 0.65, 0.75, 0.85),
                              calib_width: float = 0.10, C: float = 0.03,
                              date_col: str = "game_date") -> pd.DataFrame:
    """
    Refit and re-evaluate at several different train/test cut points.

    One chronological split gives one number, and one number on a weak
    signal is nearly uninformative -- the test window might simply have
    been kind. Walking the cut point forward asks whether the result is a
    property of the model or a property of that particular window. Four
    windows landing in the same narrow band is far more persuasive than
    one window with a tight-looking CI.
    """
    ordered = df.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)

    rows = []
    for frac in fractions:
        train_end = int(n * frac)
        calib_end = int(n * (frac + calib_width))
        train_df = ordered.iloc[:train_end]
        calib_df = ordered.iloc[train_end:calib_end]
        test_df = ordered.iloc[calib_end:]
        if len(test_df) < 200 or len(calib_df) < 60:
            continue

        model, calibrator, _ = fit_calibrated(train_df, calib_df, feature_cols,
                                              target_col, C=C)
        probs = predict_calibrated(model, calibrator, test_df[feature_cols])
        result = evaluate(test_df[target_col], probs)
        rows.append({
            "train_through": train_df[date_col].max().date(),
            "test_n": len(test_df),
            "test_hr_rate": result["base_rate"],
            "auc": result["auc"],
            "brier_skill": result["brier_skill"],
        })

    return pd.DataFrame(rows)


def print_rolling_origin(table: pd.DataFrame):
    print("\n--- Rolling-origin check (same model, different cut points) ---")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(table) >= 2:
        spread = table["brier_skill"].max() - table["brier_skill"].min()
        all_positive = (table["brier_skill"] > 0).all()
        print(f"\n  Spread across windows: {spread:.5f}")
        if all_positive:
            print("  Every window is positive. That consistency is the real")
            print("  evidence -- much stronger than any single window's CI.")
        else:
            print("  Windows disagree in sign. The effect is not stable; treat")
            print("  any single positive window as luck until it repeats.")


def permutation_test(train_df, calib_df, test_df, feature_cols, suspect_cols,
                     target_col="hit_hr", n_permutations: int = 200,
                     random_state: int = 0) -> dict:
    """
    Is a feature group carrying real signal, or did it just get lucky?

    Procedure: refit the whole pipeline `n_permutations` times, each time
    with the SUSPECT columns randomly shuffled (destroying their link to
    the outcome while preserving their distribution and their correlation
    with each other). That builds an empirical null distribution of Brier
    skill. The real model's Brier skill is then read off as a percentile
    against it.

    Interpretation:
      p < 0.05   the group is doing something a shuffled version of itself
                 would not do -- treat as real (subject to the usual
                 caveat about testing many things).
      p > 0.20   indistinguishable from noise. Drop the group. This is the
                 expected outcome for the day-of-week "5 HR on Sundays"
                 style splits, and the point of running it is to be able
                 to say that with a number instead of an opinion.
    """
    rng = np.random.default_rng(random_state)

    model, calibrator, _ = fit_calibrated(train_df, calib_df, feature_cols, target_col, C=0.03)
    real_probs = predict_calibrated(model, calibrator, test_df[feature_cols])
    real_skill = evaluate(test_df[target_col], real_probs)["brier_skill"]

    null_skills = []
    for _ in range(n_permutations):
        shuffled_train = train_df.copy()
        shuffled_calib = calib_df.copy()
        shuffled_test = test_df.copy()
        for frame in (shuffled_train, shuffled_calib, shuffled_test):
            if len(frame) == 0:
                continue
            order = rng.permutation(len(frame))
            frame[suspect_cols] = frame[suspect_cols].to_numpy()[order]

        try:
            m, c, _ = fit_calibrated(shuffled_train, shuffled_calib, feature_cols,
                                     target_col, C=0.03)
            p = predict_calibrated(m, c, shuffled_test[feature_cols])
            null_skills.append(evaluate(shuffled_test[target_col], p)["brier_skill"])
        except ValueError:
            continue

    null_skills = np.array(null_skills)
    # +1 in numerator and denominator: the standard finite-sample
    # correction, so a p-value of exactly 0 is never reported from a
    # finite number of permutations.
    p_value = (np.sum(null_skills >= real_skill) + 1) / (len(null_skills) + 1)

    return {
        "real_brier_skill": float(real_skill),
        "null_mean": float(null_skills.mean()) if len(null_skills) else float("nan"),
        "null_std": float(null_skills.std()) if len(null_skills) else float("nan"),
        "p_value": float(p_value),
        "n_permutations": int(len(null_skills)),
    }


def print_permutation_result(name: str, result: dict):
    print(f"\n--- Permutation test: {name} ---")
    print(f"  Real Brier skill      : {result['real_brier_skill']:+.4f}")
    print(f"  Shuffled null (mean)  : {result['null_mean']:+.4f} "
          f"+/- {result['null_std']:.4f}  over {result['n_permutations']} shuffles")
    print(f"  p-value               : {result['p_value']:.3f}")
    if result["p_value"] < 0.05:
        print("  VERDICT: real signal -- shuffling these columns makes the model")
        print("           measurably worse. Keep them.")
    elif result["p_value"] < 0.20:
        print("  VERDICT: ambiguous. Suggestive, not established. Keep only if")
        print("           there's a physical reason to expect the effect.")
    else:
        print("  VERDICT: indistinguishable from noise. A randomly shuffled")
        print("           version of these columns does just as well. Drop them.")


def coefficient_report(model: Pipeline, feature_cols: list) -> pd.DataFrame:
    """
    Standardised coefficients, sorted by magnitude. Because the pipeline
    scales inputs first, these ARE comparable to each other: each is the
    log-odds change per one standard deviation of that feature, so the
    ordering answers "which factor is actually moving the number."
    """
    clf = model.named_steps["clf"]
    table = pd.DataFrame({
        "feature": feature_cols,
        "coef": clf.coef_[0],
    })
    table["odds_ratio_per_sd"] = np.exp(table["coef"])
    table["abs_coef"] = table["coef"].abs()
    return table.sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True)
