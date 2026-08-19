# Baseball Predictor — Project Context
---

## What This Is
A free, pybaseball-based sports prediction project with two branches:
1. **Player props** — will a player hit a HR / walk / drive in runs in a given game
2. **Team win probability** — will a team win a given game

Built as a learning-focused sister project to Quantara, same phased-growth
philosophy: start narrow (one player, one stat), prove or disprove the
pipeline, then scale out. Unlike Quantara, several branches here ended in
honest negative results rather than working models — that's treated as a
real, useful outcome of the project, not a failure to fix.

---

## V9 (Aug 18, 2026) — two negative results, recorded so they aren't redone

Both experiments lived in standalone files that import the pipeline and
change nothing in it, so neither could affect the live model.

### Extra hitting stats do NOT help (`feature_lab.py`)
Tested xwOBA, xSLG, barrel rate, hard-hit rate, chase rate, in-zone
contact and BB/K as additional per-PA features. All available free from
the Statcast cache; all built as shrunk rolling averages shifted by one PA
so nothing sees its own outcome.

**0 of 28 (feature, target) pairs beat the current model.**

The reason is structural: every candidate is an INDIRECT measure of
something the rate histories already measure DIRECTLY. A hitter's 600-PA
home run rate is a noisy but direct read on his power; barrel rate is an
indirect read on the same thing.

Caveat that produced a follow-up: the test pooled all plate appearances,
most of which come from established hitters where rate history is already
excellent. Expected stats are supposed to win where history is THIN, so
`feature_lab.py` was rebuilt to stratify by how much history the batter
had banked, with the thin bucket (20-200 PA) as the pre-specified primary
test.

**The stratified re-test also came back negative. 0 of 28 survived
Benjamini-Hochberg, and 0 of 28 were underpowered** -- every comparison
could have detected a 0.0025 skill gain and none did. That is the useful
version of a null: the question is answered, not merely unresolved.

Three details worth keeping so this is not mis-remembered as "definitively
nothing":

- **`is_hr` x `zcontact` missed by a factor of 1.7**, not by miles:
  p = 0.0030 against a BH threshold of 0.0018 (= 0.05 x 1/28). As a single
  pre-specified hypothesis it would clear comfortably. Against it:
  `zcontact` also hit p = 0.005 with the OPPOSITE sign on walks, which
  looks more like noise landing twice than a mechanism.

- **The contact-quality family all leaned the predicted way on home runs**
  -- xslg +0.00075, barrel +0.00097, hardhit +0.00063, mean +0.00078,
  which is +25% of the 0.00318 baseline. Individually insignificant, and
  they are three views of one signal rather than three confirmations, so
  this is a hint and not a finding.

- **The xwOBA result specifically should NOT be trusted.** Its MDE came
  back at 0.00009, ten times tighter than xslg's 0.00105 despite both
  being continuous variables -- meaning the feature barely moved the
  predictions. The cause is a methodology error: `estimate_prior_strength`
  is a beta-binomial method-of-moments estimator built for 0/1 rates, and
  xwOBA is continuous on roughly [0, 2]. Applying it there is not valid,
  and if it hit the 5000 clip the feature was flattened before the test
  ran. So "xwOBA does not help" is unestablished; "this shrunk version of
  xwOBA did not move the model" is what was measured.

The decision stands regardless: `barrel` and `xslg` had proper spread,
genuinely moved the model, and still did not clear the bar. Not adopted.

### Logistic regression is the right model (`model_lab.py`)
Seven models per target. Nothing beat it; boosting was materially worse.

| model | is_hr gain | verdict |
|---|---|---|
| lr_C0.03 / lr_C1.0 | ±0.00000 | identical |
| gbm_shallow | −0.00055 | worse |
| gbm_deep | −0.00077 | worse |
| **gbm_calibrated** | **−0.00106** | **worst — a third of the entire edge** |

**Three things worth keeping:**

1. **V4's regularisation lesson did NOT transfer.** C = 0.03, 0.1 and 1.0
   give Brier skill 0.00318 on home runs, identical to five decimals. V4
   found C mattered (0.007) because V4 had 28 features to shrink. V5 has
   8. A lesson learned on one feature set does not carry to another.

2. **Isotonic calibration of rare events is a known hazard here, now
   measured twice.** V4: −0.045 Brier skill, p=1.000 emitted for 16 test
   games. V9: worst model on home runs. Each cv fold fits a step function
   to a few hundred positives and over-fits its own calibration curve.

3. **The received wisdom about boosting was backwards on this data.**
   Expected: ranks better, calibrates worse. Measured on is_k: ranks
   slightly worse (AUC 0.5957 vs 0.5968), calibrates much better
   (ECE 0.0038 vs 0.0079).

### A methodology bug worth remembering
`model_lab` initially reported two models as BETTER: +0.00001 and
+0.0000004 skill, the latter with a CI of [0.00000, 0.00000]. Both were
statistically real — C=0.03 and C=0.1 are nearly the same model, so the
difference is consistent across 48,000 rows and never changes sign — and
both were meaningless.

**Significance measures whether an effect is DETECTABLE; it says nothing
about whether it MATTERS.** With enough rows the first is satisfied by
anything. Both labs now carry a floor: `model_lab` needs a gain above
0.0005 to say BETTER (otherwise "negligible"), and `feature_lab` reports a
minimum detectable effect so that a null result can be distinguished from
an underpowered one.

### THE STANDING NEXT IDEA — point the K model at PITCHERS
Per-PA Brier skill by target: **is_k 0.0187**, is_walk 0.0071,
is_hr 0.0032, is_hit 0.0007.

The strikeout model is **six times more predictive than the home run
model** — comfortably the strongest component in the project, because
strikeouts depend least on defence, luck, and the other eight hitters.

It is currently aimed at a batter strikeout prop that most books do not
post. **Pitcher strikeout totals (over 5.5, over 6.5) are a liquid
market** and use the same per-PA rate compounded over batters faced
instead of plate appearances.

Note for that session: `gbm_shallow` on is_k halved calibration error
(ECE 0.0079 -> 0.0038) with a positive but unproven skill gain. Better
calibration matters more for a count prop than for a binary one.

---

## V8 (Aug 18, 2026) — the prop is HITS + RUNS + RBI, not hits + RBI + walks

Nolan corrected the definition: sportsbooks price **H+R+RBI**. Walks are
not in it. This was not a rename — runs scored are the one component that
does not belong to the plate appearance it comes from, and Statcast does
not record them at all.

### Why runs needed real work
Statcast's only trace of a run is free text in `des` ("Ben Rice scores.")
which names the runner but not his id — the accented-name problem again.
So official MLB boxscores became a data source (`data/batting_lines.py`,
306,291 lines across 13,854 games, ~30 min one-time backfill).

And a run cannot be attributed to a plate appearance: the batter reaches
base in PA 2 and crosses the plate during PA 5, somebody else's. Two of
the three components attach cleanly; the third does not.

### The decomposition
```
P(score in a PA) = P(home run)                    <- certain, entirely his
                 + P(reached base otherwise) x q  <- needs the rest of the order
```
25.5% of a hitter's runs are his own home runs. The other 74.5% flow
through `q`, modelled in `features/run_features.py` as
`q_league x slot_factor x obp_adjustment x batter_factor`.

**Measured, on 306k boxscore lines / 249k confirmed-starter games:**
| Quantity | Value |
|---|---|
| P(score \| reached base, non-HR) | **0.3034** |
| Slot factors | 1.161 (slot 1) → 0.914 (slot 4) → 1.085 (slot 9) |
| Per-batter factor, true s.d. | 0.094 (raw 0.120, of which 0.075 is noise) |
| Split-half correlation (odd/even games) | **0.50** — a real, persistent trait |
| Shrinkage prior | ~260 non-HR times on base |

The slot curve is U-shaped because slots 1 and 9 have the top of the order
batting behind them and slot 4 has the bottom.

### The confound that had to be checked, and wasn't ignorable
`obp_behind` correlates with the slot scoring rate at **r = 0.90** across
the nine slots, which looks like it explains the whole U-shape. It does
not. Identifying the effect only from team-to-team variation at the SAME
lineup position collapses the slope from +1.05 to **+0.367** — still real
(t = 5.2) but a third the size. Most of the slot pattern is *what kind of
hitter bats there* (leadoff hitters are fast), not who follows him.

So both terms are kept, and the slot term is documented as DESCRIPTIVE:
move a slow slugger to leadoff and the model hands him some speed he
doesn't have. `batter_factor` is estimated as a ratio to *his own slots*
precisely so that much isn't double-counted.

### No RNG anywhere
The per-PA contribution shape folds the run in **analytically**:
```
reached base, no HR -> value      w.p. 1-q
                    -> value + 1  w.p. q
everything else     -> value      (exact already)
```
The alternative — randomly picking which PAs "scored" — would need a seed
and add noise for the same mean. `per_pa_contribution_distribution` now
takes `score_prob` and does this as a mixture.

### Bugs caught during the swap
- **`fit` dropped rows with zero official times on base.** Those rows
  still contain runs (reached on error / fielder's choice, ~1.5% of
  player-games). Dropping them discarded runs while discarding nothing
  from the denominator: total predicted runs came out **−1.63%**. Keeping
  them: **+0.19%**, worst slot error 2.42% → 1.26%.
- **`walks` and `hbp` collided in the merge.** `game_totals` already has
  them, so pandas suffixed BOTH sides to `_x`/`_y` and the column vanished
  under a name nothing looked for. *The Part-2 test missed this because
  its fixture was tidier than the real input* — it built a stand-in frame
  with only slots, so there was nothing to collide with. Fixture now
  carries the real columns, plus an explicit collision guard that raises.
- **Partial boxscores were cacheable.** A game in progress returns a
  perfectly valid partial line with nothing marking it incomplete, and the
  cache is keyed on game_pk alone — so it would freeze in forever. Added
  `MIN_FINAL_PA = 54` and a `refetch=` escape hatch that `score_slate.py`
  uses for the date it is grading.
- **Two functions computing the same composite.** `attach_batting_lines`
  duplicated what became `build_run_training_frame`. Deleted; one owner.

### End-to-end verification (`test_hrr.py`, 35-player subset)
| | predicted | actual | gap |
|---|---|---|---|
| expected H+R+RBI per PA | 0.4799 | 0.4799 | 0.0000 |
| P(over 0.5) | 0.6818 | 0.6903 | −0.0086 |
| **P(over 1.5)** | **0.4820** | **0.4802** | **+0.0018** |
| P(over 2.5) | 0.3253 | 0.3199 | +0.0054 |
| P(over 3.5) | 0.2015 | 0.1975 | +0.0041 |

Total predicted runs across 249k player-games: +0.19%, no lineup slot off
by more than 1.26%.

### Sanity check against a real book
All 306k lines give mean 1.548 and P(over 1.5) = 0.394, but that includes
pinch-hitters with one PA. **Restricted to confirmed starters: mean 1.751,
P(over 1.5) = 0.446**, rising to ~0.51 at the top of the order — which is
the right neighbourhood for a book pricing over 1.5 near even money on a
projected starter. The all-inclusive figure was sampling, not definition.

### Files touched
`features/pa_table.py` (hrbw → hrr_certain + reached/reached_nonhr),
`features/run_features.py` (NEW), `model/compound.py`, `predict_slate.py`,
`train_props_v5.py`, `score_slate.py`, `dashboard.py`,
`data/batting_lines.py`, `test_hrr.py` (NEW).

### V8b — the wide pool exposed a latent starter/bullpen bug
The first live V8 run printed **"starter throws 22.8% of a hitter's plate
appearances"**, against a module constant of 52.8%. Two stacked bugs in
`data/pitcher_data.measure_bullpen_rates`:

1. **A game has TWO starters, one per side.** It grouped by `game_pk`
   alone and took the first pitcher, which identifies whoever faced the
   AWAY team. Every plate appearance by a home-team hitter against the
   away starter was counted as bullpen.
2. **"First row we hold" is not the starter.** `at_bat_number` and
   `inning` were never in `PA_COLUMNS`, so the sort was a no-op and
   `.first()` returned the pitcher faced by the lowest-numbered *batter* —
   frequently a middle reliever. And where the pool doesn't contain the
   game's opening plate appearance at all, there is no first-row answer to
   find.

**Why it stayed hidden until now:** a 60-hitter pool rarely contains both
teams from the same game (measured at 19.5% overlap on an 18-hitter
sample), so whichever side was cached, its opposing starter was usually
the one picked. At 270 hitters covering all 30 teams the overlap is near
total, and the estimate collapsed. *The bug didn't appear when the pool
widened — it was always there, and widening the pool stopped hiding it.*

**Consequence:** `blend_with_bullpen` weights tonight's actual starter by
this share, so the one pitcher whose identity is known was getting <half
his proper influence, while the "league reliever" rates he was blended
against were contaminated with starter plate appearances.

**Fix:** the starter is the pitcher who threw in **inning 1 to that side**
— robust to partial pool coverage — plus a plausibility guard that rejects
any measured share outside 0.35–0.75 and falls back loudly. Validated by
the usage decay: 99.8% of inning-1 PAs are against the starter, 45.2% by
the sixth, 1.2% by the eighth. `test_hrr.py` Part 3 recovers a known 60.0%
share from a synthetic slate and confirms both failure modes fall back
rather than guessing.

`inning` and `at_bat_number` added to `PA_COLUMNS`.

### V8c — PA projection inverts the top of the order (FOUND, NOT FIXED)
Caught on the 2026-08-18 slate once all 15 lineups were confirmed. The PA
model projects **slot 1 below slot 2** (4.270 vs 4.341).

That is structurally impossible as an average. Slot 1 bats ahead of slot 2
in every inning, so over a game he gets *at least* as many plate
appearances — the difference can be zero, never negative. Measured across
13,853 games of boxscores it is +0.088.

Against those same boxscores the whole curve is mis-shaped:

| slot | actual | predicted | gap |
|---|---|---|---|
| 1 | 4.294 | 4.270 | −0.024 |
| 2 | 4.206 | 4.341 | **+0.135** |
| 3 | 4.111 | 4.113 | +0.002 |
| 4 | 4.023 | 4.070 | +0.047 |
| 5 | 3.902 | 3.835 | −0.067 |
| 6 | 3.782 | 3.644 | −0.137 |
| 7 | 3.653 | 3.601 | −0.051 |
| 8 | 3.500 | 3.242 | **−0.258** |
| 9 | 3.354 | 3.179 | −0.175 |

Predicted spread 1.162 PA against an actual 0.940 — the model
**over-separates** the order and inverts the top two. This matters more
than its size suggests: expected PA is the largest single term feeding
every probability underneath it.

**Root cause:** `PA_FEATURES` contains `lineup_slot_sq`, so the fitted
curve is a **parabola** in slot — and a parabola turns over. Its vertex
landed near the top of the order.

The quadratic is not the villain and deleting it is not the fix: the real
decline genuinely accelerates down the order (−0.088 PA per slot at the
top, −0.15 at the bottom), curvature a linear term cannot represent. The
problem is that the curvature is *unconstrained*.

**Cost, measured** (typical hitter at 0.45 H+R+RBI per PA, on the over-1.5
line): slot 2 +1.2 pts, slot 6 −1.4, slot 8 **−2.6**, slot 9 −1.8. A
systematic 2.6-point bias on a whole lineup slot is the same order of
magnitude as the model's entire measured edge (~0.018 Brier skill).

**FIXED — `enforce_slot_monotonicity` in features/pa_projection.py.**
Keeps the quadratic, then projects each team's nine expected values onto
the nearest non-increasing sequence by pool-adjacent-violators. It is the
minimum change satisfying the constraint, preserves the team's total
expected PA, and leaves an already-monotone lineup completely untouched.
Distributions are then re-matched to the corrected mean by exponential
tilting, which is the maximum-entropy adjustment — the model's opinion
about SPREAD survives while its opinion about LEVEL is corrected.

PAVA is written out rather than imported from sklearn so the constraint is
readable in the file that depends on it.

**Same species as the `project_lineup` permutation fix**: a fact obvious
to a person, invisible to a regression, cheap to assert once noticed.

**What this does NOT fix.** Isotonic projection enforces ORDERING, not
LEVEL. The over-separation remains — slot 8 still projects 3.242 against
an actual 3.500, and the predicted spread is still ~1.16 PA against an
actual 0.94. That needs either better features or an empirical per-slot
calibration, and it should be decided with scoring data in hand rather
than by fitting harder to the same historical average.

### V8d — predict_slate now REFUSES to overwrite a clean prediction
The old guard printed a warning, only for past dates, then carried on.
Backwards: the dangerous moment is *tonight*, minutes after a good
pre-game run, when re-running to try a code change silently overwrites the
only evidence the model committed without hindsight. By the time the date
counts as "past", it is already gone.

Now it reads the existing file, and if `predicted_at_utc < start_time_utc`
it refuses and points at `score_slate.py`. `--force` overrides. Flags are
filtered out of `sys.argv` before the date is read, so
`predict_slate.py --force` doesn't parse "--force" as a game date.

Verified against the live 2026-08-18 file: refused, "written 17 min before
first pitch", file untouched.

### Still open
Team-specific bullpen quality (all bullpens still get league reliever
rates). The residual ±1.2% slot pattern is `obp_behind` slightly
over-amplifying the U-shape on top of the slot factor — small, understood,
not corrected. And the PA monotonicity fix above.

---

## V7 (Aug 18, 2026) — live slate runner, dashboard, and honest scoring

`predict_slate.py` → `dashboard.py` → `score_slate.py`. Run it for a date,
look at it, then score it after the games.

**First live slate (2026-08-18):** 270 hitters, 15 games, 525k plate
appearances of history. HR probabilities spanned **0.040 to 0.344** — an
8.6x spread, versus near-nothing in the 9-slugger era. Implied per-PA
rates check out against real baseball (Goodman 8.3%, Ohtani 7.7%,
Alika Williams 1.2%; league average is 3.5%).

**Pitcher data works.** Pulled per slate (~30 starters, not a bulk pull) —
each `statcast_pitcher` call covers every batter he faced, so a starter
goes from ~20 observations to ~600. Adding it visibly reordered the board:

| Hitter | before | after | facing |
|---|---|---|---|
| Schwarber | 0.302 | **0.220** | Cade Gibson (was TBD) |
| Olson | 0.259 | **0.310** | Zebby Matthews |
| Alonso | 0.282 | 0.247 | Rodón |
| Encarnación-Strand, O'Neill | — | dropped out | Rodón |

Three different Orioles facing Rodón all moved the same direction. That's
a coherent matchup effect, not noise.

### THE BACKTEST IS MILDLY OPTIMISTIC — and now there's a clean test

Several choices (regularisation C, the overdispersion fix, the shape
method, when to stop) were made while looking at test-set results. That's
researcher degrees of freedom; it inflates measured skill even unintentionally.

`score_slate.py` closes the loop: predictions are written BEFORE the games,
scored after, and appended to `cache/scoring_log.csv` with a
sample-size-weighted running total across slates. **One slate is noise —
~270 hitters, ~60 homers. Read the running total, not the day.** Expect the
first 15-20 slates to bounce around meaninglessly. If live skill lands
meaningfully below the backtest's +0.018 to +0.028, the backtest was
optimistic, and that is worth writing down rather than explaining away.

### Bugs found and fixed in this phase (all silent-failure shaped)
- **Park factors always neutral.** Matched team abbreviations as substrings
  of venue names; `"col"` is not in `"coors field"`, so every game silently
  used 100. Now reads the home team off the slate. Coors correctly 118.
- **Staleness could never be satisfied.** Compared newest cached game
  against the FUTURE slate date, so all 270 players re-pulled every run
  forever. Now tracks when each player was last *checked*
  (`cache/refresh_log.csv`) — an injured hitter with 20-day-old games is
  current if we checked today. Re-running a slate is now free.
- **Shape-method selection scored hybrids.** `method` was threaded into the
  outer reshaping call but not the inner one, so the two methods were never
  actually compared. Now explicit, plus bootstrap selection.
- **Dashboard served stale numbers** after a re-run (cached on filename
  only; now on modification time).
- **`%-I` is POSIX-only** and crashed on Windows. Replaced with arithmetic.
- **Games sorted by `game_pk`**, an internal id unrelated to start time.
  Now by first pitch.

### Environment note
The venv now lives at `C:\venvs\baseball`, deliberately OUTSIDE OneDrive.
A venv inside a synced folder gets its files locked by OneDrive — that's
what corrupted the original one mid-delete. Code stays in OneDrive (worth
backing up); the venv doesn't (regenerable from requirements.txt).
`requirements.txt` is now UTF-8; PowerShell's `pip freeze >` writes UTF-16,
which pip cannot read.

---

## V6 (Aug 18, 2026) — wide pool + lineup slots. The pool WAS the bottleneck.

60 stratified hitters (elite / middle / replacement by OPS) instead of 9
sluggers. 143,310 plate appearances, 35,917 player-games, real batting-order
slots from MLB's boxscore feed.

**Nearly everything improved, some of it dramatically:**

| Prop | 9 sluggers | 60 hitters |
|---|---|---|
| ≥1 HR, Brier skill | +0.0074 | **+0.0276** |
| ≥1 HR, AUC | 0.571 | **0.642** |
| ≥1 hit, AUC | 0.541 | 0.575 |
| K per PA, AUC | 0.558 | **0.604** (skill +0.0218) |
| H+RBI+BB over 1.5 | +0.0074 | **+0.0162** |
| — its 95% CI | [+0.001, +0.012] | **[+0.011, +0.021], P(>0)=1.000** |

Home run prediction roughly **quadrupled** in skill. The V4 conclusion
that HR was near an unbreakable ceiling was true *of that pool* — the
ceiling was the roster, not the sport. A model can only rank hitters apart
if the pool contains hitters who differ.

### The shrinkage estimator independently reproduced published sabermetrics

Estimated prior strength `k` (PA needed before a hitter's own rate is
trustworthy), from beta-binomial method of moments — versus Russell
Carleton's published stabilization points, which this project never used:

| Rate | our `k` | published | 
|---|---|---|
| Strikeout | **64 PA** | ~60 |
| Home run | 239 PA | ~170 |
| Walk | 288 PA | ~120 |
| Hit (BA) | **1020 PA** | ~910 |

Two land almost exactly. That is a strong validation of the estimator: it
was derived from first principles here and recovered numbers established
independently elsewhere. It also explains the whole project in one line —
**strikeout rate stabilises 16× faster than batting average**, which is
why K and BB models work and hit models don't.

### The lineup-slot effect is real and large

| Slot | mean PA | % with 5+ PA |
|---|---|---|
| 1 | 4.40 | 47.9% |
| 5 | 4.04 | 22.2% |
| 9 | 3.51 | 7.7% |

Nearly a full plate appearance between leadoff and ninth. The PA model
beats the batter's own recent average by 0.050 PA/game (MAE 0.568 vs
0.618), correlation with actual PA +0.375.

### CORRECTION to the V5 "12× leverage" claim

V5 reported that knowing the exact PA count was worth ~12× every skill
feature combined. That number was an **upper bound that included
unknowable information**, and the wide-pool run shows how much:

| Prop | historical PA | + PA model | actual PA |
|---|---|---|---|
| ≥1 hit | +0.0095 | +0.0107 | +0.0779 |

The PA model captured about +0.001 of a +0.068 gap. The remaining ~98% is
game length, blowouts and in-game removal — **genuinely unpredictable
before first pitch**. Lineup slot is real and worth having, but the "12×"
headline overstated what was reachable. Recorded as a correction rather
than quietly dropped.

### Over/under calibration: two separate bugs, found in sequence

**Bug 1 — game-level overdispersion (FIXED, verified on real data).**
Predicted P(over 0.5) = 0.813 vs actual 0.774. Convolution assumes plate
appearances are independent; real game totals have variance/mean = 1.448.
A dominant pitcher suppresses every trip, a bad one gets hit by everyone,
so totals clump and extra mass sits on zero. Fitted multiplier s.d. 0.359.
Result:

| Line | actual | before | after | err before | err after | skill before | skill after |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.7742 | 0.8131 | **0.7826** | +0.0389 | **+0.0084** | +0.0068 | **+0.0147** |
| 1.5 | 0.4981 | 0.5296 | **0.5068** | +0.0315 | **+0.0087** | +0.0162 | **+0.0191** |
| 2.5 | 0.2769 | 0.2938 | 0.2932 | +0.0169 | +0.0163 | +0.0144 | +0.0142 |
| 3.5 | 0.1439 | 0.1453 | 0.1579 | +0.0014 | +0.0140 | +0.0098 | +0.0083 |

Low lines improved 4.6× and 3.6×. Over-1.5 CI tightened to
**[+0.0150, +0.0230]**. But the 3.5 line got *worse* — which exposed:

**Bug 2 — exponential tilting inflates the tail (fix shipped, unverified).**
`scale_contribution_distribution` used exponential tilting to match a
batter's rate. Because the multiplier is exponential *in the value*, the
tail moves far more than the mean. For a hitter 25% above league:

```
value 0: 0.91x    value 2: 1.32x    value 4: 1.92x
```

A 25% better hitter credited with 92% more four-base plate appearances.
And it's applied twice — once for the batter, once per dispersion
multiplier — so it compounds. That is the +0.014 residual at 3.5.

New default method `"frequency"`: scale P(v > 0) to hit the target mean,
leave P(v | v > 0) exactly as observed. Closed form, and it preserves the
4s-to-1s ratio exactly (0.0184) where tilting inflated it 75% (0.0322).
The modelling claim: hitters differ mainly in how OFTEN they do damage,
not how spectacular it is when it lands. `select_tilt_method` picks
between the two by measured calibration on train — safe to select there
because both are zero-parameter structural choices, unlike a calibrator.
Verified on synthetic data: the selector recovers the true generating
method.

**Bug 3 — method inconsistency between the two reshaping points (FIXED).**
A distribution gets reshaped twice: once for the batter's own rate, once
per game-level dispersion multiplier. When `method` was added, the outer
call passed the selected method but the inner call silently used the
default. Every prediction changed, and the two methods became impossible
to compare fairly — the selector was scoring hybrids, not methods. The
run that "chose tilt by 0.0003" was measuring nothing. `method` is now an
explicit parameter on `compound_count_distribution_overdispersed` with no
clever default fallback.

**Selection made honest.** A 0.0003 margin is inside the noise, so
`select_tilt_method` now bootstraps over games (300 resamples) and asks
how OFTEN each method calibrates better. The tie-break is deliberately
asymmetric toward `frequency`: it reuses the observed tail ratio, while
`tilt` extrapolates it, and extrapolation should have to earn its place.
Same philosophy as the one-standard-error rule on calibrator selection.
Validated both ways on synthetic data — data generated by `frequency` is
identified 99% of the time, data generated by `tilt` 95% of the time.

### Superseded note (kept for the record)

Predicted P(over 0.5) = 0.813 against actual 0.774; over 1.5, 0.530 vs
0.498. Cause: convolution assumes plate appearances are independent, but
game totals have variance/mean ≈ 1.50, not 1.0 — a dominant pitcher
suppresses every trip, a bad one gets hit by everyone. That shared
game-level component puts extra mass on zero that independence misses.
Fixed by `fit_game_dispersion` + `compound_count_distribution_overdispersed`
(method-of-moments multiplier, mixed over a 5-point grid). On synthetic
data matched to the same 1.50 ratio, it cuts the low-line error ~3×
(+0.033 → −0.011). **Not yet verified on real data.**

### Compounding now wins on 2 of 3 binary props
hits +0.0007, walks +0.0039, HR −0.0036 — a flip from the 9-player pool,
where it lost on all three.

---

## V5 (Aug 17, 2026) — one engine, every prop

**The structural change:** stop building a classifier per prop. Estimate a
**per-plate-appearance rate**, then **compound** it over tonight's PA count.

```
ESTIMATE  per-PA probability      features/rate_features.py   (statistics + ML)
COMPOUND  over tonight's PA       model/compound.py           (exact arithmetic)
```

Why: measured on this data, **98.5% of the game-to-game variation in "did
he homer" is binomial noise** from getting only ~4 tries. A simulation
giving a model *perfect* knowledge of every game's true rate caps out at
**AUC 0.59–0.63**. V4's 0.553 was already ~90% of the way to that wall.
The ceiling is arithmetic, not effort.

Modelling at the PA level gives 29,999 rows instead of 6,667 — same data
pull, 4.5× the sample, and the target is the repeatable event rather than
a noisy summary of several.

### How repeatable is each skill? (split-half r on per-PA rate)

| Per-PA rate | Split-half r |
|---|---|
| Hit | **+0.916** |
| Walk | **+0.890** |
| HR | +0.636 |
| H+RBI+BB | +0.605 |

Shrinkage strength `k`, estimated from the data by beta-binomial method of
moments (not guessed): walk 129 PA, K 122 PA, hit 461 PA, **HR 592 PA**.
Home run rate needs ~5× the sample of walk rate before an individual's
number means anything.

### Results (test: 7,497 PA / 1,725 games from 2025-06-19)

Per-PA models:

| Target | AUC | Brier skill |
|---|---|---|
| walk | 0.5722 | +0.0063 |
| strikeout | 0.5578 | +0.0082 |
| home run | 0.5402 | +0.0005 |
| hit | 0.5109 | −0.0001 |

Game level:

| Prop | AUC | Brier skill |
|---|---|---|
| **At least 1 walk** | **0.604** | **+0.030** |
| **H+RBI+BB over 1.5** | 0.556 | **+0.0074** |
| H+RBI+BB over 2.5 | 0.560 | +0.0072 |
| At least 1 hit | 0.541 | +0.0039 |
| At least 1 HR | 0.571 | +0.0074 |

**H+RBI+BB over 1.5 is the first result in this project whose bootstrap CI
excludes zero**: +0.0074, 95% CI [+0.0011, +0.0124], P(skill > 0) = 0.990.
Calibration is excellent — predicted 0.597 vs actual 0.589 at the 1.5
line, 0.364 vs 0.358 at 2.5, 0.199 vs 0.198 at 3.5.

**Walks remain the best target in the project by a wide margin** —
Brier skill +0.030 is roughly 5× anything the HR work ever produced.
**Strikeout rate is a new find** and looks similarly tractable; it was
never tested before V5.

### Honest negative: compounding LOSES to direct modelling on binary props

| Prop | compound | direct | gap |
|---|---|---|---|
| ≥1 HR | +0.0016 | +0.0074 | −0.0058 |
| ≥1 hit | +0.0026 | +0.0039 | −0.0013 |
| ≥1 walk | +0.0272 | +0.0304 | −0.0032 |

For yes/no questions, a game-level classifier on the same features beats
compounding. Compounding's real value is that it answers questions a
classifier **structurally cannot** — every over/under line at once from
one distribution — and that's where the project's best-verified result is.

### THE BIGGEST FINDING: plate appearances are the dominant lever

Every model conditions on how many PA a hitter gets, and none predicts it.
Measuring what that costs:

| Prop | historical PA | **actual PA** | value of knowing PA |
|---|---|---|---|
| ≥1 hit | +0.0026 | **+0.0485** | **+0.0459** |
| ≥1 walk | +0.0272 | +0.0540 | +0.0268 |
| ≥1 HR | +0.0016 | +0.0172 | +0.0156 |

Knowing the exact PA count is worth **~12× the combined contribution of
every batter-skill feature in the project**. Batter identity explains only
0.9% of PA variance here — but that's a pool artifact (all 9 bat 1st–4th).
**Lineup slot and home/away are knowable before first pitch** from MLB's
lineup feed; game length and blowouts are not. This is the single highest-
value thing left to build.

### New files
`features/pa_table.py` (PA-level table, RBI from score deltas) ·
`features/rate_features.py` (beta-binomial shrinkage, PA-denominated
windows, log5 matchup) · `model/compound.py` (convolution, exponential
tilting, PA marginalisation, leverage diagnostic) · `train_props_v5.py`

---

## V4 REFRAME (Aug 17, 2026) — calibration, not classification

The HR branch changed GOAL, not target. V1–V3 chased a classifier that
could call a home run correctly and topped out at AUC 0.565. That ceiling
is not an engineering failure to be fixed — it's a property of the
problem. A single-game HR is a rare event dominated by an error term
(exact pitch location, timing to the millisecond, where the ball is
caught) that no observable feature set explains. Aiming for "60%+
confident" was chasing a number the data does not contain.

V4 asks the answerable question instead: **is the PROBABILITY honest?**
When the model prints 23.6%, does it happen about 23.6% of the time? A
well-calibrated 24% is a genuinely useful output; a falsely confident YES
never was.

**The metric changed accordingly.** AUC is still reported for continuity,
but the headline is now the **Brier skill score** — the model's Brier
score against the honest null of always predicting the base rate.
Positive = the features add real information. Zero = the model is
decoration on the base rate. This distinction matters here specifically:
the public-site screenshot that prompted V4 showed a 23.6% HR probability
against a training-pool base rate of 23.6%, i.e. a number that may be
nothing but the intercept dressed up as a prediction.

### New in V4
- `features/contact_quality.py` — sweet-spot rate (LA 8–32°), barrel rate,
  hard-hit rate, exit velo on air balls only, pull-side air rate.
  Rationale is the project's own strongest finding: consistent SKILLS
  predict better than rare EVENTS. A hitter puts ~30 balls in play per 10
  games but hits ~2 HRs — same underlying ability, ~10x the sample. This
  is the walk-rate lesson applied to power.
- `data/game_context.py` — real game-time weather (temp, wind speed +
  direction), venue and day/night from MLB's own Stats API. Statcast
  carries neither weather nor start time. Wind is parsed from MLB's
  human string ("11 mph, R To L") into an out-to-CF component and a
  pull-side component (the latter flipped by batter handedness — a lefty
  pulls to right field, so a toward-LF wind hurts him). Cached to
  `cache/game_context.csv`, checkpointed every 100 games, resumable.
- `features/context_features.py` — park/weekday/home/day-night splits,
  every one **empirical-Bayes shrunk** toward the player's own baseline
  with a 30-game prior. This is the answer to "he has 5 HRs on Sundays":
  a 6-game split with 2 HRs reports +1.4% off baseline instead of +8%.
  The feature is the DELTA from baseline, so it collapses to ~0 when the
  evidence is thin, and a near-zero coefficient then means something
  clean instead of being confounded with small-sample noise.
- `model/hr_v4.py` — three-way CHRONOLOGICAL split (train 60% / calibrate
  15% / test 25%), isotonic calibration fit on the middle slice only
  (fitting it on train is circular, on test is leakage), Brier + Brier
  skill + log loss, equal-count reliability table, standardised
  coefficient report, and a **permutation test** that shuffles a feature
  group and refits to build a null distribution — so the day-of-week
  question gets a p-value rather than an opinion.
- `train_hr_v4.py` — runs the whole thing as a 5-rung ablation ladder,
  each rung adding one feature group, so every group's contribution is
  visible rather than assumed.
- `test_v4_synthetic.py` / `test_v4_dryrun.py` — synthetic-data tests with
  a deliberately planted signal (persistent AR(1) form, a temperature
  effect, one hitter-friendly park) and a deliberate NON-effect (day of
  week). Includes a direct no-lookahead assertion: game 11's
  `sweet_spot_rate_10` is checked to equal exactly games 1–10. The dry run
  executes the real `train_hr_v4.py` with only the three network loaders
  stubbed.

### RESULTS ON REAL DATA (9 players, 6,667 games, 2022–2026)

Test window 2025-06-28 → 2026-07-31, 1,512 games, base rate 23.5%.

| Feature set | AUC | Brier skill |
|---|---|---|
| 1. Rolling form only | 0.5476 | +0.0039 |
| **2. + fatigue + pitch matchup** | **0.5533** | **+0.0059** |
| 3. + contact quality | 0.5361 | −0.0312 |
| 4. + weather | 0.5314 | −0.0541 |
| 5. + calendar & park splits | 0.5340 | −0.0010 |

**The V3 feature set is still the best, and it does beat the base rate —
barely.** Rolling-origin check at four different cut points: +0.0057,
+0.0069, +0.0061, +0.0056 (spread 0.0013, AUC 0.5554–0.5701). Every window
positive and tightly clustered. Single-window bootstrap 95% CI is
[−0.0021, +0.0121], P(skill > 0) = 0.918 — not significant alone, but the
rolling-origin consistency is the stronger evidence.

**What +0.006 Brier skill actually buys:** the model spans 15.8%–42.2%
across test games, s.d. 0.029 — it moves the number about ±5.8% around the
base rate. Reliability is good (largest bucket gap 0.049), so those
percentages can be read at face value. This is a real but small edge.

### Two methodological findings that mattered more than any feature

**1. The first run's conclusion was a calibrator artifact.** V4 originally
defaulted to isotonic regression whenever the calibration slice cleared
400 rows. On 908 rows at a 22% base rate it emitted p = 1.000 for 16 test
games (4 were HRs) and p = 0.000 for 10 more. Cost: Brier skill −0.003
(raw) → −0.045 (isotonic). The initial "nothing beats the base rate"
verdict was substantially the calibrator, not baseball. Fixed by choosing
the calibrator on a held-out half of the calibration slice, with a
**one-standard-error rule** — the simplest candidate within 1 SE of the
best wins, so complexity has to clear a bar rather than win a coin flip.
Without the 1-SE rule isotonic still won several rungs by margins smaller
than their own error bars.

**2. Default regularisation was overfitting.** `C=1.0` vs `C=0.003` on the
full feature set was worth ~0.007 Brier skill — the same order as the
entire real signal. Default is now `C=0.03`, selectable on the calibration
slice via `MODEL_C = None`.

### Feature findings

- **Contact quality: right hypothesis, no incremental value.**
  `career_barrel_rate` is the single most informative column in the whole
  feature set on its own (univariate AUC 0.5431), edging out
  `career_hr_rate` (0.5403). So the "measure the skill, not the rare
  event" idea was directionally correct. But adding the group made the
  model *worse* (+0.0039 → −0.0312) because barrel rate and HR rate
  measure the same latent power — ten collinear columns cost more in
  variance than they add in signal. Permutation test p = 0.841.
  **This is the chase-rate finding again**, in a new place: granular
  features aren't additive when they overlap an existing one.
- **Weather: real at the margin, useless in the model.** `temp_f`
  univariate AUC 0.5285 — genuinely above several kept features. But it
  adds nothing jointly (permutation p = 0.448). `wind_pull_mph` is
  univariate AUC 0.5004, i.e. exactly nothing, despite the handedness
  sign logic being verified correct.
- **Calendar & park splits: noise, as designed to reveal.** Permutation
  p = 0.617. After shrinkage, "5 HR on Sundays" contributes nothing — and
  now there's a number saying so rather than an opinion.
- **Data quality catch:** MLB's feed reported 0 °F for two dome games
  (Tropicana, Minute Maid). Now floored — anything ≤ 32 °F is treated as
  a missing reading.

### The answer to the question that started V4
A public card showing **23.6%** against a pool base rate of **23.6%** is
showing the base rate. A model with genuine signal on this problem spans
roughly 16%–42%, and the honest edge over "just quote the base rate" is
about half a percent of squared error.

---

## Current State (as of Jul 31, 2026)
- **Best working result: team win probability**, AUC 0.582 (opponent
  strength + rolling runs-allowed as a pitching-quality proxy). Real
  discrimination in both directions (not a majority-class collapse) —
  comparable in shape to a legitimate baseline, well below Vegas-level
  (~0.60–0.65) but a genuine, usable signal.
- **Best working player-level result: walk prediction**, AUC ~0.59–0.60.
  Real, if modest, signal — plate discipline is a repeatable individual
  skill, not a rare/bursty event.
- **HR prediction (single game): a working, if modest, version found.**
  Tried single player, multi-player (8 players), opposing pitcher + park
  factor, team-level opponent strength, and a loosened "HR in next 3
  games" target — every one of those landed at AUC ≈ 0.51–0.54. What
  finally moved it: a **pitch-type matchup feature** (this batter's own
  rolling HR rate against fastball/breaking/offspeed pitches, combined
  with the opposing starter's own season pitch mix into an "expected HR
  exposure" score) plus fatigue (rest days, games in the trailing 7 days)
  pushed AUC to **0.565** — the best HR result of the whole project, and
  the first one to show genuine two-directional discrimination (74%
  recall on HR games, 33% recall on non-HR games at a tuned decision
  threshold) instead of a majority-class collapse. Still modest — more
  useful as one input feeding a probability estimate than as a confident
  standalone call — but a real, working signal, not a dead end.
- **RBI prediction: dropped entirely.** AUC never moved off ~0.50–0.51
  across three attempts. Depends heavily on teammates reaching base
  first, which is outside a player's own control/recent form — no
  further lever identified worth pulling.
- **Chase rate feature (walk model): tested, no improvement** over
  rolling walk rate alone (AUC flat, ~0.59 either way). Likely because
  chase rate and walk rate measure overlapping underlying skill —
  informative on their own, redundant together for this model.

---

## How To Run
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Single-player HR baseline (V1)
python main.py

# Single-player "gets a hit" comparison test
python test_hit_prediction.py

# Single-player HR + opposing pitcher + park factor (V3)
python train_v3.py

# Multi-player HR baseline (8 players, no matchup context)
python train_multi_player.py

# Multi-player HR + team-level opponent strength (the "bridge")
python train_bridge.py

# Team win probability (30 teams, opponent strength + pitching proxy)
python train_team_win.py

# Multi-target player props: HR this game, HR next 3 games, walk (+ chase rate)
python train_multi_targets.py

# --- V4: calibrated HR probability ---
# Verify the pipeline on synthetic data first (fast, no network):
python test_v4_synthetic.py
python test_v4_dryrun.py

# Then the real thing. Set SKIP_WEATHER = True at the top of the script
# for a fast first pass that skips the ~1 call-per-game Stats API pull.
python train_hr_v4.py

# --- V5: per-PA rate engine, every prop from one model ---
# No new data needed -- runs entirely off the existing cache.
python train_props_v5.py
```

**First run of any script touching team data or a new player will be
slow** (pybaseball's built-in rate-limit delays). Subsequent runs load
from `cache/` instead — see Caching below. Delete the relevant `cache/*.csv`
file to force a fresh pull (e.g. after adding a new feature column that
isn't in the old cached data).

---

## Project Architecture
```
baseball_predictor/
├── cache/                          # on-disk cache, created automatically
│   ├── statcast_{First}_{Last}_{start}_{end}.csv   # per-player Statcast pulls
│   └── team_features_full.csv      # full 30-team feature table
├── data/
│   ├── loader.py                   # get_player_id (name lookup, with a fuzzy-
│   │                                # match fallback for accented names --
│   │                                # see Known Issues), load_batter_statcast,
│   │                                # load_batter_statcast_cached
│   ├── cache.py                    # load_cached / save_cache -- simple CSV
│   │                                # disk cache used by loader.py and
│   │                                # team_pipeline.py
│   ├── team_data.py                # load_team_season (schedule_and_record wrapper)
│   ├── pitching_data.py            # opposing-pitcher season stats computed
│   │                                # from Statcast directly (NOT FanGraphs --
│   │                                # see Known Issues). Used by train_v3.py only.
│   └── park_factors.py             # static HR park factor table (approximate)
├── features/
│   ├── build_features.py           # single-player game log + rolling features (V1)
│   ├── multi_player_features.py    # multi-player version: game log + rolling
│   │                                # features incl. chase_rate, keyed by batter
│   │                                # so rolling windows never cross players
│   ├── prop_targets.py             # add_forward_hr_target (loosened HR window);
│   │                                # add_rbi_target exists but is UNUSED (RBI dropped)
│   ├── fatigue_features.py         # rest_days, games_last_7_days -- free,
│   │                                # from data already pulled
│   ├── pitch_mix_features.py       # pitch-type matchup: batter HR rate by
│   │                                # pitch bucket (fastball/breaking/offspeed)
│   │                                # x opposing starter's own pitch mix --
│   │                                # the feature that finally moved HR's AUC
│   ├── team_features.py            # team game log, rolling win/run-diff/RA,
│   │                                # attach_opponent_features (the opponent-
│   │                                # strength join)
│   └── team_pipeline.py            # get_full_team_features -- cached wrapper
│                                    # around the full 30-team pull, shared by
│                                    # train_team_win.py and train_bridge.py
├── model/
│   └── train.py                    # FEATURE_COLS, train_test_split_by_time,
│                                    # train_hr_model (V1 single-player model)
├── main.py                         # V1 entry point (single player HR)
├── test_hit_prediction.py          # V1-style "gets a hit" test (standalone script)
├── train_v3.py                     # single-player HR + opposing pitcher + park
├── train_multi_player.py           # multi-player HR baseline (no matchup context)
├── train_bridge.py                 # multi-player HR + team opponent strength
├── train_team_win.py               # team win probability, full pipeline
├── train_multi_targets.py          # HR / HR-next-3 / walk, side by side
├── train_pitch_matchup.py          # HR + pitch-type matchup + fatigue --
│                                    # the best HR result found (AUC 0.565)
├── debug_schedule.py                # diagnostic used to find the team dropna() bug
├── debug_alvarez.py                 # diagnostic used to find the accented-name bug
├── requirements.txt
└── CONTEXT.md                       # this file
```

---

## Data Sources
- **pybaseball** (free) throughout — no paid API. Originally scoped around
  Sportradar (paid, has a Statcast package) but switched to pybaseball
  since the goal was a free build; pybaseball's Statcast wrapper gives
  equivalent pitch-level data (exit velo, launch angle, zone, etc.) at no
  cost.
- **Player data**: `statcast_batter()` — pitch-by-pitch, per player.
- **Team data**: `schedule_and_record()` — per-game team results, scraped
  from Baseball-Reference.
- **Opposing pitcher stats** (`train_v3.py` only): computed directly from
  each pitcher's own `statcast_pitcher()` pull (K-rate, BB-rate, HR-rate,
  avg exit velocity allowed) — **not** FanGraphs. `pybaseball.pitching_stats()`
  (FanGraphs-backed) currently returns a 403 on every call; this is a
  known, unresolved issue in pybaseball itself (FanGraphs anti-bot
  protection), not something fixable from this project's code. If it gets
  fixed upstream, `data/pitching_data.py` could be simplified back to a
  single FanGraphs call, but there's no urgency — the Statcast-based
  replacement works and avoids a second broken dependency.

---

## Caching
`data/cache.py` provides a simple CSV-based disk cache. Two things get
cached:
1. **Per-player Statcast pulls** (`load_batter_statcast_cached`) — keyed
   by player name + date range.
2. **The full 30-team feature table** (`get_full_team_features`) — one
   combined file, since building it requires ~90 rate-limited requests.

**Cache invalidation is manual** — there's no automatic staleness check.
Delete the relevant file(s) in `cache/` to force a fresh pull:
- After adding a new feature column upstream (old cached data won't have
  it, and code expecting the column will crash with a `KeyError`) — this
  bit us once already when chase_rate was added without clearing the
  player cache first.
- If you want genuinely current data for a season still in progress.

---

## Key Technical Decisions & Fixes
- **pandas 3.0 compatibility**: pybaseball's `schedule_and_record()`
  internally uses a chained-assignment `inplace=True` pattern that
  silently no-ops under pandas's Copy-on-Write behavior (mandatory as of
  pandas 3.0, can't be disabled). Fixed by pinning `pandas<3.0` in the
  venv — not fixable from this project's code, since the bug is inside
  pybaseball itself.
- **Team model `dropna()` bug** (found and fixed): an early version of
  `add_team_rolling_features` called a bare `.dropna()`, which wiped out
  ~99% of rows because `schedule_and_record`'s raw columns (`Inn`, `Save`,
  etc.) are legitimately sparse for reasons unrelated to the engineered
  features. Fixed by scoping `dropna(subset=...)` to only the columns this
  project actually built.
- **Accented-name player lookup** (found and fixed): `playerid_lookup`
  does an exact string match against the Chadwick ID registry, which
  stores some names accented even in the *last name* field (e.g. Yordan
  Álvarez, not just displayed but stored as "álvarez"). An unaccented
  exact search returns nothing. Fixed via a `fuzzy=True` fallback in
  `get_player_id`.
- **Opponent-strength join (team model)**: requires real calendar dates,
  since two different teams' schedule pages need to be matched to the
  *same* game. `schedule_and_record`'s `Date` column has no year and uses
  ambiguous doubleheader suffixes (`(1)`/`(2)`) — parsed manually in
  `parse_schedule_dates()`.
- **Lookahead-bias guard**: every rolling feature across every script is
  `.shift(1)` before use — a game's features never include that game's
  own outcome. Same principle as Quantara's `Position = Signal.shift(1)`.
  The one deliberate exception: `add_forward_hr_target`'s forward-looking
  window is a *target*, not a feature — using future games as a label is
  correct (that's what's being predicted), the lookahead rule only
  applies to features peeking at the future.
- **Chronological train/test splits** throughout — never shuffled.
- **AUC over accuracy, always.** Repeatedly, accuracy looked fine (60–80%)
  while the model had actually collapsed to predicting the majority class
  every time (0.00 precision/recall on the minority class). AUC — which
  doesn't depend on a classification threshold — was the metric that
  caught this every time and is the one to trust across this whole project.
- **Decision threshold tuning (train_pitch_matchup.py only)**: once a
  feature combination (pitch-type matchup + fatigue) finally produced a
  real AUC improvement on HR, the default 0.5 classification cutoff was
  swapped for one tuned via Youden's J statistic (the ROC point
  maximizing true-positive rate minus false-positive rate), computed on
  training data only. This is what let the classification report show
  genuine two-directional discrimination (74% HR recall, 33% no-HR
  recall) instead of hiding real signal behind a majority-collapse
  artifact. Not applied to the other scripts, since their underlying AUC
  was too close to 0.50 for a better threshold to reveal anything real.

---

## Key Research Findings
- **Consistent individual skills predict better than rare/bursty events.**
  Walk rate (plate discipline — a stable, repeated behavioral tendency)
  produced the best player-level AUC (~0.59). HR (a rare, high-variance
  event even for players with genuinely strong underlying power) and RBI
  (dependent on teammates, not just the player) never moved meaningfully
  off a coin flip, regardless of what was tried.
- **More data alone doesn't fix a weak-signal target.** Multi-player HR
  (8 players, ~3,800 games) nudged AUC from 0.538 (1 player) to ~0.54 —
  real but marginal. The problem was never really sample size.
- **Matchup context helps team-level predictions far more than
  player-level ones -- UNLESS the matchup feature is the right SHAPE.**
  Team win probability jumped from AUC ~0.50 (own form only) to 0.582
  (adding opponent strength) — both QUALITY-average features (is this
  team/pitcher good overall). The same quality-average approach applied
  to player HR prediction (opposing pitcher in `train_v3.py`, then team
  strength in `train_bridge.py`) produced no improvement either time.
  What finally worked for HR was a STYLE-fit feature instead of a quality
  average: this batter's own HR rate against fastball/breaking/offspeed
  pitches, combined with the opposing starter's own pitch mix (see
  `features/pitch_mix_features.py`). Quality averages dilute a specific
  pitcher's relevant trait into a team-wide or career-wide blend; a
  style-fit feature asks a much more targeted question ("does today's
  opponent's specific approach play into this hitter's specific
  strength") that turned out to carry real signal (AUC 0.541 → 0.565).
  Fatigue (rest days, trailing-week workload — free from data already
  pulled) was combined in the same test and likely contributed some of
  that gain too, though the two weren't isolated from each other.
- **Loosening a noisy target doesn't automatically help.** "HR in next 3
  games" (vs. "HR this game") made the positive class more common
  (53.8% vs 23.6%) but didn't improve AUC (0.51–0.53) — just moved which
  direction the model's majority-collapse tendency pointed.
- **Granular features aren't always additive.** Chase rate (built from
  every pitch's swing decision) seemed like it should out-inform walk
  rate (built only from full plate-appearance outcomes), but added no
  measurable AUC gain — likely because the two measure overlapping
  underlying plate-discipline skill.

---

## Known Issues / Technical Debt
- **RBI target code still present but unused** — `add_rbi_target` in
  `features/prop_targets.py` was left in the file after RBI was dropped
  as a target. Harmless dead code; fine to delete later if desired.
- **3 of 30 team-seasons fail to pull** in `train_team_win.py` — almost
  certainly one or more of the 4 uncertain abbreviations (CHW, KC, SD,
  SF, TB, WSH have Baseball-Reference-specific alternates: CWS, KCR, SDP,
  SFG, TBR, WSN). Not chased down since the model already works well on
  the ~87 that do succeed; worth fixing if pursuing full data completeness.
- **Debug scripts left in the project root** (`debug_schedule.py`,
  `debug_alvarez.py`) — used to diagnose the two bugs above, kept as
  reference/reusable diagnostics rather than deleted.
- **True starting-pitcher identity is not available anywhere in this
  project.** `train_v3.py` approximates it from the first pitcher a
  single batter faced each game (works, but only for a single-player
  pipeline). The team model uses a rolling runs-allowed proxy instead of
  real starter data, since `schedule_and_record` only gives the
  win/loss-decision pitcher (not reliably the starter), and getting real
  starters project-wide would require a much larger bulk Statcast pull
  across every game.
- **FanGraphs-backed pybaseball functions are broken project-wide** (not
  just for pitching stats) — `team_batting`, `team_pitching`, and
  `team_fielding` all hit the same 403 issue. None are currently used in
  this project, but worth remembering if extending it later.
- **`pandas<3.0` pin is a workaround, not a real fix** — depends on
  pybaseball eventually updating its own chained-assignment code for
  Copy-on-Write compatibility.

---

## Version Roadmap
### V1 ✅ — Single player, single stat (HR), logistic regression
### V2 ✅ — This version. Multi-player pool, opposing pitcher + park
  factor, team win probability (own form → opponent strength → pitching
  proxy), multiple player props tested side by side (HR / HR-next-3 /
  walk / RBI), caching layer, accented-name and pandas-3.0 compatibility
  fixes. RBI dropped as a target. HR-per-game initially looked like a
  dead end (every quality-average feature tried landed at AUC ≈ 0.51–0.54)
  until a pitch-type STYLE-fit matchup feature + fatigue pushed it to
  AUC 0.565 with real two-directional discrimination — the session's
  clearest lesson that feature *shape* (style-fit vs. quality-average)
  mattered more than feature *quantity*.
### V3 (possible future directions, not started)
  - Extend the "consistent skill" finding: test other repeatable-skill
    props (strikeout rate / contact rate) the same way walk rate was
    tested, since that category has been the one clear success.
  - Extend the "style-fit beats quality-average" finding: try a similar
    matchup-shaped feature for walk prediction (e.g. does this pitcher's
    pitch mix line up with counts this batter tends to work deep or
    chase in) rather than assuming walk rate alone is the ceiling.
  - Fix the 3 failed team-seasons (BR abbreviation variants) for full
    30-team data completeness.
  - True starting-pitcher data for the team model (would require the
    larger bulk Statcast pull deferred in V2).
  - If pybaseball's FanGraphs endpoints get fixed upstream, revisit
    `data/pitching_data.py` to simplify back to a direct FanGraphs pull.

---

## Libraries
- pybaseball — Statcast, Baseball-Reference data (free, no API key)
- pandas (pinned `<3.0` — see Known Technical Decisions)
- scikit-learn — LogisticRegression, metrics