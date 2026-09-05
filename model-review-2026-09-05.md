# Model review — 2026-09-05

> Status: everything in Tiers 1–3 was fixed the same day; Tier 2 is built
> behind flags with a lab. See `fixes-2026-09-05.md` (project) and the V10
> section of CONTEXT.md for what was done and what it measured.

Read: CONTEXT.md, all eight project docs, `predict_slate.py`, `score_slate.py`,
`rate_features.py`, `pa_table.py`, `compound.py`, `pa_projection.py`,
`pitcher_workload.py`, `pitcher_data.py`, `team_model.py`, `compare_market.py`,
`odds_lines.py`, `park_factors.py`, plus the scoring, pitcher-scoring, market
and form logs and the 9/01 and 9/04 slate/pitcher files.

## Where the model stands, pooled over the five scored slates

Hitter props, all rows (N = 1,179 per prop): H+R+RBI over 1.5 +0.026,
walks +0.024, hits +0.020, HR +0.013, TB over 1.5 +0.019. Clean rows only
(N = 567): walks +0.030, HR +0.019, everything else +0.006 to +0.017. That is
in line with the backtest's +0.018 to +0.028. The hitter engine is doing what
it said it would.

Pitcher strikeouts, 32 graded starts: Brier skill **−0.14 (over 5.5) and
−0.15 (over 6.5)**, predicted 5.11 K vs 4.2 actual on 9/01. Against the
market (n = 5): model Brier 0.361, market 0.251. The backtest said +0.06 to
+0.08. That gap is the first item below, and it is not a small-sample story.

---

## Tier 1 — deployed code that differs from what was validated

### 1. The K prop compounds a per-batter rate that was never backtested

`build_pitcher_props` walks the lineup with `p_is_k_vs_starter`, which is
the **hitter** logistic regression's output. Its pitcher input is
`pit_is_k_starter_only`, produced by `data/pitcher_data.build_pitcher_rates`
— a career total over the pitcher's entire cache from `START_DATE =
2022-01-01`, with no trailing window and shrunk toward a 2022–2026 league
mean. `WorkloadModel.k_rate(pid)` — the 12-month, 250-BF-shrunk rate the
K model doc validated — is computed and then only *exported* to the CSV.

On the 9/01 and 9/04 pitcher files, implied deployed rate
(`expected_k / expected_bf`) versus `k_rate`:

| pitcher | k_rate (12 mo) | deployed | starts |
|---|---|---|---|
| Gerrit Cole | 0.248 | 0.327 | 17 |
| Gavin Williams | 0.295 | 0.219 | 31 |
| Freddy Peralta | 0.233 | 0.311 | 31 |
| Blake Snell | 0.272 | 0.347 | 9 |

Spearman correlation between the two across 62 starters: **0.65**. The
deployed rate has 1.7× the spread (sd 0.049 vs 0.029). Wider spread from a
stale input is exactly the signature of the −0.14 skill: confident and
wrong at the extremes. The era-drift fix (+6.8% high on all-history,
measured in `pitcher_workload.py`) was applied to the model that is
displayed and not to the one that is compounded.

Fix, in order of directness:

- Compute the per-batter K probability as
  `log5(bat_is_k, workload.k_rate(pid), league_k_12mo)` — the pieces
  already exist — instead of routing through the hitter LR. Then the
  deployed number *is* the validated one.
- Or, minimum change: give `build_pitcher_rates` the same 12-month window
  (`as_of − 12 months`) and a 12-month league prior, so the hitter LR at
  least sees a current pitcher.
- Either way, re-run the rolling-origin backtest **on the code path
  `predict_slate` actually executes**. The K backtest script is not in the
  repo, so whichever path it used cannot be checked now; put it in.

### 2. PA projection: train/serve skew, and the parabola is still over-separating

Training (`add_pa_history_features`): `batter_pa_mean_20` is a 20-game
rolling mean; `team_pa_mean_20` is a 50-row rolling mean grouped by
`home_team` — which is the *venue's* team, so for an away batter it is the
opponent's number. Serving (`_project_pa`): `batter_pa_mean_20` is the
hitter's **all-time** mean PA per game including pinch-hit games, and
`team_pa_mean_20` is set equal to it. Two of five features mean different
things at fit time and at prediction time.

9/04 slate, expected PA by slot: 4.55, 4.42, 4.31, 4.21, 4.12, 3.95, 3.79,
3.64, 3.41. Measured on 13,853 games (V8c): 4.29 at leadoff, 3.35 at
ninth, spread 0.94. The model still gives leadoff +0.26 PA too many and
spreads 1.14. PAVA fixed ordering, not level — V8c said so.

Fix: drop the multinomial regression. Build `P(PA = k | slot, is_home)`
empirically from `cache/batting_lines.csv` (306k lines, 249k confirmed
starters — thousands of games per cell). It is monotone by construction,
has no parabola to turn over, and has nothing to skew. If team quality
matters, tilt the cell distribution to a team-level mean rather than
adding a feature.

### 3. Starter share is a league constant; the workload model already knows better

`blend_with_bullpen` uses `starter_share = 0.528` for every hitter facing
every starter. But `workload.expected_bf(pid)` is per pitcher, and the
split by lineup slot is exact arithmetic, not a model:

    starter PAs for slot s = #{ n ≤ BF : n ≡ s (mod 9) }

    BF = 22:  slots 1–4 face him 3 times, slots 5–9 twice
    BF = 27:  every slot 3 times
    BF = 18:  every slot twice

Marginalise over `bf_pmf(pid)` and divide by that slot's expected PA to
get the share. An ace at 26 BF gives the leadoff hitter ~0.66 exposure; a
five-inning starter at 19 BF gives the 9-hitter ~0.55. `pitcher_data.py`'s
own table shows the constant is worth ±5 points of HR probability at the
extremes. This is the cheapest structurally-new information in the
project: it is already computed, just not connected.

### 4. Team model has no home-field advantage and is ungraded

9/01 teams file: `home_win_prob` ranges 0.448–0.574, mean 0.503. MLB home
teams win ~53–54%. The only HFA in the model is the tie-share
(`home_extra_win`, clipped 0.45–0.58) plus `is_home` inside the hitter LR;
nothing adds the ~0.15 runs a home side is worth. Projected runs average
4.17 (home) and 4.26 (away) against a league ~4.4 — the doc noted the
sanity check lands low, and the reason is that summing nine starters
omits bench PAs (~5% of a team's plate appearances).

Two cheap fixes:

- Measure home win rate and home/away run differential from
  `team_lines.csv`, add it as a run adjustment (or a fixed logit shift).
- Benchmark and grade it. The Odds API's featured-markets endpoint
  (`/sports/baseball_mlb/odds?markets=h2h,totals`) costs **1 credit per
  market for the entire slate**, not per event. Moneylines and totals are
  ~2 credits a night versus ~30 for pitcher strikeouts. `score_slate`
  should grade win probability and total runs against `team_lines.csv`,
  which already holds the truth.

---

## Tier 2 — modelling changes worth a lab run

### 5. Shrinkage target and training window are five-year pooled

`add_batter_rolling_rates`: `prior_mean = df[target].mean()` over every
PA in the cache, 2022–2026. A 150-PA window in September 2026 is shrunk
(k = 64 for K, ~240 for HR, ~1000 for hits) toward a league rate that
includes 2022. The K model doc measured all-history at +6.8% versus the
last 12 months; hitters' league rates drift the same way. Separately, the
LR is fit on all years with equal weight and re-fit nightly.

    shrunk = (x + k · p_league) / (n + k)

    with p_league = trailing-365-day league rate as of that PA's date,
    not the full-cache mean.

The LR intercept absorbs a uniform shift; it cannot absorb the fact that
thin hitters are being pulled toward the wrong number relative to
established ones. Test in `model_lab` with rolling origin: 24-month
training window, and/or exponential recency weights on rows.

### 6. Feed logits, not probabilities

`train_rate_model` is a logistic regression whose inputs `bat_*`,
`pit_*_allowed`, `matchup_*` are on the probability scale. Per-PA HR rate
runs 1%–8%; the model has to bend a linear-in-p feature onto a logit link.

    use  logit(p) = ln(p / (1 − p))  as the feature, per column

Same model, one transform, testable in `model_lab` in an afternoon. The
"HR spread too wide" watch item in the clean-vs-tainted doc (top bucket
said 16.9%, got 12.2%; bottom said 6.9%, got 12.2%) is what a
probability-scale HR feature would produce at the tails.

### 7. Platoon is one coefficient

`platoon_edge = (stand != p_throws)` with a single weight. The
left-on-left penalty is materially larger than the right-on-right one,
for K and for power. Either two indicators (`lhb_vs_lhp`, `rhb_vs_rhp`) or shrunk
batter-side splits (the pitcher side already has two-level shrinkage;
the batter side has none). This is a change in *shape*, not another
level feature — it is information the current eight columns cannot
express regardless of coefficient.

### 8. One population contribution shape for every hitter

TB and H+R+RBI use `per_pa_contribution_distribution` over the whole pool,
then `scale_contribution_distribution(method="frequency")` to hit the
hitter's mean. A slugger and a slap hitter with the same expected TB/PA
get the same P(4 | v > 0). Pooled TB over 2.5 and 3.5 are the two weakest
counting lines (+0.012, +0.009 all rows). `bases_per_hit` is already
computed per hitter for the mean; use his shrunk single/double/HR mix to
build his own per-PA TB vector, with `p_is_hr` and `p_is_hit` from the
per-PA models as the anchors. For H+R+RBI the same applies through
`hrr_certain`.

### 9. The runs/RBI/bases paths use raw, unshrunk means

`rbi_rate = pa.groupby("batter")["rbi"].mean()`, `obp_series`, and the
RBI two-stage inputs `bat_is_hr_career` / `bat_is_hit_career` — raw
per-batter means (correction: `bases_per_hit` was already shrunk), and the last two
`fillna(0.0)`, i.e. a debut hitter gets a HR rate below any hitter
alive. The 9/04 slate has a hitter with 5 PA of history. Everything else
in the engine shrinks; these should use `estimate_prior_strength` +
`shrink` like the rest.

---

## Tier 3 — small correctness items

10. **Off-by-one on tonight's features.** `latest = pa.groupby("batter").tail(1)`
    takes the last PA row, whose rolling columns are `.shift(1)` — they
    exclude that PA. Every hitter's features tonight omit his most recent
    plate appearance. Same for `form_z`. One extra PA out of 150 is small;
    it is also free to fix (compute the un-shifted window for the last row).

11. **Pitcher rolling rates have no within-game order.**
    `add_pitcher_rolling_rates` sorts by `(pitcher, game_date)` only, so a
    training PA's `pit_*_allowed` can include later PAs from the same
    game. Small training-only leak; add `at_bat_number` to the sort.

12. **PA definition.** `is_k` misses `strikeout_double_play`. `events`
    non-null also includes non-PA rows (`caught_stealing_*`, `pickoff_*`,
    `game_advisory`) that get counted as plate appearances with no
    outcome. Whitelist PA-ending events instead of `notna()`.

13. **Park factors.** Static table, and `TEAM_ALIASES` maps `ATH → OAK →
    92` while the Athletics play in Sacramento (2025–27), a hitter's park.
    The HR factor is also the only park term in the hits, walk and K
    models. Savant publishes per-season, per-handedness, per-stat factors
    as one CSV; swapping the table is not a new feature, it is the
    existing feature done right.

14. **Thin-pitcher fallback.** `starts_seen = 0` gets the league mean BF
    (22.7). Debuts and spot starters face fewer, and the market knows it:
    8/31 Anthony Molina, 0 starts, model 0.86 vs market 0.56 on over 1.5.
    Measure `E[BF | 0–2 prior starts in window]` from the starts table and
    use it as that pitcher's prior; exclude `starts_seen < 5` from the
    edge list until then.

15. **Hitter market benchmark.** `compare_market` covers only
    `pitcher_strikeouts`. The batter markets exist in `KNOWN_MARKETS` and
    the credit budget is the constraint. One night a week of
    `batter_home_runs` (~15 credits) would still answer the question the
    docs keep calling the biggest epistemic gap.

---

## What not to do

Do not add another level feature. The five negative results are the
right prior and the two labs are the right gate. Items 3, 6, 7 and 8 above
change the *structure* of what the model can express (exposure by slot,
link scale, an interaction, a per-hitter shape); item 5 changes what it is
shrunk toward. None is a second measurement of "how good is he".

## Suggested order

1 (K path), 2 (PA table), 3 (starter share by slot) — each is a
correctness fix to something already deployed. Then 4 (grade the team
model, 2 credits a night). Then 5–7 through `model_lab` with rolling
origin. 8–9 when the counting-line tails become the limiting prop.
