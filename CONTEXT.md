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

## V12 (Sep 13-14, 2026) — the front page, a betting record, and two markers measured before being built

The theme of this version is **committing a number before it can be
graded, and measuring a feature before building it**. Three things that
looked like UI work turned out to be evidence problems, and two feature
requests were answered by a lab script rather than a feature.

### Bet slips, committed before first pitch

Nolan asked which of the previous night's slips hit. They could not be
answered: the Bet ready tab built slips at render time, so they existed
only as pixels, and re-deriving them the next morning would have graded
today's code against last night's games.

`slips.py` now writes `cache/slips_{date}.csv` before first pitch (step
6/6 of `run_slate`), and the dashboard READS that file instead of
rebuilding. Two consequences beyond gradeability: the slip on screen and
the slip in the log are the same object, and there is exactly ONE
definition of what a slip is — building it in two places is how the card
and the copy text drifted apart once already.

`score_slips.py` grades them, and is now **steps 6-10 of
`score_slate`**. Run back to back the two scorers refreshed the same ~270
players over the network TWICE; the actuals from step 3 are handed
straight over and step 7 is a no-op. Verified with a stub that raises if
the loader is called at all: 0 refreshes.

**The output is expected-vs-actual, never a win/loss record**, because a
record cannot be read here:

| tier | slips/night | expected hits | one hit every | nights to detect 20% miscalibration |
|---|---|---|---|---|
| SAFE | 5 | 1.74 | 0.6 nights | ~38 |
| MEDIUM | 5 | 0.33 | 3 nights | ~286 |
| MIXED | 5 | 0.18 | 5 nights | ~526 |
| LOTTO | 5 | 0.02 | **56 nights** | ~5,536 |

A LOTTO record of 0-for-200 is exactly what a 0.4% bet should look like.
SAFE is the only tier that will say anything this season. The ± is the
**Poisson-binomial** deviation, `sqrt(Σ p(1-p))` — these slips vary too
much in probability for `sqrt(n·p̄(1-p̄))`, which would overstate the bar
several times over.

**The one thing this data can measure that nothing else can:** slips are
logged with whether they reuse a game, so pooling correlated slips
separately turns the correlation warning into a number. Correlated legs
land together more often than the product implies, so the "all hit"
figure is too LOW by whatever that gap turns out to be.

Three bugs, all found by reading real output, none of which raised
anything:

1. **Nested props.** Avoiding only a repeated GAME produced `Yandy Diaz
   Hit 73.2% + Hits 1.5 33.1% + Hits 2.5 10.0%` — one bet wearing three
   labels, worth its smallest leg rather than the product. Each leg must
   now be a different PLAYER, the only rule that is safe without
   hard-coding which props imply which.
2. **The arm leg surfaced the fakest edges.** Ranking by |edge| put three
   `Over 3.5 K` lines on top (+35.6%, +26.1%, +17.9%) — exactly the
   compression bias. Demoting them was not enough; a suspect arm (an OVER
   at a line ≤ 4.5) is now EXCLUDED from slips entirely. A leg the model
   is measurably wrong about does not belong on a bet slip at any rank.
3. **The arm leg recommended the side the model dislikes.** It always
   wrote `Over L` with the signed edge, so `Glasnow Over 7.5 K, 29.7%,
   market 42%, -12.1%` appeared as a pick. A negative edge on the over is
   a positive edge on the under. Now `Under 7.5 K — 70.3%, +12.1%`.

### The compression: under-informed, not mis-calibrated

`lab_k_spread.py`, 202 pitcher-nights over 10 captured slates:

```
model_k = 2.11 + 0.67 × market_k    [0.61, 0.74]    r = 0.82
```

Stable night to night, 0.52 to 0.75. **But this is not a defect to fix by
stretching.** Binned against what actually happened, for pitchers with
10+ starts of history, the model sits on the diagonal in **5 of 5 bins**
for both strikeouts and batters faced, and `actual_k` on `model_k` gives
slope 1.11 [0.86, 1.36].

A correctly shrunk estimator IS narrower than the truth — that is what
shrinkage is for, and `sd(projection) = r × sd(true value)`. Against its
own information content the model is at 82% of optimal width; the market
is at 85%. **The gap to the market is information, not calibration**
(r 0.46 vs 0.55). Stretching the output would break the one thing that
currently holds.

### The relief tier, and an opener bug it uncovered

The largest error in the model was pitchers with no starting history:
projected 22.6 batters faced, actual 10.4, **−12.2 on 20 starts**. The
names showed two populations, not one — Derek Law 6, Tim Mayza 9, Taylor
Clarke 6 are *relievers* in bulk games, while Jedixson Paez 23, Andrew
Sears 23 are genuine debut STARTERS the existing prior already handles.

`WorkloadModel.fit` was **already computing `relief_bf`** and nothing
read it. Riley Cornelio was classified `role = "reliever"`, that
classification was written into the slate file, and he was projected for
22.65 batters anyway. Fixed with a third tier in `_opener_entry`: relief
work shrunk toward the OPENER population (`RELIEF_PRIOR_APPEARANCES = 12`).

**The test then caught a live bug nobody had noticed.** `bf_pmf` and
`expected_bf` disagreed by six batters for every opener, and had since
the opener tier landed:

```
OPENER     expected_bf   7.01   bf_pmf mean  13.01
RELIEVER   expected_bf   4.70   bf_pmf mean  13.01
```

The league PMF is fitted on non-opener starts so its support bottoms out
at 13, and exponential tilting cannot pull a distribution below its own
support. Since `k_count_distribution` reads `bf_pmf` and not
`expected_bf`, **every opener's strikeout probabilities were computed off
a 13-batter distribution** while the headline said 7. Fixed by fitting a
second PMF from opener starts plus relief appearances.

The lesson: neither number was wrong on its own. They were wrong
*relative to each other*, and no test compared them. `test_relief_tier.py`
asserts `bf_pmf mean == expected_bf` for all four tiers.

### Versus-opponent: real, and it does not repeat

`lab_opponent.py`, 50 starters, 4,599 starts, 111,313 PA. Asked for as a
feature; answered as a measurement.

The naive version is three claims stacked, and only one is new
information. Testing the RESIDUAL after both main effects, against a
parametric bootstrap null:

```
real effect       2.3% of K rate       z = +5.32, p < 0.001
positive control  3.0% injected   ->   3.0% recovered
```

So something IS there. Then the decisive test — does it repeat?

```
split-half r = +0.0185   null -0.0008 ± 0.0295   z = +0.66
```

**It does not.** The overdispersion is drift and within-start clustering,
not a property of the pairing. Stronger than a sample-size objection:
even with unlimited data there would be nothing to predict with. No
vs-opponent column was built.

> A first attempt used a label shuffle as its control and the control came
> back MORE dispersed than the real data (1.268 vs 1.150). Shuffling the
> opponent labels destroys the opponent main effect, which is real and
> large. The null has to preserve the fitted main effects and re-draw only
> the coin flips.

**What survives:** recent form, pitch counts, home/away — the lighter of a
pitcher's two halves still holds a median 46 starts and 1,069 PA. Shown
as a record, not a signal, in the new Recent form panel
(`pitcher_form.py` → `cache/pitcher_form_{date}.csv`, ~25 KB a night,
because the statcast caches are 1.2 GB and the deployed app can never
read them).

### Two hot/cold markers: one dead, one built on the other signal

**The hitter marker, after 13 slates: nothing.**

```
Hot    predicted 36.0%   actual 33.5%
Cold   predicted 33.9%   actual 32.1%
HOT minus COLD residual  -0.71 points ± 2.32   z = -0.31
```

Both groups undershot by about the same amount — the model's own drift,
not a form effect — and the sign is backwards. Not a kill at 490 hot
rows, but not a licence to build a second one on faith.

So the pitcher version was **tested on history before being built**
(`lab_pitcher_form.py`, 4,599 cached starts), because the live log would
take a decade at fifteen starters a night:

| feature | r | p | bottom vs top decile |
|---|---|---|---|
| recent K rate, z | +0.019 | 0.210 | +1.13% K rate |
| **fastball mph** | **+0.052** | **0.001** | **+2.19% K rate** |
| recent K rate (shuffled) | +0.007 | 0.656 | +0.45% |
| velocity (shuffled) | +0.001 | 0.963 | −0.49% |

**The obvious version fails and the physical one works** — a direct
measurement beats an outcome filtered through defence and luck, exactly
as `context-features-availability.md` predicted for exit velocity. The
effect is ~2.2 points of K rate between deciles, about **half a strikeout
over 23 batters**. Velocity does NOT predict batters faced (r = 0.015,
p = 0.32), so it leaves the outs prop alone.

Two of my own controls caught bugs in the test itself:

- Shuffling the feature **within a pitcher** preserves every
  between-pitcher difference, so the "null" returned z = +2.5 with the
  pairing already destroyed. Fixed by de-meaning both feature and
  residual within pitcher.
- De-meaning within (pitcher, MONTH) to remove the seasonal velocity arc
  returned r = −0.31, z = −14.2 — an effect bigger than anything real,
  manufactured by de-meaning inside ~5 observations whose rolling windows
  overlap the residuals. Removing the LEAGUE's month effect instead left
  the result unchanged, so the season was never the explanation.

Shipped as `velo_base` / `velo_recent` / `velo_drop` / `velo_state` on
`pitchers_{date}.csv`, cut at **±0.5 mph** (chosen from the data: 1.85
points of K rate at z = +3.8, fires on 40% of starts). Written by
`pitcher_form.py` and carried into `pitcher_row_log.csv` by
`_log_pitcher_rows`, which is the only place it sits beside the outcome.
Same contract as the hitter marker: shown, logged, **fed to nothing**.

### Per-pitcher strikeout lines, and two guards

`predict_slate` already wrote `k_dist` — the whole PMF as a
comma-separated string. Summing `k_dist[6:]` reproduces the stored
`prob_k_over_5.5` to seven decimals, so **any line, any pitcher, any
slate already on disk**, nothing re-run. The fixed 5.5/6.5 columns are
replaced by Line / Over / Mkt / Books / Edge, with the line taken from
the captured odds where one exists.

Two guards the data argued for:
- **Thin lines are suppressed.** Joe Ryan's only captured line on 9/13
  was 3.5 from THREE books, read as a +21-point edge. A 132-start starter
  does not get a 3.5 line unless those books know something. Edge is
  uncoloured below four books.
- **`Edge` uses fixed ±5-point thresholds, not quartiles.** A quartile
  always paints a top quarter, so on a night of broad agreement it would
  colour the largest trivial disagreement and call it a find.

### Odds credits: a re-pull was re-buying the whole slate

`fetch_slate_odds` only skipped games that had already STARTED. A game
already captured but not yet started was fetched again at full price —
the per-event endpoint has no idea it already sold you that game. Pulling
at 10am and again at 6pm cost 15 + 10 = **25 credits for one slate**, or
~20 nights of a 500-credit month instead of ~33.

Now **incremental by default** (matched on home/away team names, since
the output file has no `event_id`), with `--refresh` to re-buy at a later
line. Verified on the real 9/13 file: 14 captured, 15 offered, exactly 1
fetched. `cache/credit_log.csv` records every call and `run_slate` ends
with a CREDITS block whose "Plan says" figure is the API's own
`x-requests-remaining`.

### The front page, and the reload

The Slate page was rebuilt around what is true TONIGHT rather than a
ranked list of twenty hitters that looked identical every night: a
templated summary line, a POOL of finding cards that fire when their
condition is met, every game as a clickable row, and an in-place
drill-down carrying the team model's runs and scorers (**no winner** —
the win probability was measured at Brier 0.237-0.265 against 0.25 for a
coin flip and only ever spanned 45-57%).

Clicking a game used to reload the page, because `<a href="?game=">` is a
real browser navigation that Streamlit does not intercept. The clock is
now a button, so it reruns over the websocket; the rest of the row stays
HTML on a shared grid so the columns still line up. Prop selection is
chips rather than a dropdown, for the same reason.

Side effect worth keeping: `st.query_params["game"] = x` preserves other
keys, so the saved-hitter list survives on its own. The old link had to
re-append `&picks=` by hand or opening a game silently emptied the
sidebar.

### Numbered steps, and error bars on the level check

`run_slate` (6), `score_slate` (12) and `score_slips` (5) all print
numbered banners now. The argument is strongest for `score_slate` step 3:
it refreshes ~270 players over the network, takes minutes, and is the
step most likely to fail because Statcast posts hours behind the last
out. It used to be one undifferentiated wall.

The Results level check now carries a standard error and a verdict. The
−0.4 strikeout bias was NOT corrected for: t = −2.36 across thirteen
September dates, with date means swinging −1.11 to +1.67 and their spread
barely above sampling noise. September is when outings shorten anyway, so
an offset fitted there would partly be fitting a seasonal effect. The ±
is measured **across slates**, not across starts — fifteen starters on
one night share a league-wide pattern of bullpen use and are not fifteen
independent draws.

---

## V11 (Sep 6-12, 2026) — three bugs found by looking at outputs, and a new prop

Nothing in this version came from planning a feature. Every item is
something that turned up while reading numbers that were already on the
screen, which is the argument for putting numbers on the screen.

### The flag lab, run properly — two verdicts flipped

`fixes-2026-09-05.md` recorded the flag results from ONE held-out period.
Re-run as `flag_lab.py --origins 3` (246k PAs; cuts 2024-08-29 /
2025-05-26 / 2025-08-27), two of five changed:

| Flag | One origin said | Three origins say |
|---|---|---|
| PER_HITTER_SHAPE | worth turning on | confirmed — **ON since 9/06** |
| PITCHER_RATE_WINDOW_DAYS | "is_k BETTER +0.0007" | noise; **WORSE on walks at 2 of 3** |
| LOGIT_FEATURES | "negligible" | BETTER on is_k ×3 — on a dead target |
| LEAGUE_PRIOR_TRAILING_DAYS | negligible | negligible 12/12 |
| PLATOON_SPLIT | negligible | negligible 12/12 |

PER_HITTER_SHAPE, all four total-bases lines (the 2.5 row was missing
from the 9/05 write-up entirely):

| Line | 2024-08-29 | 2025-05-26 | 2025-08-27 |
|---|---|---|---|
| 0.5 | +0.0066 | +0.0065 | +0.0069 |
| 1.5 | +0.0018 | +0.0019 | +0.0021 |
| 2.5 | −0.0001 | −0.0000 | −0.0002 |
| 3.5 | +0.0015 | +0.0017 | +0.0015 |

0.5 and 1.5 exclude zero at every origin and replicate almost exactly;
2.5 is a true no-op (|gain| <= 0.0002 against a 0.0005 floor), not a cost.
**Scoring-log rows before 2026-09-06 were produced with the flag OFF**, so
pooled total-bases numbers mix two models across that date.

LOGIT_FEATURES is the instructive one. It is BETTER on `is_k` at all
three origins with calibration improving (dECE −0.0022 to −0.0028) — and
since the 9/05 K rewrite `is_k` no longer reaches any graded prop. The
pitcher strikeout prop compounds `per_batter_k_probs` (measured rates, no
regression), and `score_slate` grades no hitter strikeout prop. A flag
that is genuinely better on a target nothing reads is worth nothing.

Two Tier-2 predictions from the review are now measured and dead: the
trailing league prior (the era-drift argument, which DID matter on the
pitcher side) is negligible 12/12 for hitters, and split platoon
coefficients are negligible 12/12.

`flag_lab.py` now appends every row of every trial to
`cache/flag_lab_results.csv`. It previously printed and wrote nothing,
which is why the missing 2.5 line could not be looked up, only
re-measured. **A lab whose output exists only in a terminal has not
finished.**

### Bug 1 — in-play prices were being graded as closing lines

The worst of the three, because it corrupted the one benchmark that
answers "is this good" rather than "is this better than nothing".

The Odds API keeps returning a game after first pitch with LIVE odds
attached. Nothing filtered them. On 2026-09-06, **10 of 14** game-line
rows were priced in-play, including a 0.971 and a 0.086 — teams four runs
up and four runs down, priced as though somebody had forecast it. That
handed the "market" a Brier of **0.1368** against the model's 0.2401.

No book prices baseball moneylines at 0.137. The model was being graded
against hindsight.

Recomputed on pre-game prices only, 2026-09-05 **reverses**:

| 9/05 | n | Market Brier | Model Brier |
|---|---|---|---|
| As logged (2 in-play) | 15 | 0.2495 | 0.2540 |
| Pre-game only | 13 | **0.2811** | **0.2509** |

Three fixes, at three layers:
- `data/odds_lines.fetch_slate_odds` skips started games **before**
  spending a credit. Player props are charged per event, so a started
  game costs a credit and returns a price that cannot be used. Same
  exclusion, credit kept.
- `fetch_game_lines` cannot skip per event (one call, whole slate) so it
  discards started games from the response instead.
- `compare_market` and `score_teams` both drop rows whose price was taken
  after their own first pitch, which cleans files captured before the fix.

### Bug 2 — captures overwrote each other

Discovered immediately after Bug 1, and *caused* by its fix. Once started
games are skipped, a late run returns only the games still to come — and
`to_csv` replaced the whole file with those few. On 9/06 a 31-line
capture became 7. **Odds are the one input that cannot be re-fetched at
any price**; predictions are protected after the fact by
`preserve_committed_rows` and odds have no equivalent.

Both odds files and `market_compare_*` now MERGE: a game absent from the
new capture keeps its earlier row, a game in both takes the newer price
(a later pre-game price is a better closing line). Each row carries its
own `fetched_at_utc`, so one file legitimately holds several capture
times and readers can tell which price was taken when.

### Bug 3 — the workload model was fitted before its own data arrived

`predict_slate` fitted `WorkloadModel` from `load_starter_history`, which
is **cache-only by design**, and then called `build_pitcher_rates` — the
call that FETCHES a starter's Statcast and creates that cache — a hundred
lines later.

So any pitcher appearing on a slate for the first time had no cache file
at fit time: zero starts, league-mean priors, and his full history
written a minute too late to be read. On 2026-09-07 that was **Joe Ryan
(132 career starts), Nick Pivetta, Brayan Bello, Jonah Tong and Derek
Law**, all reported as debutants. Cache file mtimes 15:10:03-15:11:13
against a prediction stamped 15:12:23.

It looked like debuts rather than a defect because the same pitcher never
showed zero twice — his cache existed by the next slate. Fixed by moving
the fit below the fetch; nothing between them used it (checked by AST
name reference, not by text search).

This had been true since the K model went live, and September is call-up
season, so it was getting worse.

### The per-pitcher row log, and what it found in one query

`pitcher_scoring_log.csv` keeps two rows per slate. It can answer "how is
the K model doing" and nothing else — no pitcher in it, so every other
question needed the whole join rebuilt from prediction files and the
boxscore cache.

`score_slate` now also writes `cache/pitcher_row_log.csv`: one row per
graded starter with `starts_seen`, `career_starts`, `days_since_last_start`,
projections, actuals and residuals. Backfilled to 192 starters across 8
slates. It answered the standing question on the first query:

| Starts in window | n | Proj BF | Actual BF | BF error | K error |
|---|---|---|---|---|---|
| **0** | 11 | 22.6 | **12.8** | **+9.8** | +1.80 |
| 1-4 | 15 | 21.6 | 17.2 | +4.4 | +0.92 |
| 5-10 | 14 | 21.9 | 21.8 | +0.1 | +1.00 |
| 11-19 | 42 | 22.0 | 22.2 | −0.2 | +0.26 |
| 20+ | 110 | 23.0 | 22.6 | +0.3 | +0.25 |

Under 20 starts the model runs **+0.71 K high (t = +3.07)**; over 20,
+0.25 and not significant.

But the error is concentrated almost entirely at zero, and it is **not
the strikeout rate — it is batters faced**. Projected 22.6, actual 12.8.
Trevor Williams faced 3. Derek Law faced 6. These are openers and bullpen
games priced as if they will go six innings. Excluding the one row
mislabelled by Bug 3, the genuinely-thin ten still face 13.5 against 22.6
projected. **The zero-start prior is roughly nine batters — three innings
— too generous.**

### Two kinds of zero

`starts_seen == 0` means a debut OR a veteran back from a long layoff,
and they want different priors. `WorkloadModel` now keeps
`career_starts` and `last_start_dates`; `predict_slate` exports both.

    Burnes   window 0   career 107   464 days out  ->  returning
    Ryan     window 25  career 130    35 days out  ->  established
    Tong     window 2   career   4   349 days out  ->  debut/thin

Days-out alone does not separate them — Tong has been away nearly as long
as Burnes. It takes both numbers. Not a model input: a marker, the same
arrangement as the form marker.

Burnes on 9/08 is the case to watch: the model said 23.0 batters faced and
5.0 K where the market had him at 59.8% over 2.5 K against the model's
87.7%. A pitcher back from fifteen months is on a leash no prior measured
from debutants knows about.

### NEW: the outs prop

Outs recorded — the market's "pitcher outs", and arguably more fundamental
than strikeouts, since a manager pulls a starter by innings and pitch
count and batters faced is the *consequence*.

**Deriving outs was the whole problem, and the obvious method is wrong.**
Mapping each event to the outs it records and summing gives, measured over
one pitcher's 2,000 plate appearances, 0.018 outs for `single` and 0.989
for `strikeout` — because runners are thrown out BETWEEN plate appearances
and the batter's event cannot see it. A few percent per PA is half an out
per start.

A completed half-inning has no such problem: it is worth exactly
`3 - outs when he entered`, whatever happened on the bases. And a starter
who appears in inning n+1 completed inning n. So every half-inning but his
last is exact, and the event map is used for at most one plate appearance
per start. Checked against published season totals:

| Year | Derived | Published |
|---|---|---|
| 2025 | 17.55 outs (5.85 IP) | 5.85 IP |
| 2024 | 18.22 outs (6.07 IP) | 6.07 IP |
| 2022 | 18.33 outs (6.11 IP) | 6.12 IP |

Fitted exactly like batters faced — same window, same shrinkage, league
shape exponentially tilted onto the pitcher's mean — so `outs_pmf` answers
any line, and `outs_dist` is stored alongside `k_dist`. Lines published:
14.5 through 18.5.

**Kept separate from batters faced rather than derived from it.** Causally
outs come first, but rebuilding BF on outs would disturb the validated K
prop. Coherence is checked instead: `implied_baserunners` = projected BF
minus projected outs. Measured on 280 real starts it is **6.85**, against
6.53 actual baserunners — the 0.32 gap is double plays and runners thrown
out, where one batter yields two outs. Expect 6.0 to 7.5.

Graded against real innings pitched from the boxscore line already pulled
for batters faced, so it costs no extra fetch. **Model only** — the outs
market is charged per event and would have taken the odds bill from 17 to
32 credits a night, which does not fit a 500-credit month.

### Dashboard

- **Pitchers table is a sortable dataframe**, like All hitters. Two things
  that were crammed into cells became columns, because a column can be
  sorted and a suffix cannot: **IP** (was "(5.3 ip)" inside the Outs cell)
  and **Rest** (days since last start — better than the BACK chip it
  replaced, because normal rest is 4-6 days so a 464 sorts to the top and
  a 6-day and a 40-day layoff stop being the same badge).
- Amber thresholds: **<= 10 starts** on the Pitchers tab (the K prior is
  250 batters faced, about ten starts before a pitcher's own rate leads),
  **< 20** on the market table, where the bar is higher because a
  disagreement with five books means nothing if the model half-knows him.
- Results tab: a **Ranking (AUC) column** and a collapsible "How to read
  this" built from the live log. Discrimination was invisible on that page
  — a model that quotes the league rate to everyone matches every total
  and has an AUC of 0.50, and calibration alone cannot tell those apart.
- **Game Lines tab retired** (`SHOW_GAME_LINES = False`, one line to
  restore). Four graded slates: home-win Brier 0.237-0.265 against 0.25
  for a coin flip, win probabilities never leaving 0.45-0.57, totals about
  half a run worse than the market. The team model keeps running and keeps
  being graded into `team_scoring_log.csv`; it just no longer costs a tab.
- **"My team"** — up to 8 saved hitters in the sidebar, kept in the URL
  (the deployed app is one process, so a file would be a single global
  list every viewer overwrote).

### Odds budget

Player props are per event (~15/night), game lines per market for the
whole slate (2). Dropping game lines takes the nightly bill from 17 to
**15**, and coverage from 29 nights per 500 credits to **33**.

Everything else in the pipeline is free: `check_lineups` is one schedule
call, `predict_slate` is Statcast and models, `compare_market` reads local
files. So re-running predictions late to pick up confirmed lineups costs
nothing and should be done.

`run_slate.py` runs the four steps in dependency order and refuses to buy
odds if predictions failed. It exists because the manual order had
`compare_market` (which READS `pitchers_{date}.csv`) running before
`predict_slate` (which WRITES it) — producing no output, no log row, and
no error.

### Two bug shapes worth remembering

**`int(x or 0)` is not NaN-safe.** NaN is truthy, so `float("nan") or 0`
evaluates to NaN and `int()` raises. The guard reads like it handles
missing values and handles only `None` and zero. Now `_int_or()`.

**Every new column arrives half-populated on the first re-run.**
`preserve_committed_rows` deliberately mixes rows written by different
versions of the code, so a column added today is present on refreshed rows
and absent on preserved ones. Present-but-NaN is a different case from
absent, and it is the one that breaks. Both the outs columns and
`career_starts` shipped with that case tested; `career_starts` shipped
without it and took the dashboard down.

---

## V10 (Sep 5, 2026) — a review, fourteen fixes, and one self-inflicted wound

A full read of the code against the docs (`model-review-2026-09-05.md`,
also in the project). Everything below was rated by severity and fixed in
that order. Every correctness fix is live; every *modelling* change is
behind a flag in `model_flags.py`, default off, with `flag_lab.py` to
decide.

### Severity 1 — deployed code that was not what was validated

**The K prop compounded an untested rate.** `build_pitcher_props` walked
the lineup with the hitter LR's `p_is_k_vs_starter`, whose pitcher input
was the starter's career rate since 2022 with no window. The 12-month
shrunk `WorkloadModel.k_rate` the K doc validated was computed and only
written to the CSV. On the 9/01 and 9/04 slates the two agreed at
Spearman 0.65 and the deployed one had 1.7x the spread; live skill on 32
starts was **-0.14** against a backtest of +0.06.

Now: `per_batter_k_probs` = the hitter's shrunk 600-PA strikeout odds
ratio (against a 12-month hitter-side league) x the pitcher's 12-month
shrunk odds x a measured platoon factor. `backtest_k_props.py` runs THAT
path, rolling origin, one month at a time, on 2,834 starts Apr-Aug 2026:

| line | said | did | Brier skill |
|---|---|---|---|
| over 4.5 | 0.562 | 0.556 | +0.074 |
| **over 5.5** | 0.405 | 0.393 | **+0.094** |
| **over 6.5** | 0.266 | 0.256 | **+0.097** |
| over 7.5 | 0.157 | 0.152 | +0.076 |

Every month positive (+0.057 to +0.135 at 5.5). The lineup walk beats a
flat per-pitcher rate by ~+0.01 in every month. The old backtest script
was never committed; this one is, and score_slate now splits the pitcher
running total at 2026-09-05 so the two paths are never pooled.

**Thin pitchers.** A pitcher with no history got the league mean, 22.7
batters faced. Measured on career-thin starts (<= 2 prior starts in the
cache) the number is ~21.0 and the K rate is lower too. Both priors are
now the new-pitcher values for new pitchers and league for veterans --
one version pulled veterans toward the call-up mean and cost 0.3 BF
across the established group, which the backtest caught. Thin pitchers
went from -0.04 to about -0.00 Brier skill; `compare_market` no longer
ranks them in the edge list (`MIN_STARTS_FOR_EDGE = 5`).

**PA projection had train/serve skew.** `batter_pa_mean_20` was a 20-game
rolling mean at fit time and the all-time mean at serving time;
`team_pa_mean_20` was grouped by the VENUE's team at fit time and set
equal to the batter's mean at serving time. Replaced by
`EmpiricalPAModel`: P(PA = k | slot, home/away) counted from starters'
games since 2024 -- eighteen cells, thousands of games each, nothing to
skew. Two honest notes: (1) part of V8c's "leadoff over-projected" was a
pool-versus-all-players comparison -- pool hitters in slot 1 really do
average 4.5 PA against 4.3 for all starters, because regulars are not
lifted late; (2) the spread still came down, 1.14 -> 1.03, matching the
pool's own history.

### Severity 2 — deployed and biased

**Starter share per pitcher and per slot.** `blend_with_bullpen` used
0.528 for everyone. The starter faces batters 1..BF and slot s gets
#{n <= BF : n = s mod 9} of them -- exact arithmetic, marginalised over
the workload model's BF distribution, divided by the slot's expected PA.
An ace at 26 BF gives the leadoff hitter ~0.66; a five-inning starter
gives the 9-hitter ~0.55. Exported as `starter_share` on the slate.

**Team model.** Bench share: the lineup sum omits pinch hitters and subs,
measured at **7.9% of runs** on 1,886 games with both lineups complete
(the first cut used incomplete-lineup games and said 17%). Lineup sums
are divided by (1 - share). Home-field: the run gap (+0.085 home) is
measured and printed but NOT applied -- `is_home` is already in every
per-PA model and in the PA table, so applying it again double-counts.
`score_slate.score_teams` grades win probability and totals against
final scores, logs `cache/team_scoring_log.csv`, reports the home-side
residual, and compares to the closing moneyline/total when
`python -m data.odds_lines DATE gamelines` was run (2 credits for the
whole slate via the featured-markets endpoint).

**Park factors.** `ATH -> OAK -> 92` sent the Athletics' home games to the
Coliseum two seasons after they left it. ATH now has its own entry (108,
Sutter Health Park -- check against Savant's current figure).

### Severity 3 — small correctness

- **Off-by-one on tonight's features.** Rolling columns are `.shift(1)`,
  and features were read off each hitter's last real row, so every
  hitter's most recent plate appearance was excluded every night.
  `add_tonight_rows` appends one outcome-less row per slate hitter before
  the rolling features run and reads tonight from that.
- **Pitcher rolling rates** had no within-game order (could see later PAs
  from the same game in training). Sorted by at_bat_number.
- **PA definition.** `events.notna()` counted `truncated_pa` and
  baserunning events as plate appearances; `is_k` missed
  `strikeout_double_play`. Whitelisted. Verified against boxscores:
  934 of the 940 player-games where Statcast ran one PA over the boxscore
  were `truncated_pa`; the tables now match to 0.006 PA per game.
- **Unshrunk inputs.** Per-batter OBP, RBI rate, and the RBI model's
  career HR/hit rates (`fillna(0)` for a debut hitter) were raw means
  next to an engine that shrinks everything. Now shrunk with the same
  estimator (`estimate_prior_strength_counts` for RBI).

### Modelling flags -- built, measured, default OFF (`model_flags.py`)

`flag_lab.py`, held-out from 2025-09-20, 581k plate appearances:

| flag | result |
|---|---|
| LOGIT_FEATURES | +0.0002 to +0.0003 skill on HR/K/walk, CIs exclude 0, ECE down -- real and negligible |
| PLATOON_SPLIT | negligible |
| LEAGUE_PRIOR_TRAILING_DAYS=365 | negligible (the LR intercept absorbs it) |
| PITCHER_RATE_WINDOW_DAYS=365 | is_k **BETTER** +0.0007 [+0.0002, +0.0011]; walks -0.0003 |
| **PER_HITTER_SHAPE** | total bases, game level, PA fixed: **+0.0069** at 0.5, +0.0022 at 1.5, +0.0017 at 3.5, all CIs exclude 0 |

PER_HITTER_SHAPE is the one that clearly earns its place. Turning it on
changes the TB lines and therefore the running total; do it deliberately.

### The wound

`test_slate_dryrun.py` wrote its fabricated lineup slots to the LIVE
`cache/lineup_slots.csv` -- it always had -- and running it during this
session destroyed the real file (222k rows fetched from boxscores over
weeks). Recovery: `rebuild_lineup_slots.py` reconstructs lineups exactly
from the Statcast caches (a side's first nine distinct batters in
at_bat_number order, written only when every PA up to the ninth is
present) -- validated 100.00% against the 44,840 real rows in
`_pa_export.csv`, and covering 3,349 games with both lineups (1,428 of
2026). The rest refill through the normal boxscore fetcher on the next
run. The test now writes to `cache/_dryrun_lineup_slots.csv`, patches
`cache_path` so predict_slate reads that too, uses the All-Star break
Monday as its date, and cleans up. Same lesson as V8d: a test that can
touch the record it exists to protect will eventually do so.

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

## Current State (as of Sep 14, 2026)

The live system is the nightly loop below: `run_slate.py` before the games,
`score_slate.py` after, and a six-tab Streamlit dashboard. What it is
actually good at, measured rather than claimed:

- **The pitcher strikeout model is the strongest component.** It sits on
  the diagonal in 5 of 5 bins against real outcomes for pitchers with 10+
  starts of history, and its projections are at 82% of their own optimal
  width. Its gap to the market (r 0.46 vs 0.55) is information, not
  calibration.
- **The hitter props are calibrated but barely separate players** — 1.2x
  between the best bat on the board and the median on the common props.
  The low-base-rate props (HR 3.7x, TB 3.5 2.8x) are where the model
  actually distinguishes anyone, which is what the Bet ready tab's LOTTO
  tier is built on.
- **The team win model is off** — Brier 0.237-0.265 against 0.25 for a
  coin flip, spanning only 45-57%. Its runs and expected scorers are
  still shown; the winner is not.
- **Four things are shown and fed to nothing on purpose**, each waiting
  on its own Results row: the hitter hot/cold marker (13 slates, nothing),
  the pitcher velocity marker (measured prior, no live slates), home/away
  splits, and the slip record.

Everything below this line about V1-V4 AUC figures is the ORIGINAL
research phase and is kept for the negative results it records. It
describes a different, single-player codebase and should not be read as
the current system.

### The original research phase (as of Jul 31, 2026)
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

### The nightly loop (V12) — two commands

```powershell
# BEFORE the games. Six steps in dependency order; only step 4 costs
# anything. Odds pulls are INCREMENTAL -- a game already in
# odds_{date}.csv is skipped -- so a second run tonight buys only what is
# new. Ends with a CREDITS block.
python run_slate.py                    # today
python run_slate.py 2026-09-14
python run_slate.py --no-odds          # free: lineups, predictions,
                                       # recent form and slips only
python run_slate.py --dry-run          # print the plan, spend nothing

#   1/6  Lineups          check_lineups.py
#   2/6  Predictions      predict_slate.py   -> slate_, pitchers_, teams_
#   3/6  Recent form      pitcher_form.py    -> pitcher_form_, velo marker
#   4/6  Strikeout props  data.odds_lines    -> odds_            [COSTS]
#   5/6  Model vs market  compare_market.py  -> market_compare_
#   6/6  Commit slips     slips.py           -> slips_

# To deliberately re-buy a later, stronger line on games already priced:
python -m data.odds_lines 2026-09-14 --refresh

# The odds step retries connection failures and 5xx (3 tries, <10s) but
# NEVER retries a timeout -- the request may already have been billed.
# If it gives up, just run it again: captured games are skipped.

# AFTER the games. Twelve steps; slips are graded as 6-10, reusing the
# Statcast refresh step 3 already did rather than making a second one.
python score_slate.py 2026-09-14

# Standalone re-grade, when the hitter pass does not need re-running:
python score_slips.py 2026-09-14 --dry-run
```

### The labs — measure before building

```powershell
python lab_opponent.py        # sample | effect | persist
python lab_k_spread.py        # market | spread | calib
python lab_pitcher_form.py    # velocity vs recent-K-rate form
python flag_lab.py --origins 3
```

### Tests

Nothing here needs the network or a key; all of it runs off the cache or
synthetic data.

```powershell
python test_workload.py       # 32 assertions on the workload model
python test_relief_tier.py    # the four expected_bf tiers, and that
                              # bf_pmf agrees with expected_bf
python test_slips.py          # slip building and slip grading
python test_velocity.py       # 16 checks: a NaN velo_z must not blank a
                              # projection, and the clamp must bind
python test_hitter_log.py     # 14 checks, the important one being that
                              # re-scoring REPLACES a date, never appends
python test_odds_retry.py     # which HTTP failures may be retried.
                              # Asserts on urlopen CALL COUNT, because the
                              # call count is the credit count
python check.py               # renders every dashboard tab path, 6 cases,
                              # plus the clock / .gitignore / nested-
                              # expander guards
```

### Older single-purpose scripts

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt   # NOT bare pip -- the venv was
                                            # moved and pip.exe has the old
                                            # path compiled in

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
├── run_slate.py                     # the nightly pipeline, 6 numbered steps,
│                                    # ends with the CREDITS ledger
├── predict_slate.py                 # every prop for a slate, written BEFORE
│                                    # first pitch (preserve_committed_rows)
├── pitcher_form.py                  # recent form + the VELOCITY marker.
│                                    # Runs here rather than in the dashboard
│                                    # because it reads the 1.2 GB statcast
│                                    # caches, which are untracked forever
├── slips.py                         # what a bet slip IS -- one definition,
│                                    # imported by dashboard.py, written to
│                                    # slips_{date}.csv before first pitch
├── score_slate.py                   # 12 numbered steps; slips are 6-10
├── score_slips.py                   # whole-slip grading, expected vs actual
├── compare_market.py                # model against captured closing lines
├── dashboard.py                     # Streamlit: Slate / Games / Pitchers /
│                                    # All hitters / Bet ready / Results
├── lab_opponent.py                  # does a vs-opponent effect PERSIST? (no)
├── lab_k_spread.py                  # is the model compressed? (no -- under-
│                                    # informed, which is a different fix)
├── lab_pitcher_form.py              # velocity vs recent-K form (velocity wins)
├── flag_lab.py                      # feature flags over multiple origins
├── test_workload.py                 # \
├── test_relief_tier.py              #  > the tests that actually catch things
├── test_slips.py                    # /
├── debug_schedule.py                # diagnostic used to find the team dropna() bug
├── debug_alvarez.py                 # diagnostic used to find the accented-name bug
├── requirements.txt
├── README.md                        # how to run it -- start here
└── CONTEXT.md                       # this file: why every decision is
                                     # what it is, and what was measured
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

### What is TRACKED in git, and the rule behind it (V7-V12)

`.gitignore` ignores `cache/*` and then negates the exceptions. The rule
it keeps re-learning, stated at the top of that file:

> **Anything the dashboard READS, or any record that ACCUMULATES, has to
> be tracked. Everything else in `cache/` is regenerable; these are not.**

It has bitten more than once — the Results tab said "No slates scored
yet" for a day while `scoring_log.csv` sat populated on disk.

| file | why tracked | size |
|---|---|---|
| `slate_*.csv` | the predictions, committed pre-game | ~130 KB/night |
| `pitchers_*.csv` | Pitchers tab, and the velocity marker | ~25 KB |
| `pitcher_form_*.csv` | Recent form panel — the statcast caches behind it are 1.2 GB and never tracked | ~25 KB |
| `slips_*.csv` | the slips, committed pre-game | ~25 KB |
| `slip_log.csv` | accumulates, one row per slip per night | small |
| `odds_*.csv`, `market_log.csv` | **the least regenerable files here** — historical odds are not on the free tier, so a night not captured before first pitch can never be benchmarked by anyone, ever | few KB |
| `credit_log.csv` | the CREDITS block; accumulates | few hundred B |
| `market_compare_*.csv` | Pitchers tab market block | few KB |
| `scoring_log.csv`, `pitcher_scoring_log.csv`, `form_log.csv`, `team_scoring_log.csv` | the running records | few hundred KB/season |
| `teams_*.csv`, `team_lines.csv` | team model input and record | ~1 MB total |

**Never tracked:** `statcast_player_*.csv` (~1.2 GB), `statcast_pitcher_*.csv`,
`batting_lines.csv`, `lineup_slots.csv`, `game_context.csv`,
`pitchmix_*`/`pitchcontrol_*` (~1,900 files). Losing these costs time, not
information — but `.git` was 1.1 GB once, which is why they stay out.

`dashboard.py` imports numpy, pandas, streamlit, altair and **one** project
module — `slips.py`, which defines what a slip is and reads no cache of its
own. That header comment in `.gitignore` used to claim it imported no
project modules and read only `slate_*.csv`; both had quietly stopped being
true, and it was the justification for the whole file.

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
- **"HR is a coin flip" is FALSE and was superseded — corrected 2026-09-15.**
  The bullet below is a V2 TRAINING result (`train_v3.py`, `train_bridge.py`,
  AUC 0.51–0.54) and it does not describe the deployed model. Graded over 16
  live slates and 3,538 rows, `prob_hr` has the **highest AUC of any hitter
  prop**: 0.625 mean per slate, sd 0.067, z = +7.5 against no skill, above
  0.500 on 15 of 16 slates. Calibration is near-exact — said 11.52%,
  happened 11.50%. The per-PA rate engine (V5) and the pitch-mix style-fit
  feature replaced what the V2 experiment measured. Keep the bullet below
  for the lesson about feature SHAPE; do not read it as a live fact about HR.
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

- **A lab is only as big as the files sitting next to it, and nothing
  warns you.** On 2026-09-14 every lab in this project ran against
  `cache/statcast_pitcher_*.csv` on whichever machine invoked it. The
  working copy held 50 caches; this machine had 196. No error, no warning
  — the measurements simply used a quarter of the evidence. `lab_log5.py`
  is the demonstration: a −0.72% effect at z = −1.27 on 50 caches, and
  +0.03% at z = −0.55 on all 196. The lean was noise, and at the larger
  sample the smaller result would have shown at z = −3.2 had it been real.
  The fix is `distill_caches.py`: the labs consume about 4 MB of the
  1.25 GB, so that 4 MB is now written to `cache/distilled/` and every lab
  reads it and PRINTS the row count. A silent loader is how this stayed
  invisible. Each lab still applies its own `MIN_BF` after loading — the
  distiller keeps everything down to 8 batters so one table can serve
  several thresholds, and skipping that step silently moved
  `lab_shrinkage` from 4,198 starts to 4,408.
- **The opposing lineup is not where the error is, and that is now
  measured rather than assumed.** Four distinct questions, often confused:
  (1) does the model know who is batting — yes, `per_batter_k_probs`
  walks the real lineup in order with handedness; (2) is there signal in
  opponent IDENTITY beyond its hitters — no, `lab_opponent.py`, does not
  persist; (3) times through the order — real in baseball (−3.46 points,
  z = −8.84 over 23k matchups) but already absorbed, because `k_rate` is
  total K over total BF across all three looks, so a correct look factor
  must preserve the mean and then moves a printed probability by 0.2
  points; (4) does log5 COMBINE them correctly — yes, +0.03% at z = −0.55
  on 288k plate appearances, bounding any effect under ~0.05 K a start.

---

## Known Issues / Technical Debt
- ~~**The zero-start batters-faced prior is about three innings too
  generous**~~ (V11) — **FIXED in V12.** It was two populations, not a
  bad prior: relievers in bulk games (Derek Law 6 batters, Tim Mayza 9)
  averaged in with genuine debut starters (Paez 23, Sears 23). A third
  tier in `_opener_entry` projects relief-only arms from their own relief
  workload shrunk toward the opener population. `relief_bf` had been
  computed and unread the whole time.
- **The model's strikeout projections are ~82% of their optimal width**
  (V12), and that is NOT a calibration bug — a correctly shrunk estimator
  is narrower than the truth. The 0.67 slope against the market is an
  INFORMATION gap (r 0.46 vs 0.55), so the fix is features the model does
  not have — announced pitch limits, rest and injury news, umpire,
  catcher — and emphatically not stretching the output.
- **A uniform −0.4 strikeout bias** on pitchers with 10+ starts of
  history (V12). t = −2.36 over thirteen September dates, date means
  swinging −1.11 to +1.67. Deliberately NOT corrected for: September is
  when outings shorten, so an offset fitted there would partly be fitting
  a season. The Results level check now carries the error bar that will
  eventually settle it.
- **Two markers are shown and fed to nothing, by design** (V12). The
  hitter hot/cold marker has 13 slates and says nothing (hot minus cold
  −0.71 points, z = −0.31). The pitcher VELOCITY marker arrived with a
  measured reason to exist (4,599 cached starts, r = 0.052, p = 0.001,
  ~2.2 points of K rate between deciles) but has zero live slates. Neither
  enters the model until its own row in Results says so.
- **A pitcher returning from a long layoff gets the debutant prior.**
  `career_starts` and `days_since_last_start` are now exported and marked
  but not used by the model. A rehab return is on a shorter leash than a
  call-up and probably wants its own prior — needs more than the one
  Burnes case to fit.
- **The outs prop has no market benchmark.** Model-only by choice (the
  outs market is charged per event, ~15 credits a night, which does not
  fit the 500-credit budget alongside strikeouts). It is graded against
  real innings pitched, so "better than nothing" is answerable and "better
  than the book" is not.
- **The hitter strikeout prop (`prob_k`) is written and never graded.**
  `score_slate` grades no binary K prop, so the column on the dashboard
  has never been checked. Small fix — the truth column is already in the
  PA table, it needs adding to `BINARY_PROPS`.
- **Total bases pools two models across 2026-09-06**, the date
  `PER_HITTER_SHAPE` was turned on. Read a step change there as the flag,
  not as noise.
- **No hitter-prop odds exist anywhere in this project** (V12), so every
  hitter number is model-only and "market disagreement" is not computable
  for a bat at any threshold. Player props are charged per event and the
  free tier is 500 credits a month, so the budget goes to strikeouts. The
  Bet ready tab uses LIFT over the slate median as the honest stand-in and
  says so; it is not the same claim.
- **`velo_state` is "Unknown" for any starter with no statcast cache**
  (V12) — a debut, or a pitcher predict_slate named before his pull ran.
  Absent rather than wrong, but it means the marker's live sample grows
  slower than fifteen a night.
- **The team model is graded but unread** (V11). `teams_*.csv` and
  `team_scoring_log.csv` keep being written; the tab is off. It has no
  home-field term — the measured home-minus-away residual across four
  slates averaged NEGATIVE, so adding one on theory would have been wrong.
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

### V12 ✅ (Sep 13-14, 2026) — commit it before you grade it
  Slips committed before first pitch and graded after; the relief tier
  and the opener PMF bug; per-pitcher strikeout lines from the stored
  `k_dist`; incremental odds pulls and a credit ledger; numbered steps in
  all three runners; the Slate page rebuilt; the velocity form marker.
  Two feature requests answered with a lab script instead of a feature
  (vs-opponent does not persist; recent-K-rate form is noise).

### V12.1 ✅ (Sep 14, 2026) — five hypotheses died, one warning shipped
  A measured over-confidence in the strikeout probabilities: on 2.5 and
  3.5 lines the model has said 73.8% and it has happened 60.3%. Five
  explanations were tested and all five failed — shrinkage strength
  (`K_PRIOR_BF = 250` looked optimal; **see V12.2, this was measured on 50
  of 196 caches and is WRONG**), distribution width (standardised residual variance 0.957 ±
  0.080), projection-above-recent-form (r = −0.045, p = 0.657), times
  through the order (real, already absorbed), and bad-and-short coupling
  (r = −0.041 — the wrong SIGN; long outings carry LOWER rates).
  So the symptom is reported and no cause is claimed: a red pill at the
  top of the Pitchers tab, only on nights with low-line OVERs the model
  likes, plus a live calibration curve in Results. `backfill_k_dist.py`
  repaired both — they were rendering nothing, because `k_dist` entered
  the row log that same day and the 318 existing rows had none; the
  distributions were in `cache/pitchers_{date}.csv` all along and 315 of
  318 joined back on `(game_date, pitcher_id)`.
  New: `distill_caches.py` + `distilled.py`, `pull_caches.py`,
  `lab_tto.py`, `lab_tto_who.py`, `lab_coupling.py`, `lab_log5.py`,
  `lab_shrinkage.py`. `harness.py` gained `st.empty()`.

### V12.2 ✅ (Sep 14, 2026) — the reruns, and three reversals
  `distill_caches.py` cut 196 caches to 4.3 MB and every lab was re-run on
  all of them. Three findings from earlier the same day reversed:
  - **`K_PRIOR_BF = 250` is too strong.** See NEXT item 0. The claim that
    "soft arms BEAT their own rate" was a 50-cache artifact: at 196 they
    come in at 3.97 K against a shrunk projection of 4.11.
  - **The per-pitcher order penalty DOES persist.** Split-half r = -0.025
    (z = -0.16) on 42 pitchers became r = +0.2105 (z = +2.05) on 115, with
    the spread itself at z = +3.53 and an implied true spread of 2.46
    points. And arsenal depth predicts it, in the direction Nolan argued:
    mix entropy r = +0.209, pitch types r = +0.190, fastball share
    r = -0.128 — all three consistent, none individually decisive.
  - **The penalty is multiplicative, not additive.** Held-out K rate vs
    penalty r = -0.182, z = -2.84; power arms lose 4.89 points where soft
    arms lose 2.99, and a "keep 0.842 of your rate" form fits eight times
    better than a flat subtraction.
  None of this changes the "already absorbed" conclusion, which is
  structural: `k_rate` is total K over total BF across all three looks, so
  any look factor — per-pitcher, multiplicative or otherwise — must
  preserve the mean, and then it only redistributes within the start.
  `lab_coupling.py` stayed dead and got more so: r = -0.0662 at z = -6.17,
  still the wrong SIGN.

### V12.3 ✅ (Sep 15, 2026) — the velocity marker confirmed, a fourth reversal
  `lab_pitcher_form.py` re-run on all 196 caches (11,148 starts, up from
  4,599), now reading `cache/distilled/` like the rest.
  - **Velocity holds and strengthens**: r = +0.0815, z = +8.59 against a
    200-permutation null. Survives the seasonal-arc control (+0.0821 with
    season removed) and still does not predict batters faced (p = 0.41).
    Worth +0.36 K over 23 batters at 2 sd off his own norm.
  - **Recent K rate reverses from noise to real** — r = +0.019, p = 0.21 on
    50 caches became r = +0.0335, z = +3.54 on 196. But it does NOT survive
    alongside velocity: the two correlate at r = +0.19, and fitted together
    recent K rate drops to z = +1.91 and +0.08 K while velocity is
    untouched at z = +8.10. So the original decision to build the marker on
    velocity and NOT on recent strikeout rate still stands — now for a
    better reason than the one it was made for.
  - **The control was the bug.** It was a single shuffle with a fixed seed,
    and on this many starts one draw has an SE near 0.0095. It drew +0.0183
    for recent K rate (p = 0.053), which reads like the null firing and the
    real effect being contaminated. It was neither — averaged over 200
    permutations the null sits at -0.0000 +/- 0.0095. A one-draw control is
    not conservative, just noisy, and this one was noisy in the direction
    that would have retired a real effect.

### V12.4 ✅ (Sep 15, 2026) — two model changes, both from the full sample
  The first changes to the model itself after a week of measurement.
  - **`K_PRIOR_BF` 250 -> 150.** 250 is not wrong in aggregate (total bias
    is flat at -0.06 K across the whole range) but wrong as a TILT: on the
    model's own trailing-12-month shape over 9,723 starts it over-projects
    soft arms by 0.204 K while letting power arms sag by 0.146, a 0.35 K
    spread. At 150 that spread is 0.03 K. Soft arms carry the 2.5 and 3.5
    lines, so this is worth ~3 points of probability exactly where the
    calibration curve is worst.
  - **The velocity marker now reaches the projection.** It was computed by
    `pitcher_form.py` AFTER `predict_slate` ran, so the dashboard showed it
    beside a number it could not affect. `pitcher_form.velocity_for()` now
    serves it from the statcast caches and `predict_slate` calls it before
    projecting; run_slate's step order and `pitcher_form.main()` are
    unchanged. Slope +0.0093 per sd, clamped at +/-2 (4.9% of starts reach
    it, ~0.43 K at the clamp), rate only — velocity does not predict
    batters faced (p = 0.41).
  - **The slope is not the number the lab prints.** +0.0114 expanding and
    uncapped; +0.0106 against a 12-month baseline, because `k_rate` is
    already a 12-month rate and would count part of the decline twice;
    +0.0093 once the production window definition is used, since
    `pitcher_form` excludes the recent window from its baseline and the lab
    does not. Measuring the slope in the exact form it ships is the whole
    point of the last three sessions.
  - `test_velocity.py` (16 checks) guards the two silent failures: a NaN z
    propagating into a blank projection, and the clamp not binding.

### V12.5 ✅ (Sep 15, 2026) — the strikeout-form marker
  Velocity says how hard he is throwing; it does not say how he has been
  DOING. On 8 sampled pitchers the two markers disagreed on 6. Landen Roupp
  read velocity Normal (z = -0.21) with his last four starts 2.15 sigma
  under his own strikeout rate.
  - `kform_z` / `kform_state` in `pitcher_form.py`, same shape as the
    velocity marker; a **Form** column beside Velo; four columns kept by
    `score_slate` so it can grade itself.
  - **Display only, permanently.** Recent K rate is real but does not
    survive next to velocity (jointly z = +1.91 against velocity's +8.10,
    the two correlate at +0.19). Feeding both counts the same signal twice.
  - Window 4, cut 1.25, measured: Hot beats Cold by 1.8 points of K rate on
    the next start, z = +5.13, firing on 30% of rows. The window/cut cells
    tested are inside each other's error bars.
  - **The first measurement said +4.5% at z = +12.67 and was wrong** — `z`
    and the outcome were both computed against the same baseline, so
    baseline noise pushed both the same way. De-meaning within pitcher cut
    the effect by a factor of three. Same shared-denominator trap as the
    order-penalty lab.

### V12.6 ✅ (Sep 15, 2026) — hitter row log
  `scoring_log.csv` keeps per-slate aggregates only, so band-level
  calibration — the thing that caught the strikeout over-confidence — could
  not be done for hitters at all. `score_slate._log_hitter_rows` now writes
  one row per clean hitter-game to `cache/hitter_row_log.csv`: every
  `prob_*` the slate carried, every outcome, lineup slot, and the hot/cold
  marker. ~220 rows a slate, ~25 KB. Re-scoring REPLACES a date rather than
  appending it, matching the other two logs; `test_hitter_log.py` (14
  checks) guards that specifically. Backfill by re-running `score_slate.py`
  on past dates.
  Also corrected: **HR is not a coin flip.** That belief came from a V2
  training result; graded live it is the best-ordered hitter prop on the
  board (AUC 0.625, z = +7.5, 16 slates, said 11.52% / happened 11.50%). It
  still belongs only on LOTTO slips — `tier_of` is pure probability and HR
  clears the 28% MEDIUM cut on 6 of 1,367 rows — but because it is a long
  shot, not a bad bet.

### V12.7 ✅ (Sep 24, 2026) — retry on the odds fetch, and the asymmetry it turns on
  A failed odds pull costs a window that cannot be re-bought, so `_get()`
  — the single HTTP choke point in `data/odds_lines.py` — now retries.
  Three attempts, exponential backoff from 1.5s with +25% jitter, under
  10 seconds in the worst case so it cannot itself eat a lock time.
  **But it retries only the failures that provably cost nothing**, and the
  reason is that the two mistakes are not symmetric:
  - **Giving up too early is free.** `already_captured()` reads
    `odds_{date}.csv` and returns the games already held, so a re-run buys
    only what is missing. Recoverable at zero cost.
  - **Retrying too eagerly is unrecoverable.** Player props are billed PER
    EVENT. If the first request reached the server, the retry is a second
    charge and nothing gives it back.
  So: connection refused/reset, DNS failure, network unreachable and 5xx
  are retried — none of them reached a server that could bill. **Timeouts
  are NOT**, because twenty seconds of silence is not proof of absence;
  the API may have served a slow query and charged for it while we stopped
  listening. A truncated body is not retried either — a half-read answer is
  proof the server answered. 4xx is never retried: 401 is a bad key, 422
  bad parameters, and 429 means the quota is gone and asking again makes it
  worse. Ambiguity falls on the do-not-retry side by construction.
  `test_odds_retry.py` asserts on the **number of urlopen calls** in every
  case, because the call count is the credit count. Verified the way the
  other guards were — by removing the timeout clause first and watching all
  three timeout cases fail — then restored. No dependency added;
  `odds_lines` uses `urllib`, so `tenacity` and `requests` were not needed.

### WATCH — both changes are live and neither has been graded
  Every pitcher on the board moves tonight. `pitcher_row_log.csv` records
  `k_rate` and the `velo_*` fields per start, so the two are separable
  after the fact. Before trusting either, check the Results calibration
  curve after ~10 slates: the low-line band should close some of its
  -13.5 points if the prior change did what it measured.

### NEXT — what the evidence points at, in order
  0. ~~**Lower `K_PRIOR_BF` from 250.**~~ **DONE (V12.4)** — now 150.
  0b. ~~**Wire the velocity marker in**~~ — **DONE (V12.4).**
  1. **The information gap to the market** (r 0.46 vs 0.55). Not a
     calibration fix — the model is at 82% of its own optimal width and
     sits on the diagonal. It needs things it does not have: announced
     pitch limits, rest and injury news, umpire, catcher, weather.
  2. **Let the two markers speak.** The pitcher velocity marker has a
     measured prior (~+0.5 K hot minus cold) and zero live slates; the
     hitter marker has 13 slates and nothing. Neither enters the model
     until its own Results row says so.
  3. **The correlation number.** Slips are logged with whether they reuse
     a game. Once SAFE has ~40 nights, the gap between correlated and
     clean slips IS the correlation, and the "all hit" figure can stop
     being too low by an unknown factor.
  4. **`prob_k` is still written and never graded** — the truth column is
     already in the PA table, it needs adding to `BINARY_PROPS`.

---

## Libraries
- pybaseball — Statcast, Baseball-Reference data (free, no API key)
- pandas (pinned `<3.0` — see Known Technical Decisions)
- scikit-learn — LogisticRegression, metrics