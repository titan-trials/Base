# Baseball Predictor

A free, pybaseball-based MLB player-prop model. It projects every prop for
tonight's slate before first pitch, commits bet slips to disk so they can be
graded honestly afterwards, and serves the whole thing as a Streamlit
dashboard.

No paid data. The only thing that costs anything is the strikeout line
capture, which runs on The Odds API free tier — **500 credits a month, about
25 slates.**

This file is how to run it. **`CONTEXT.md` is why** — every decision, every
measurement behind it, and every hypothesis that died. If a number here looks
arbitrary, it is explained there.

---

## Setup

```powershell
python -m pip install -r requirements.txt
```

**Use `python -m pip`, not bare `pip`.** The venv was created under
`Downloads\` and the folder later moved to OneDrive. Windows venvs are not
relocatable — the launchers in `Scripts\` have the old absolute path compiled
into them, so `pip.exe` fails with "Unable to create process". `python -m pip`
bypasses the launcher.

Two pins are load-bearing and must not be "upgraded":

- **`pandas==2.3.3`** — must stay below 3.0. pybaseball's
  `schedule_and_record()` uses a chained-assignment `inplace=True` pattern
  that silently no-ops under the Copy-on-Write behaviour pandas 3.0 makes
  mandatory. It does not raise; it just returns wrong data.
- `requirements.txt` is **UTF-8**. PowerShell's `pip freeze > file` writes
  UTF-16, which pip cannot read back. To regenerate:
  `python -m pip freeze | Out-File -Encoding utf8 requirements.txt`

### The API key

Only the odds step needs one. Everything else runs without it.

```powershell
$env:ODDS_API_KEY = "yourkey"        # this session
```

or put it in a file called `.odds_api_key` in the project root (gitignored):

```powershell
Set-Content -Path .odds_api_key -Value 'yourkey' -Encoding ascii -NoNewline
```

Environment variable wins. The key is never accepted as a literal in code.
Free tier: <https://the-odds-api.com/>

---

## The nightly loop — two commands

### Before the games

```powershell
python run_slate.py                    # today
python run_slate.py 2026-09-14         # a specific date
python run_slate.py --no-odds          # free: everything but step 4
python run_slate.py --dry-run          # print the plan, spend nothing
```

Six steps, in the order they actually depend on. Only step 4 costs anything.

| # | step | script | writes |
|---|---|---|---|
| 1/6 | Lineups | `check_lineups.py` | |
| 2/6 | Predictions | `predict_slate.py` | `slate_`, `pitchers_`, `teams_` |
| 3/6 | Recent form | `pitcher_form.py` | `pitcher_form_`, velo marker |
| 4/6 | Strikeout props | `data.odds_lines` | `odds_`  **← COSTS** |
| 5/6 | Model vs market | `compare_market.py` | `market_compare_` |
| 6/6 | Commit slips | `slips.py` | `slips_` |

It ends with a CREDITS block showing what was spent and what remains.

**Run the odds step as late as you can before the earliest first pitch.**
Predictions are protected after the fact — `preserve_committed_rows` keeps
the number committed to before a game began — but odds have no equivalent
and cannot be re-fetched. A game already underway is a price lost forever.
`run_slate` prints how many of those there are before anything runs.

Everything except step 4 is free, so the pattern that fits the budget is:
poll lineups freely → re-run predictions as often as you like → **one** odds
capture, late → compare.

### After the games

```powershell
python score_slate.py 2026-09-14
```

Twelve steps. Slips are graded as steps 6–10, reusing the Statcast refresh
step 3 already did rather than making a second one over the network.

```powershell
python score_slips.py 2026-09-14 --dry-run   # re-grade slips only
```

**Re-running `score_slate.py` on a past date is safe and is how you
backfill.** All three row logs replace a date rather than appending it —
odds arrive late and boxscores get corrected, so re-running is by design.

### The dashboard

```powershell
streamlit run dashboard.py
```

Tabs: Slate / Games / Pitchers / All hitters / Bet ready / Results. It reads
committed CSVs and rebuilds nothing, so what is on screen is what was
graded. It imports exactly one project module (`slips.py`, the single
definition of what a slip is).

The deployed copy on Streamlit Community Cloud reads whatever is **committed
to git**. If a panel is on your machine but missing on the website, the cause
is almost always that its cache file is not whitelisted in `.gitignore` — see
*What is tracked* below.

---

## Spending credits deliberately

```powershell
python -m data.odds_lines 2026-09-14              # fill in missing games
python -m data.odds_lines 2026-09-14 --refresh    # re-buy a later line
```

Player props are billed **per event** (~15 a night); game lines are billed
per market for the whole slate (2 a night). About 17 a night against 500 a
month.

**A plain re-run is nearly free.** `already_captured()` reads
`odds_{date}.csv` and skips any game already in it, so a second run buys only
what is missing. `--refresh` deliberately ignores that and re-buys.

The fetch retries connection failures, DNS failures and 5xx — three attempts,
backoff from 1.5s with jitter, under 10 seconds total. It **never** retries a
timeout or a truncated response, because either may already have been served
and billed, and it never retries 4xx. If it gives up, just run it again.

---

## Labs — measure before building

Nothing in the model gets built until a lab says it exists. Several of these
returned honest negatives, which is a result, not a failure.

```powershell
python lab_opponent.py        # sample | effect | persist
python lab_k_spread.py        # market | spread | calib
python lab_pitcher_form.py    # velocity vs recent-K-rate form
python lab_tto.py             # times through the order
python lab_log5.py            # the batter x pitcher combination
python flag_lab.py --origins 3
```

Labs read `cache/distilled/` (built by `distill_caches.py`) rather than the
1.2 GB of raw Statcast caches. Each lab applies its own `MIN_BF` filter after
loading — the distiller does not do it for them.

---

## Tests

None of these need the network or a key.

```powershell
python test_workload.py       # 32 assertions on the workload model
python test_relief_tier.py    # the four expected_bf tiers
python test_slips.py          # slip building and slip grading
python test_velocity.py       # 16 checks on the velocity adjustment
python test_hitter_log.py     # 14 checks; re-scoring must REPLACE a date
python test_odds_retry.py     # which HTTP failures may be retried
python check.py               # renders every dashboard tab path
```

`check.py` also carries three guards that exist because each of them broke
once: the live clock is present, every cache file the dashboard reads is
whitelisted in `.gitignore`, and no `st.expander` is nested inside another
(Streamlit forbids it at runtime, so only an AST walk catches it before
deploy).

---

## Caching, and what is tracked in git

`cache/` is created automatically and is where everything lands. Cache
invalidation is **manual** — delete the relevant file to force a fresh pull,
which you must do after adding a feature column upstream (old cached rows
won't have it and the code will `KeyError`).

`.gitignore` ignores `cache/*` and then negates the exceptions. The rule:

> **Anything the dashboard READS, or any record that ACCUMULATES, has to be
> tracked. Everything else in `cache/` is regenerable; these are not.**

Tracked: `slate_*`, `pitchers_*`, `pitcher_form_*`, `slips_*`, `slip_log`,
`odds_*`, `market_log`, `credit_log`, `market_compare_*`, `scoring_log`,
`pitcher_scoring_log`, `pitcher_row_log`, `hitter_row_log`, `form_log`,
`team_scoring_log`, `teams_*`, `team_lines`.

`odds_*.csv` is the least regenerable file in the project — historical odds
are not on the free tier, so a night not captured before first pitch can
never be benchmarked by anyone, ever.

Never tracked: `statcast_player_*` (~1.2 GB), `statcast_pitcher_*`,
`batting_lines`, `lineup_slots`, `game_context`, `pitchmix_*`/`pitchcontrol_*`
(~1,900 files). Losing these costs time, not information — and `.git` was
1.1 GB once.

**`git add -A` skips ignored files silently.** Adding a new dashboard panel
means adding its cache file to the whitelist in the same commit, or the
deployed app renders without it and says nothing.

---

## If something looks wrong

| symptom | first thing to check |
|---|---|
| A panel works locally, not on the website | its cache file is not whitelisted in `.gitignore` |
| "No slates scored yet" with data on disk | same |
| `pip` fails "Unable to create process" | use `python -m pip` |
| pybaseball `pitching_stats()` returns 403 | known upstream (FanGraphs anti-bot). Not fixable here; pitcher stats come from Statcast directly |
| First run of anything is very slow | pybaseball's rate-limit delays. Later runs read `cache/` |
| A prop reads fine in aggregate but feels off | check the **band-level** calibration in Results, not the average — `pitcher_row_log.csv` and `hitter_row_log.csv` exist for exactly this |
| Odds step gave up | just re-run it; captured games are skipped |

---

## House rules

- **Measure before building.** Two feature requests in V12 were answered by
  a lab script instead of a feature.
- **Commit the number before grading it.** Slips are written to disk before
  first pitch; the dashboard reads that file rather than rebuilding, so the
  slip on screen and the slip in the log are the same object.
- **A guard is not trusted until it has been watched to fail.** The
  `.gitignore` check, the nested-expander check and the odds retry policy
  were each verified by deliberately breaking them first.
- **Report expected-vs-actual, never a win/loss record.** A LOTTO tier going
  0-for-200 is exactly what a 0.4% bet should look like. Error bars use the
  Poisson-binomial `sqrt(Σ p(1−p))`, not `sqrt(n·p̄(1−p̄))`.
