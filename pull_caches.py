"""
Add pitcher statcast caches, to settle the log5 combination question.

    python pull_caches.py --dry-run      # what it would pull, and why
    python pull_caches.py                # pull the default 28
    python pull_caches.py --n 40
    python pull_caches.py --verify       # just report where the sample stands

WHY
---
lab_log5.py asks the one opponent question still open: when a contact
hitter faces a power pitcher, does he strike out as much as the log5
formula says? Every version of that test leans the same way -- the hitter
resists more than the formula predicts -- and none of them clears two
sigma:

    contact vs power, within-batter   -0.72%   null -0.15% +/- 0.45   z = -1.27

That is not a dead result like lab_opponent.py, where the estimate sat on
top of its null. It is an underpowered one, and underpowered has a price
tag:

    observed 0.72 pts | current noise +/-0.45 | n = 104,087 PA
    to reach 2 sigma, noise must fall to +/-0.36 pts
    -> 1.56x the plate appearances = 162,635 PA
    -> about 78 cached pitchers instead of 50

So 28 more caches answers it, in whichever direction it goes. That is the
entire reason this script exists, and when the question is settled this
script has no further purpose.

WHAT IT PULLS AND IN WHAT ORDER
-------------------------------
Not random pitchers. Candidates come from the slates already run -- every
starter named in `cache/pitchers_{date}.csv` and every pitcher in
`pitcher_row_log.csv` -- ranked by how often they have actually turned up.

That ordering matters twice over. A pitcher who starts against Nolan's
slates regularly brings more plate appearances per cache AND makes the
model better on nights he pitches, so the same 178 MB does two jobs. A
pitcher pulled at random might be a September reliever with 40 plate
appearances to his name.

COST
----
Free in money. About 6.4 MB and roughly 2,600 plate appearances per
pitcher, so 28 of them is ~180 MB of disk and ~72,000 plate appearances.
Baseball Savant is rate-limited, so `refresh_pitcher` sleeps between calls
and this takes a few minutes rather than seconds. These files are NOT
tracked by git -- .gitignore excludes `cache/statcast_pitcher_*.csv`
deliberately, because putting them in a repository is what took .git to
1.1 GB once already.

SAFE TO STOP AND RERUN
----------------------
Every pull writes its own file before the next one starts, and pitchers
already on disk are skipped. Ctrl-C and run it again and it picks up where
it left off. It never touches an existing cache except through
`refresh_pitcher`, which is the same incremental top-up run_slate uses
nightly.
"""
import argparse
import glob
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
START = "2022-03-01"        # matches the span of the existing caches

# ---- the target, in the units this script can actually count -----------
#
# lab_log5.py needs ~162,635 plate appearances to get its noise down to
# +/-0.36 points. But that figure is in FILTERED plate appearances -- the
# lab drops any hitter under 40 PA and any pitcher under 200, because a
# rate built on fewer is a rumour. This script counts raw rows, and the
# two are not the same number: 124,087 raw currently yields 104,087
# usable, a ratio of 0.839.
#
# Counting raw rows against a filtered target would have reported the
# sample 84% complete when it is not, and told Nolan to pull 15 caches
# when the answer is 27.
TARGET_USABLE = 162_635
USABLE_SHARE = 0.839
TARGET_PA = int(TARGET_USABLE / USABLE_SHARE)


def cached_ids():
    out = set()
    for p in glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv")):
        base = os.path.basename(p)
        try:
            out.add(int(base.split("_")[-1].split(".")[0]))
        except ValueError:
            continue
    return out


def candidates():
    """Starters seen on past slates, most frequent first, uncached only."""
    seen, names = Counter(), {}
    files = sorted(glob.glob(os.path.join(CACHE, "pitchers_*.csv")))
    log = os.path.join(CACHE, "pitcher_row_log.csv")
    if os.path.exists(log):
        files.append(log)
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "pitcher_id" not in df.columns:
            continue
        col = "pitcher" if "pitcher" in df.columns else None
        for _, r in df.iterrows():
            pid = pd.to_numeric(r["pitcher_id"], errors="coerce")
            if pd.isna(pid):
                continue
            pid = int(pid)
            seen[pid] += 1
            if col and pid not in names and isinstance(r[col], str):
                names[pid] = r[col]
    have = cached_ids()
    return [(pid, n, names.get(pid, str(pid)))
            for pid, n in seen.most_common() if pid not in have]


def sample_now():
    """Plate appearances currently available to the labs."""
    total, files = 0, 0
    for p in glob.glob(os.path.join(CACHE, "statcast_pitcher_*.csv")):
        try:
            d = pd.read_csv(p, usecols=lambda c: c == "events",
                            low_memory=False)
        except Exception:
            continue
        files += 1
        total += int(d["events"].notna().sum())
    return files, total


def verify():
    files, pa = sample_now()
    print(f"{files} cached pitchers, {pa:,} plate appearances "
          f"(~{int(pa * USABLE_SHARE):,} usable after the lab's minimums)")
    pct = pa / TARGET_PA
    print(f"lab_log5.py needs ~{TARGET_PA:,} raw (~{TARGET_USABLE:,} usable) "
          f"to resolve the log5 question at 2 sigma — currently at {pct:.0%}")
    if pa >= TARGET_PA:
        print("  -> enough. Run: python lab_log5.py")
    else:
        # ~2,600 PA per cache, measured across the existing 50.
        need = int((TARGET_PA - pa) / 2600) + 1
        print(f"  -> about {need} more cache(s). Run: python pull_caches.py "
              f"--n {need}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Add pitcher statcast caches for lab_log5.py.")
    ap.add_argument("--n", type=int, default=28,
                    help="how many new pitchers to pull (default 28)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="report sample size and exit")
    args = ap.parse_args()

    if args.verify:
        return verify()

    files, pa0 = sample_now()
    print(f"cached now: {files} pitchers, {pa0:,} plate appearances")
    print(f"target:     ~{TARGET_PA:,} plate appearances\n")

    cand = candidates()
    if not cand:
        print("No uncached pitchers found in the slate files. Either every")
        print("starter seen so far is cached, or cache/pitchers_*.csv is "
              "missing.")
        return 1

    pick = cand[:args.n]
    print(f"{len(cand)} uncached starters available; taking the top "
          f"{len(pick)} by how often they have appeared:\n")
    for i, (pid, n, name) in enumerate(pick, 1):
        print(f"  {i:>3}. {name:<26} id {pid:<8} seen on {n} slate row(s)")

    if args.dry_run:
        print(f"\n--dry-run: nothing pulled. Estimated "
              f"{len(pick) * 2600:,} new plate appearances, "
              f"~{len(pick) * 6.4:.0f} MB.")
        return 0

    try:
        from data.pitcher_data import refresh_pitcher
    except Exception as exc:
        print(f"\nCould not import refresh_pitcher ({type(exc).__name__}: "
              f"{exc}).\nRun this from the project folder.")
        return 1

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"\npulling {START} to {end} — safe to Ctrl-C, reruns resume\n")
    ok = 0
    for i, (pid, _, name) in enumerate(pick, 1):
        try:
            got = refresh_pitcher(pid, START, end, verbose=True)
        except KeyboardInterrupt:
            print("\nstopped. Rerun to continue where this left off.")
            break
        except Exception as exc:
            print(f"  {i:>3}. {name:<26} FAILED ({type(exc).__name__}: {exc})")
            continue
        if got is None or got.empty:
            print(f"  {i:>3}. {name:<26} nothing returned")
            continue
        n_pa = int(got["events"].notna().sum()) if "events" in got else 0
        ok += 1
        print(f"  {i:>3}. {name:<26} {len(got):>7,} pitches, "
              f"{n_pa:>6,} plate appearances")

    files, pa1 = sample_now()
    print(f"\n{ok} pulled. {files} cached pitchers, {pa1:,} plate "
          f"appearances ({pa1 - pa0:+,}).")
    if pa1 >= TARGET_PA:
        print("Enough to settle it. Run: python lab_log5.py")
    else:
        need = int((TARGET_PA - pa1) / 2600) + 1
        print(f"About {need} more to reach the target — "
              f"python pull_caches.py --n {need}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
