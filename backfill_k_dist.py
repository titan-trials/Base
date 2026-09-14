"""
Put k_dist and outs_dist into the rows of pitcher_row_log.csv written
before score_slate started keeping them.

    python backfill_k_dist.py --dry-run
    python backfill_k_dist.py

WHY THIS IS NEEDED
------------------
On 2026-09-14 `_log_pitcher_rows` gained `k_dist` and `outs_dist` in its
keep list, so that any line can be graded retroactively out of the row log
rather than only the two the slate happened to be priced at. That was the
whole point: the calibration problem lives at 2.5 and 3.5, and 2.5 was
never in K_LINES.

But the change only affects rows written AFTER it. The 318 rows already in
the log have no distribution attached, which means the two things built on
top of it -- the Results calibration curve and the low-line warning above
the Pitchers table -- are computing over an empty frame and rendering
nothing at all. They are not broken. They are dark, and they would stay
dark for however many nights it took to accumulate a new log from scratch.

They do not have to be. Every distribution those rows need was written at
the time, into `cache/pitchers_{date}.csv`, which is tracked and still
here. The row log has game_date and pitcher_id; the slate files have
game_date in the filename and pitcher_id in a column. That is a join.

WHAT IT WILL NOT DO
-------------------
Nothing is recomputed. A distribution is copied across exactly as it was
predicted on the night, or the row is left alone. Backfilling a row log
with a number produced by today's model would be the worst possible
outcome here -- the log is the only record of what the model actually said
before it knew the answer, and a calibration curve fitted to hindsight
would read perfect and mean nothing.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LOG = os.path.join(CACHE, "pitcher_row_log.csv")
COLS = ["k_dist", "outs_dist"]


def slate_frames():
    """Every archived slate, keyed by the date in its filename."""
    out = {}
    for path in sorted(glob.glob(os.path.join(CACHE, "pitchers_*.csv"))):
        m = re.search(r"pitchers_(\d{4}-\d{2}-\d{2})\.csv$",
                      os.path.basename(path))
        if not m:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "pitcher_id" not in df.columns:
            continue
        have = [c for c in COLS if c in df.columns]
        if not have:
            continue
        df = df[["pitcher_id"] + have].copy()
        df["pitcher_id"] = pd.to_numeric(df["pitcher_id"],
                                         errors="coerce").astype("Int64")
        out[m.group(1)] = df.dropna(subset=["pitcher_id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    if not os.path.exists(LOG):
        print(f"No {LOG}.")
        return 1
    log = pd.read_csv(LOG)
    print(f"pitcher_row_log.csv: {len(log)} rows")

    if "game_date" not in log.columns or "pitcher_id" not in log.columns:
        print("  log has no game_date/pitcher_id to join on.")
        return 1

    slates = slate_frames()
    print(f"  {len(slates)} archived slates carry a distribution")
    if not slates:
        return 1

    log["pitcher_id"] = pd.to_numeric(log["pitcher_id"],
                                      errors="coerce").astype("Int64")
    for c in COLS:
        if c not in log.columns:
            log[c] = pd.NA

    before = {c: int(log[c].notna().sum()) for c in COLS}
    filled, unmatched = 0, 0

    for date, sl in slates.items():
        mask = log["game_date"].astype(str).str[:10] == date
        if not mask.any():
            continue
        lookup = {c: dict(zip(sl["pitcher_id"], sl[c]))
                  for c in COLS if c in sl.columns}
        for idx in log.index[mask]:
            pid = log.at[idx, "pitcher_id"]
            if pd.isna(pid):
                continue
            hit = False
            for c, table in lookup.items():
                val = table.get(pid)
                # Only ever FILL. A row that already carries a distribution
                # was written by the new score_slate and is authoritative;
                # overwriting it from an archived slate would silently swap
                # in a file that may have been regenerated since.
                if val is not None and pd.isna(log.at[idx, c]):
                    log.at[idx, c] = val
                    hit = True
            filled += int(hit)
            if not hit and pd.isna(log.at[idx, "k_dist"]):
                unmatched += 1

    after = {c: int(log[c].notna().sum()) for c in COLS}
    for c in COLS:
        print(f"  {c}: {before[c]} -> {after[c]} rows populated")
    print(f"  {filled} rows filled, {unmatched} had no matching slate row")

    gradeable = int((log["k_dist"].notna()
                     & log.get("strikeouts", pd.Series(dtype=float)).notna()
                     ).sum())
    print(f"  {gradeable} rows now carry BOTH a distribution and an actual "
          f"— that is what the calibration curve counts.")

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    # The log is append-only history and the single record of what the
    # model said before it knew. A backup costs nothing and the alternative
    # is unrecoverable.
    backup = LOG.replace(".csv", "_prebackfill.csv")
    if not os.path.exists(backup):
        pd.read_csv(LOG).to_csv(backup, index=False)
        print(f"  backup written: {os.path.basename(backup)}")
    log.to_csv(LOG, index=False)
    print(f"  wrote {os.path.basename(LOG)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
