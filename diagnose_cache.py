"""
Find the corrupted player caches, and say what SHAPE the corruption is.

    python diagnose_cache.py            # scan every player cache
    python diagnose_cache.py --gaps     # ALSO look for silently lost rows
    python diagnose_cache.py --fast     # stop each file after 5 bad rows
    python diagnose_cache.py 514888     # one player id

WHAT BROKE
----------
score_slate died on:

    ValueError: time data " Yordan"" doesn't match format "%Y-%m-%d"

`" Yordan"` is not a date, it is the back half of `"Alvarez, Yordan"`. A
row got split on the comma inside a quoted name, so every field after it
shifted left by one and a NAME landed in the game_date column.

That tells us the file is damaged, but not HOW, and the how decides the
fix:

  - a handful of rows in one file  -> the write was interrupted; delete
                                      and re-pull that player
  - the same break in many files   -> the writer is producing bad CSV and
                                      re-pulling just recreates it
  - whole file unreadable          -> encoding, not quoting

So this script reports the shape, and changes nothing. READ ONLY -- it
does not delete, rewrite or re-pull anything.

WHY IT DOES NOT USE PANDAS TO FIND THEM
---------------------------------------
pandas either raises or silently coerces; neither tells you which LINE is
wrong. csv.reader honours quoting exactly the way a correct writer does,
so a properly quoted "Alvarez, Yordan" is one field and a broken one is
two. A row whose field count differs from the header IS the corruption,
and that is checkable without parsing 1.2 GB into memory.
"""
import os
import re
import csv
import sys
import glob

csv.field_size_limit(10 ** 8)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
SHOW_ROWS = 3          # raw lines to print per damaged file
FAST_STOP = 5          # bad rows before giving up on a file, with --fast


def scan(path, fast=False):
    """Returns a report dict for one cache file. Never raises."""
    out = {"path": path, "rows": 0, "bad_width": [], "bad_date": [],
           "width": None, "error": None, "widths": {}}
    try:
        with open(path, "r", encoding="utf-8", errors="replace",
                  newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                out["error"] = "empty file"
                return out
            out["width"] = len(header)
            try:
                date_col = header.index("game_date")
            except ValueError:
                out["error"] = "no game_date column"
                return out

            for lineno, row in enumerate(reader, start=2):
                out["rows"] += 1
                out["widths"][len(row)] = out["widths"].get(len(row), 0) + 1

                if len(row) != out["width"]:
                    if len(out["bad_width"]) < 200:
                        out["bad_width"].append((lineno, len(row), row[:4]))
                elif date_col < len(row) and not DATE.match(row[date_col]):
                    if len(out["bad_date"]) < 200:
                        out["bad_date"].append(
                            (lineno, row[date_col], row[:4]))

                if fast and (len(out["bad_width"]) + len(out["bad_date"])
                             >= FAST_STOP):
                    out["error"] = "stopped early (--fast)"
                    break
    except Exception as exc:                      # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def gaps(path):
    """
    Lost rows that the field-count check CANNOT see.

    A dropped chunk of bytes only breaks the column count if it lands
    mid-row. If it swallowed whole lines, or stopped neatly on a field
    boundary, the file scans clean while quietly missing pitches -- and
    load_player_cache's dropna() would remove the evidence without a word.

    The invariant that catches it: within one at-bat, Statcast numbers the
    pitches 1, 2, 3... with no holes. A batter's cache holds only HIS
    at-bats, so at_bat_number is expected to skip; pitch_number inside an
    at-bat is not.

    Returns (holes, dupes, at_bats) where holes is a list of
    (game_pk, at_bat, missing_pitch_numbers).
    """
    seen = {}
    dupes = 0
    with open(path, "r", encoding="utf-8", errors="replace",
              newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0, 0
        try:
            gp = header.index("game_pk")
            ab = header.index("at_bat_number")
            pn = header.index("pitch_number")
        except ValueError:
            return None, 0, 0                  # not a pitch-level cache

        width = len(header)
        for row in reader:
            if len(row) != width:
                continue                        # already reported above
            try:
                key = (row[gp], row[ab])
                num = int(float(row[pn]))
            except (ValueError, IndexError):
                continue
            bucket = seen.setdefault(key, set())
            if num in bucket:
                dupes += 1
            bucket.add(num)

    holes = []
    for key, nums in seen.items():
        want = set(range(1, max(nums) + 1))
        missing = sorted(want - nums)
        if missing:
            holes.append((key[0], key[1], missing))
    return holes, dupes, len(seen)


def raw_lines(path, lineno, before=1, after=1):
    """The offending line and its neighbours, as written on disk."""
    got = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle, start=1):
            if lineno - before <= i <= lineno + after:
                got.append((i, line.rstrip("\n")))
            if i > lineno + after:
                break
    return got


def report_gaps(holey):
    """
    Classify the holes, because most of them are not damage.

    A hole that starts at pitch 1 -- missing 1, or 1-2, or 1-2-3 -- is the
    signature of a batter who entered an at-bat ALREADY IN PROGRESS. A
    pinch hitter sent up with two strikes on the man he replaced owns the
    rest of that at-bat, and this cache holds only his own pitches, so his
    at_bat_number legitimately begins at pitch 3. That is baseball, not a
    lost row.

    A hole in the MIDDLE -- present 1-4, missing 5-6, present 7 -- has no
    such explanation and is worth a look.

    Two independent facts in the 2026-09-24 scan said the leading holes
    are in the SOURCE data rather than in these files:

      - six players have the SAME hole in their canonical cache and in a
        legacy statcast_id*.csv pulled about eight weeks earlier, by
        different code. Two independent pulls do not lose the same pitch.
      - the game_pks span 2022 through 2026. A write that went wrong
        during one refresh cannot reach back four seasons.

    So this section reports, and does not accuse.
    """
    print()
    print("=" * 72)
    print("PITCH-NUMBER HOLES")
    print("=" * 72)
    if not holey:
        print("  None. Every at-bat numbers its pitches 1..N with no holes.")
        return

    leading, interior = [], []
    for rep in holey:
        lead = [h for h in rep["holes"] if h[2] == list(range(1, len(h[2]) + 1))]
        mid = [h for h in rep["holes"] if h not in lead]
        if mid:
            interior.append((rep, mid))
        if lead:
            leading.append((rep, lead))

    n_lead = sum(len(v) for _, v in leading)
    print(f"  {n_lead} at-bats across {len(leading)} files are missing only")
    print(f"  their LEADING pitches. Expected -- a batter who came in")
    print(f"  mid-at-bat owns only the rest of it. Not treated as damage.")

    dupes = [r for r in holey if r["dupes"]]
    if dupes:
        print(f"\n  {len(dupes)} files have DUPLICATE pitch numbers, which "
              f"dedupe should have removed:")
        for rep in dupes[:10]:
            print(f"    {os.path.basename(rep['path'])}: {rep['dupes']}")

    print()
    if not interior:
        print("  No interior holes. Nothing here needs explaining.")
        return
    n_mid = sum(len(v) for _, v in interior)
    print(f"  {n_mid} at-bats have an INTERIOR hole -- pitches missing from")
    print(f"  the middle of an at-bat, which substitution does not explain:")
    for rep, mid in interior:
        for gp, ab, missing in mid:
            print(f"    {os.path.basename(rep['path'])}  game {gp} "
                  f"at-bat {ab}: missing "
                  f"{', '.join(str(m) for m in missing[:10])}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv

    if args:
        files = []
        for pid in args:
            files += glob.glob(os.path.join(CACHE, f"statcast_player_{pid}.csv"))
            files += glob.glob(os.path.join(CACHE, f"statcast_id{pid}_*.csv"))
    else:
        files = (sorted(glob.glob(os.path.join(CACHE, "statcast_player_*.csv")))
                 + sorted(glob.glob(os.path.join(CACHE, "statcast_id*_*.csv"))))

    if not files:
        print("No player caches found in cache/.")
        return 1

    total_mb = sum(os.path.getsize(f) for f in files) / 1e6
    print(f"Scanning {len(files)} files, {total_mb:,.0f} MB"
          f"{' (--fast)' if fast else ''}.\n")

    do_gaps = "--gaps" in sys.argv
    damaged, clean, unreadable, holey = [], 0, [], []
    for i, path in enumerate(files, start=1):
        if i % 25 == 0 or i == len(files):
            print(f"  ...{i}/{len(files)}", flush=True)
        rep = scan(path, fast=fast)
        if do_gaps and not rep["error"]:
            holes, dupes, at_bats = gaps(path)
            # Duplicates matter even when nothing is missing: dedupe runs
            # on every save, so a survivor means the key is not unique.
            if holes or dupes:
                rep["holes"] = holes
                rep["dupes"] = dupes
                rep["at_bats"] = at_bats
                holey.append(rep)
        if rep["error"] and not rep["error"].startswith("stopped"):
            unreadable.append(rep)
        elif rep["bad_width"] or rep["bad_date"]:
            damaged.append(rep)
        else:
            clean += 1

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  clean      {clean}")
    print(f"  damaged    {len(damaged)}")
    print(f"  unreadable {len(unreadable)}")

    for rep in unreadable:
        print(f"\n  [unreadable] {os.path.basename(rep['path'])}"
              f"  -- {rep['error']}")

    for rep in damaged:
        name = os.path.basename(rep["path"])
        nbad = len(rep["bad_width"]) + len(rep["bad_date"])
        print()
        print("-" * 72)
        print(f"{name}")
        print(f"  {rep['rows']:,} rows, header has {rep['width']} columns")
        print(f"  {len(rep['bad_width'])} rows with the WRONG COLUMN COUNT"
              f"{'+' if len(rep['bad_width']) == 200 else ''}")
        print(f"  {len(rep['bad_date'])} rows where game_date is NOT a date"
              f"{'+' if len(rep['bad_date']) == 200 else ''}")

        widths = sorted(rep["widths"].items(), key=lambda kv: -kv[1])[:4]
        print("  column counts seen: "
              + ", ".join(f"{w} cols x{n:,}" for w, n in widths))

        share = nbad / max(rep["rows"], 1)
        if nbad <= 5 or share < 0.001:
            verdict = "a few rows; the rest of the file looks intact"
        elif share < 0.05:
            verdict = "scattered damage"
        else:
            verdict = "WIDESPREAD -- treat the whole file as suspect"
        print(f"  damaged share: {share:.2%} ({nbad} rows) -- {verdict}")

        # WHERE the damage sits is what names the cause. Bad rows bunched
        # at the end mean a write that stopped partway. Bad rows spread
        # through the file mean whatever wrote it was producing bad CSV
        # all along, and re-pulling would just make it again.
        bad_lines = sorted([n for n, _, _ in rep["bad_width"]]
                           + [n for n, _, _ in rep["bad_date"]])
        if bad_lines:
            last = rep["rows"] + 1
            span = f"{bad_lines[0]}"
            if len(bad_lines) > 1:
                span += f"-{bad_lines[-1]}"
            print(f"  damage sits at line {span} of {last} "
                  f"({100 * bad_lines[0] / last:.0f}% through the file)")
            # Deliberately NOT guessing the cause from position. The first
            # version of this script called anything not at EOF "the
            # writer", which was wrong: these files are fully rewritten
            # every refresh, so damage lands wherever the lost bytes were,
            # and one bad row in 10,000 is not a writer that emits bad CSV.

        for lineno, width, head in rep["bad_width"][:SHOW_ROWS]:
            print(f"\n  line {lineno}: {width} columns (want {rep['width']})")
            for n, text in raw_lines(rep["path"], lineno):
                mark = ">>" if n == lineno else "  "
                print(f"   {mark} {n}: {text[:160]}")

        for lineno, value, head in rep["bad_date"][:SHOW_ROWS]:
            print(f"\n  line {lineno}: game_date = {value!r}")
            for n, text in raw_lines(rep["path"], lineno):
                mark = ">>" if n == lineno else "  "
                print(f"   {mark} {n}: {text[:160]}")

    if do_gaps:
        report_gaps(holey)

    print()
    if damaged:
        ids = []
        for rep in damaged:
            m = re.search(r"statcast_(?:player_|id)(\d+)", rep["path"])
            if m:
                ids.append(m.group(1))
        print("  damaged player ids: " + " ".join(ids))
    print("  Nothing was changed. This script only reads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
