"""
Grade last night's bet slips, whole slips rather than legs.

    python score_slips.py                 # yesterday
    python score_slips.py 2026-09-13
    python score_slips.py --dry-run       # grade, print, write nothing

Reads cache/slips_{date}.csv -- the slips committed before first pitch by
slips.py -- and asks of each one: did EVERY leg land?

WHY THIS IS SEPARATE FROM score_slate.py
----------------------------------------
It reuses score_slate's own loaders rather than reimplementing them, so
the outcomes a slip is graded against are byte-for-byte the ones the
per-leg scorer uses. Two paths to "what actually happened" is how they
drift, which is the argument iter_scoreable already makes inside
score_slate.

WHAT THE NUMBERS CAN AND CANNOT SAY
-----------------------------------
This is the part worth reading before the first result arrives.

Twenty slips a night, and about 2.3 of them are expected to land:

    SAFE     ~1.7 a night      one hit every 0.6 nights
    MEDIUM   ~0.3 a night      one hit every 3 nights
    MIXED    ~0.2 a night      one hit every 5 nights
    LOTTO    ~0.02 a night     one hit every 56 nights

So a win/loss record is a real answer for SAFE and close to meaningless
everywhere else. A LOTTO record of 0-for-200 is not a failure; it is
exactly what a 0.4% bet is supposed to look like, and reading it as a
failure is the mistake this file exists to prevent. To detect a 20%
miscalibration takes roughly 38 nights for SAFE, 286 for MEDIUM, 526 for
MIXED and 5,500 for LOTTO.

That is why the output is EXPECTED versus ACTUAL with a standard error,
not a record. Expected is the sum of the committed slip probabilities,
which accumulates information every night even when nothing hits.

THE ONE THING THIS CAN MEASURE THAT NOTHING ELSE CAN
----------------------------------------------------
A slip's probability is the product of its legs, which is right only when
the legs are independent. Two legs from one lineup share a pitcher, a park
and a night, so they land together more often than the product implies --
the Bet ready tab says so in words and has never been able to say by how
much.

Slips are logged with whether they reuse a game, so pooling correlated
slips separately turns that warning into a number. If correlated SAFE
slips beat their product and clean ones sit on it, the gap IS the
correlation, measured rather than asserted.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LOG = os.path.join(CACHE, "slip_log.csv")

# How a leg's prop maps onto a column of the actuals frame. Count props
# are graded "strictly greater than the line", the same convention
# prob_over uses -- over 1.5 hits means two, not one.
BINARY = {"prob_hr": "got_hr", "prob_hit": "got_hit", "prob_walk": "got_walk"}
COUNTS = {"prob_hrr_over_": "hrr", "prob_tb_over_": "total_bases",
          "prob_hits_over_": "hits"}


def grade_leg(prop_key, line, actual_row):
    """
    1, 0, or NaN for a single leg.

    NaN means "cannot be graded" -- the player did not appear, or no
    official line exists for him. That is deliberately NOT a miss: a slip
    containing an ungradeable leg is dropped rather than counted against
    the model, because a scratched hitter is not evidence about a
    probability.
    """
    if actual_row is None:
        return float("nan")
    if prop_key in BINARY:
        v = actual_row.get(BINARY[prop_key])
        return float("nan") if pd.isna(v) else float(v > 0)
    for prefix, col in COUNTS.items():
        if str(prop_key).startswith(prefix):
            try:
                thresh = float(str(prop_key)[len(prefix):])
            except ValueError:
                return float("nan")
            v = actual_row.get(col)
            return float("nan") if pd.isna(v) else float(v > thresh)
    if str(prop_key).startswith("pitcher_k_"):
        v = actual_row.get("strikeouts")
        if pd.isna(v) or pd.isna(line):
            return float("nan")
        return float(v > line) if prop_key.endswith("over") else float(v < line)
    return float("nan")


def hitter_actuals(slips, date, verbose=True):
    """{(player_id, game_pk): row} from score_slate's own loader."""
    import score_slate as ss
    preds = ss.load_predictions(date)
    played = ss.load_actuals(preds, date, verbose=verbose)
    if "batter" in played.columns and "player_id" not in played.columns:
        played = played.rename(columns={"batter": "player_id"})
    out = {}
    for _, r in played.iterrows():
        out[(int(r["player_id"]), int(r["game_pk"]))] = r
    return out


def pitcher_actuals(slips, date, verbose=True):
    """{(pitcher_id, game_pk): row} of official starter lines."""
    from data.pitching_lines import get_pitching_lines
    pks = sorted({int(x) for x in slips["game_pk"].dropna().unique()})
    if not pks:
        return {}
    lines = get_pitching_lines(pks, verbose=verbose, refetch=pks)
    if lines.empty:
        return {}
    starters = lines[lines["is_starter"] == 1]
    return {(int(r["pitcher_id"]), int(r["game_pk"])): r
            for _, r in starters.iterrows()}


def grade(slips, hitters, pitchers):
    """One row per SLIP: how many legs landed, and whether all of them did."""
    legs = slips.copy()
    got = []
    for _, r in legs.iterrows():
        pid = r.get("player_id")
        pk = r.get("game_pk")
        key = (int(pid), int(pk)) if pd.notna(pid) and pd.notna(pk) else None
        src = pitchers if r["kind"] == "arm" else hitters
        got.append(grade_leg(r["prop_key"], r.get("line"),
                             src.get(key) if key else None))
    legs["hit"] = got

    rows = []
    for (w, t), g in legs.groupby(["window", "tier"]):
        n = len(g)
        graded = int(g["hit"].notna().sum())
        rows.append({
            "window": int(w), "tier": t, "n_legs": n,
            "n_graded": graded,
            "slip_p": float(g["slip_p"].iloc[0]),
            "legs_hit": float(g["hit"].sum(skipna=True)),
            # A slip with an ungradeable leg is not a miss. It is not
            # evidence either way, so it is dropped from the totals and
            # counted here so the drop is visible rather than silent.
            "hit": (float(g["hit"].min()) if graded == n else float("nan")),
            "n_games": int(g["n_games"].iloc[0]),
            "solo": bool(g["solo"].iloc[0]),
            "correlated": bool(g["clash"].any() or g["solo"].iloc[0]),
            "legs": " + ".join(f'{x["name"]} {x["prop"]}'
                               for _, x in g.iterrows()),
        })
    return pd.DataFrame(rows), legs


def report(scored):
    ok = scored.dropna(subset=["hit"])
    print(f"\n{'=' * 72}\nSLIPS\n{'=' * 72}")
    dropped = len(scored) - len(ok)
    if dropped:
        print(f"  {dropped} slip(s) had a leg that could not be graded "
              f"(scratched, or no official line) and are left out.")
    if ok.empty:
        print("  Nothing gradeable.")
        return
    print(f"\n  {'tier':8} {'slips':>5} {'expected':>9} {'hit':>4} "
          f"{'gap':>7}")
    for t in ("MIXED", "SAFE", "MEDIUM", "LOTTO"):
        sub = ok[ok["tier"] == t]
        if sub.empty:
            continue
        exp, act = sub["slip_p"].sum(), sub["hit"].sum()
        print(f"  {t:8} {len(sub):>5} {exp:>9.2f} {act:>4.0f} "
              f"{act - exp:>+7.2f}")
    print(f"  {'-' * 38}")
    print(f"  {'all':8} {len(ok):>5} {ok['slip_p'].sum():>9.2f} "
          f"{ok['hit'].sum():>4.0f} {ok['hit'].sum() - ok['slip_p'].sum():>+7.2f}")

    winners = ok[ok["hit"] > 0]
    if len(winners):
        print(f"\n  LANDED")
        for _, r in winners.sort_values("slip_p").iterrows():
            print(f"    {r['tier']:7} {r['slip_p']:6.2%}  {r['legs']}")
    else:
        print("\n  No slip landed in full. On a single night that is the "
              "usual outcome — about 2.3 of 20 are expected to.")

    # Every slip's near-miss count, which is the thing a win/loss record
    # throws away: a SAFE slip that went 2-for-3 is a different night from
    # one that went 0-for-3, and both read as "lost".
    print(f"\n  LEGS WITHIN SLIPS")
    for t in ("MIXED", "SAFE", "MEDIUM", "LOTTO"):
        sub = ok[ok["tier"] == t]
        if sub.empty:
            continue
        tot, hit = sub["n_legs"].sum(), sub["legs_hit"].sum()
        print(f"    {t:8} {hit:.0f} of {tot:.0f} legs "
              f"({hit / tot:.0%})" if tot else "")


def main():
    ap = argparse.ArgumentParser(description="Grade committed bet slips.")
    ap.add_argument("date", nargs="?", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="grade and print, but do not append to slip_log")
    args = ap.parse_args()
    date = args.date or (pd.Timestamp.today()
                         - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    path = os.path.join(CACHE, f"slips_{date}.csv")
    if not os.path.exists(path):
        print(f"No cache/slips_{date}.csv.\n"
              f"Slips are committed before first pitch by slips.py — a slate "
              f"predicted before that step existed has none, and rebuilding "
              f"them now would grade today's code against last night's "
              f"games rather than what was actually on the board.")
        return 1
    slips = pd.read_csv(path)
    if slips.empty:
        print("Slip file is empty.")
        return 1

    print(f"SCORING SLIPS {date}")
    print(f"  {slips.groupby(['window', 'tier']).ngroups} slips, "
          f"{len(slips)} legs.")

    hitters = hitter_actuals(slips, date)
    arms = slips[slips["kind"] == "arm"]
    pitchers = pitcher_actuals(arms, date) if len(arms) else {}

    scored, legs = grade(slips, hitters, pitchers)
    report(scored)

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    scored = scored.assign(game_date=date)
    keep = scored.dropna(subset=["hit"])
    if keep.empty:
        print("\n  Nothing gradeable; slip_log unchanged.")
        return 0
    # Merge rather than append, the same rule the other logs follow: a
    # re-run of a date must replace that date's rows, not double them.
    if os.path.exists(LOG):
        try:
            prior = pd.read_csv(LOG)
            prior = prior[prior["game_date"].astype(str) != date]
            keep = pd.concat([prior, keep], ignore_index=True)
        except Exception:
            pass
    keep.to_csv(LOG, index=False)

    run = keep.dropna(subset=["hit"])
    print(f"\n  cache/slip_log.csv — {len(run)} slips over "
          f"{run['game_date'].nunique()} night(s).")
    if run["game_date"].nunique() >= 2:
        print(f"  {'tier':8} {'slips':>5} {'expected':>9} {'hit':>4} "
              f"{'gap':>8} {'verdict':>28}")
        for t in ("MIXED", "SAFE", "MEDIUM", "LOTTO"):
            sub = run[run["tier"] == t]
            if sub.empty:
                continue
            exp = sub["slip_p"].sum()
            act = sub["hit"].sum()
            # Poisson-binomial: the variance of a sum of independent
            # indicators is the sum of p(1-p), not n*p*(1-p).
            se = float(np.sqrt((sub["slip_p"] * (1 - sub["slip_p"])).sum()))
            verdict = ("not yet separable from noise"
                       if se <= 0 or abs(act - exp) <= 2 * se
                       else ("running HOT" if act > exp else "running COLD"))
            if exp < 1:
                verdict = "too few expected to read"
            print(f"  {t:8} {len(sub):>5} {exp:>9.2f} {act:>4.0f} "
                  f"{act - exp:>+8.2f} {verdict:>28}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
