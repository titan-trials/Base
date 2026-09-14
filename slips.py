"""
Build tonight's bet slips, and commit them before first pitch.

    python slips.py                 # today
    python slips.py 2026-09-13
    python slips.py --legs 4

WHY THIS IS A FILE AND NOT JUST A DASHBOARD VIEW
------------------------------------------------
Nolan asked which of last night's slips hit. They could not be answered,
because the slips were never written down -- the Bet ready tab built them
at render time from the slate, so re-deriving them the next day would grade
whatever today's code and today's slate file happen to produce rather than
what he actually saw.

That is the rule this project keeps re-learning, and it is written into
.gitignore twice already: **a number you want to grade later has to be
committed before the game starts.** Predictions are protected that way
(preserve_committed_rows); odds are captured that way; slips were not.

So the slips are built here, once, and written to cache/slips_{date}.csv.
The dashboard READS that file rather than rebuilding, which means the slip
on screen and the slip that gets graded are the same object. About 60 rows
a night, a few KB.

WHAT A SLIP IS
--------------
Games are clustered by first pitch (a 30-minute gap is the break), and each
window gets four slips:

    MIXED    one leg from each tier, plus an arm -- a long shot riding
             along with two near-certainties
    SAFE     every leg a near-certainty
    MEDIUM   live, and well clear of the field
    LOTTO    long odds the model rates far above a typical player

Prop types mix freely inside a slip -- a hit, a walk, a total-bases line, a
pitcher's strikeout number. Risk levels only mix in MIXED.

WHAT THIS IS NOT
----------------
Not a record of what anyone bet. It is a record of what the model put on
the board, which is the thing that can be graded honestly against what
happened. Roughly 2.3 of the 20 slips a night are expected to land, and
1.7 of those are SAFE -- a LOTTO slip is expected about once every 56
nights, so its record will read 0-for-everything for a long time and that
is the correct-looking result, not a failure.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

# The canonical prop map. Lives here rather than in dashboard.py because
# two definitions of "what the props are" is how they drift -- the same
# argument iter_scoreable makes in score_slate.py.
PROPS = {
    "prob_hr":            ("HR",           "Power"),
    "prob_tb_over_1.5":   ("TB 1.5",       "Power"),
    "prob_tb_over_2.5":   ("TB 2.5",       "Power"),
    "prob_tb_over_3.5":   ("TB 3.5",       "Power"),
    "prob_hit":           ("Hit",          "Contact"),
    "prob_hits_over_1.5": ("Hits 1.5",     "Contact"),
    "prob_hits_over_2.5": ("Hits 2.5",     "Contact"),
    "prob_hrr_over_0.5":  ("H+R+RBI 0.5",  "Combined"),
    "prob_hrr_over_1.5":  ("H+R+RBI 1.5",  "Combined"),
    "prob_hrr_over_2.5":  ("H+R+RBI 2.5",  "Combined"),
    "prob_hrr_over_3.5":  ("H+R+RBI 3.5",  "Combined"),
    "prob_walk":          ("Walk",         "Disc."),
}

# Cut points, set against the real spread of the props rather than picked
# round: 0.58 sits above every prop's median except the two "over 0.5"
# ones, and 0.28 sits below every median except the genuine long shots
# (HR, TB 2.5+, hits 1.5+, H+R+RBI 3.5).
SAFE_MIN, LOTTO_MAX = 0.58, 0.28
# Fewer books than this is one or two shops posting an alternate, not a
# consensus, and a disagreement with it is not an edge.
MIN_BOOKS = 4
# At or below this the market is pricing a short outing, which is exactly
# where the model's compression makes it read high -- measured over 202
# pitcher-nights it moves 0.67 strikeouts for every 1 the market moves.
# A positive edge on the OVER here is the bias, not a find.
LOW_LINE = 4.5
WINDOW_GAP = pd.Timedelta(minutes=30)

# (tier, rank key). SAFE by probability -- you want the surest thing, and
# there is almost nothing to choose between bats up there. MEDIUM and
# LOTTO by lift over the slate median, because down there raw probability
# just re-sorts by which prop has the higher base rate, which is a fact
# about the prop and not about the player.
TIERS = [("SAFE", "p"), ("MEDIUM", "lift"), ("LOTTO", "lift")]
TIER_ORDER = ["MIXED", "SAFE", "MEDIUM", "LOTTO"]


def tier_of(p: float) -> str:
    return "SAFE" if p >= SAFE_MIN else ("LOTTO" if p < LOTTO_MAX else "MEDIUM")


def market_lines(date: str, cache=CACHE) -> dict:
    """{player: (line, market_prob, n_books)} from the captured odds."""
    path = os.path.join(cache, f"odds_{date}.csv")
    if not os.path.exists(path):
        return {}
    try:
        odds = pd.read_csv(path)
    except Exception:
        return {}
    if not {"player", "line", "prob_over"} <= set(odds.columns):
        return {}
    if "market" in odds.columns:
        odds = odds[odds["market"] == "pitcher_strikeouts"]
    odds = odds.dropna(subset=["player", "line", "prob_over"])
    if odds.empty:
        return {}
    # The line closest to even money is the one the book is surest about.
    odds = odds.assign(_d=(odds["prob_over"] - 0.5).abs())
    best = odds.sort_values("_d").groupby("player").first()
    return {str(k): (float(v["line"]), float(v["prob_over"]),
                     int(v["n_books"]) if "n_books" in best.columns
                     and pd.notna(v.get("n_books")) else 0)
            for k, v in best.iterrows()}


def _pmf(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        arr = np.array([float(x) for x in text.split(",") if x.strip()])
    except ValueError:
        return None
    return arr if arr.size and arr.sum() > 0 else None


def _prob_over(pmf, line):
    """P(count > line). Over 5.5 means six or more, so the sum starts at 6."""
    if pmf is None:
        return float("nan")
    start = int(np.ceil(line))
    return float(pmf[start:].sum()) if start < len(pmf) else 0.0


def hitter_legs(frame, props) -> pd.DataFrame:
    """One row per (player, prop) with its probability and its lift."""
    med = {k: float(frame[k].median()) for k in props
           if k in frame.columns and pd.notna(frame[k].median())
           and float(frame[k].median()) > 0}
    parts = []
    for key in props:
        if key not in med:
            continue
        col = pd.to_numeric(frame[key], errors="coerce")
        ok = col.notna() & (col > 0)
        if not ok.any():
            continue
        cols = [c for c in ("player_id", "name", "team", "opponent",
                            "game_pk", "lineup_slot") if c in frame.columns]
        sub = frame.loc[ok, cols].copy()
        sub["p"] = col[ok]
        sub["lift"] = sub["p"] / med[key]
        sub["prop_key"] = key
        sub["prop"] = props[key][0]
        sub["base"] = med[key]
        sub["line"] = float("nan")
        sub["kind"] = "bat"
        sub["edge"] = float("nan")
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def arm_legs(pit, market) -> pd.DataFrame:
    """
    One row per starter with a real captured line, on the side the model
    likes.

    THE SIDE MATTERS. An earlier version always wrote "Over L" and attached
    the signed edge, which put "Tyler Glasnow Over 7.5 K, 29.7%, market
    42%, -12.1%" on a slip -- a leg the model is saying NOT to take,
    printed as a pick. A negative edge on the over is a positive edge on
    the under, and the under is the bet.

    Suspect legs are dropped outright rather than demoted. Demoting was
    tried and still put a +17.9% OVER at a 3.5 line on top of a SAFE slip
    carrying its own warning; a leg the model is measurably wrong about
    does not belong on a slip at any rank.
    """
    if pit is None or pit.empty or "k_dist" not in pit.columns:
        return pd.DataFrame()
    rows = []
    for _, r in pit.iterrows():
        hit = market.get(str(r.get("pitcher")))
        if not hit or hit[2] < MIN_BOOKS:
            continue
        line, mkt_over, books = hit
        p_over = _prob_over(_pmf(r.get("k_dist")), line)
        if not pd.notna(p_over):
            continue
        if p_over >= mkt_over:
            side, p, mq = "Over", p_over, mkt_over
        else:
            side, p, mq = "Under", 1.0 - p_over, 1.0 - mkt_over
        if side == "Over" and line <= LOW_LINE:
            continue                       # the measured bias, not a find
        rows.append({
            "player_id": r.get("pitcher_id"), "name": r.get("pitcher"),
            "team": r.get("team"), "opponent": r.get("opponent"),
            "game_pk": r.get("game_pk"), "lineup_slot": pd.NA,
            "p": p, "lift": float("nan"),
            "prop_key": f"pitcher_k_{side.lower()}",
            "prop": f"{side} {line:.1f} K", "base": mq, "line": line,
            "kind": "arm", "edge": p - mq})
    return pd.DataFrame(rows)


def _pick(pool, arms, by, n):
    """
    One slip: up to `n` legs from a single tier.

    TWO exclusions, and the first matters more than it looks. An earlier
    version only avoided reusing a GAME and produced

        Yandy Diaz  Hit       73.2%
        Yandy Diaz  Hits 1.5  33.1%
        Yandy Diaz  Hits 2.5  10.0%

    which is not a parlay. The props are NESTED -- two hits implies one hit
    implies a hit -- so those legs are one bet wearing three labels, with a
    true probability equal to the smallest of them rather than the product.
    A home run implies H+R+RBI 0.5 the same way. Excluding the player
    outright is the only rule that is safe without hard-coding which props
    imply which.

    Games are avoided second, as a preference: a one-game window has
    nothing to swap in, so a clash is flagged rather than treated as an
    error.

    At most ONE arm, and it goes in first -- it is the only leg with a
    measured edge, so it earns its place rather than winning a ranking it
    cannot be scored on.
    """
    used_g, used_n, out = set(), set(), []
    if arms is not None and not arms.empty:
        a = arms.reindex(arms["edge"].astype(float).abs()
                         .sort_values(ascending=False).index).iloc[0]
        out.append({"row": a, "clash": False})
        used_g.add(a["game_pk"])
        used_n.add(a["name"])
    while len(out) < n:
        avail = pool[~pool["name"].isin(used_n)]
        if avail.empty:
            break
        fresh = avail[~avail["game_pk"].isin(used_g)]
        clash = fresh.empty
        best = (avail if clash else fresh).nlargest(1, by).iloc[0]
        used_g.add(best["game_pk"])
        used_n.add(best["name"])
        out.append({"row": best, "clash": clash})
    return out


def _mixed(group, arms):
    """One leg per tier, plus an arm. The slip a long shot rides along in."""
    used_g, used_n, out = set(), set(), []
    if arms is not None and not arms.empty:
        a = arms.reindex(arms["edge"].astype(float).abs()
                         .sort_values(ascending=False).index).iloc[0]
        out.append({"row": a, "clash": False})
        used_g.add(a["game_pk"])
        used_n.add(a["name"])
    for tname, by in TIERS:
        pool = group[(group["tier"] == tname) & (~group["name"].isin(used_n))]
        if pool.empty:
            continue
        fresh = pool[~pool["game_pk"].isin(used_g)]
        clash = fresh.empty
        best = (pool if clash else fresh).nlargest(1, by).iloc[0]
        used_g.add(best["game_pk"])
        used_n.add(best["name"])
        out.append({"row": best, "clash": clash})
    return out


def build_slips(frame, pit=None, market=None, legs=3, props=None) -> pd.DataFrame:
    """
    Every slip for this slate, one row per LEG.

    `frame` is the slate (needs `start`, `game_pk` and the prob_ columns).
    Returns an empty frame rather than raising when there is nothing to
    build, so a caller can write an empty file and a reader can say so.
    """
    props = props or PROPS
    market = market or {}
    if frame is None or frame.empty or "start" not in frame.columns:
        return pd.DataFrame()
    bd = frame.dropna(subset=["start"]).copy()
    if bd.empty:
        return pd.DataFrame()

    starts = (bd.drop_duplicates("game_pk")[["game_pk", "start"]]
                .sort_values("start"))
    starts["window"] = (starts["start"].diff() > WINDOW_GAP).cumsum()
    win = dict(zip(starts["game_pk"], starts["window"]))

    legs_df = hitter_legs(bd, props)
    if legs_df.empty:
        return pd.DataFrame()
    legs_df["window"] = legs_df["game_pk"].map(win)
    legs_df = legs_df.dropna(subset=["window"])
    legs_df["tier"] = legs_df["p"].map(tier_of)

    arms = arm_legs(pit, market)
    if not arms.empty:
        arms["window"] = arms["game_pk"].map(win)
        arms = arms.dropna(subset=["window"])
        # An arm is tiered by its own probability like everything else --
        # that is what lets it mix into a slip instead of sitting apart.
        arms["tier"] = arms["p"].map(tier_of)

    rows = []
    for w, g in legs_df.groupby("window"):
        wg = bd[bd["game_pk"].isin(g["game_pk"])]
        t0, t1 = wg["start"].min(), wg["start"].max()
        n_games = int(wg["game_pk"].nunique())
        aw = arms[arms["window"] == w] if not arms.empty else pd.DataFrame()
        built = {"MIXED": _mixed(g, aw if not aw.empty else None)}
        for tname, by in TIERS:
            built[tname] = _pick(
                g[g["tier"] == tname],
                aw[aw["tier"] == tname] if not aw.empty else None,
                by, int(legs))
        for tname in TIER_ORDER:
            picked = built.get(tname) or []
            for i, leg in enumerate(picked, start=1):
                r = leg["row"]
                rows.append({
                    "window": int(w),
                    "window_start": t0, "window_end": t1,
                    "n_games": n_games,
                    # In a one-game window every leg is same-game by
                    # definition, so the clash flag says nothing there.
                    "solo": n_games == 1,
                    "tier": tname, "leg_no": i, "n_legs": len(picked),
                    "player_id": r.get("player_id"), "name": r["name"],
                    "team": r.get("team"), "opponent": r.get("opponent"),
                    "game_pk": r.get("game_pk"),
                    "prop_key": r["prop_key"], "prop": r["prop"],
                    "line": r.get("line"), "p": float(r["p"]),
                    "lift": r.get("lift"), "base": r.get("base"),
                    "kind": r["kind"], "edge": r.get("edge"),
                    "clash": bool(leg["clash"]) and n_games > 1,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # The product, which is what a book pays on. Right for legs in
    # different games and TOO LOW when two legs share one -- correlated
    # legs land together more often than independence implies.
    out["slip_p"] = out.groupby(["window", "tier"])["p"].transform("prod")
    return out


def main():
    ap = argparse.ArgumentParser(description="Commit tonight's bet slips.")
    ap.add_argument("date", nargs="?", default=None)
    ap.add_argument("--legs", type=int, default=3,
                    help="legs per pure slip (MIXED is always one per tier)")
    args = ap.parse_args()
    date = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")

    slate_path = os.path.join(CACHE, f"slate_{date}.csv")
    if not os.path.exists(slate_path):
        print(f"No cache/slate_{date}.csv -- run predict_slate first.")
        return 1
    frame = pd.read_csv(slate_path)
    if "start_time_utc" in frame.columns:
        frame["start"] = pd.to_datetime(frame["start_time_utc"], utc=True,
                                        errors="coerce")
    pit = None
    pit_path = os.path.join(CACHE, f"pitchers_{date}.csv")
    if os.path.exists(pit_path):
        try:
            pit = pd.read_csv(pit_path)
        except Exception:
            pit = None

    slips = build_slips(frame, pit, market_lines(date), legs=args.legs)
    if slips.empty:
        print("Nothing to build -- no start times or no probability columns.")
        return 1

    slips = slips.assign(game_date=date, built_at_utc=pd.Timestamp.utcnow()
                         .strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = os.path.join(CACHE, f"slips_{date}.csv")
    slips.to_csv(path, index=False)

    per = slips.drop_duplicates(["window", "tier"])
    print(f"SLIPS {date}")
    print(f"  {per['window'].nunique()} windows, {len(per)} slips, "
          f"{len(slips)} legs -> cache/slips_{date}.csv "
          f"({os.path.getsize(path) / 1024:.0f} KB)")
    print()
    print(f"  {'tier':8} {'slips':>5} {'mean':>8} {'expected hits':>14}")
    for t in TIER_ORDER:
        sub = per[per["tier"] == t]
        if sub.empty:
            continue
        print(f"  {t:8} {len(sub):>5} {sub['slip_p'].mean():>8.2%} "
              f"{sub['slip_p'].sum():>14.2f}")
    print(f"\n  After the games: python score_slips.py {date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
