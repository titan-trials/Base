"""
Build a STRATIFIED player pool -- hitters spanning the full range of
ability, not just nine stars.

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
The pool up to V5 was nine elite sluggers. That choice quietly capped
every model in the project, in a way that's easy to miss:

A model's job is to say "this hitter, tonight, is more likely to do X than
that one." If every hitter in the pool is roughly equally good, there is
almost nothing to say. Measured on the V5 data, the per-PA hit-rate model
scored AUC 0.511 -- essentially nothing -- even though hit rate is the
MOST repeatable skill in the whole project (split-half r = +0.92). The
skill is real and measurable; there was just no spread among those nine
for the model to point at.

The same artifact hit the plate-appearance finding: batter identity
explained only 0.9% of PA variance, because all nine bat first through
fourth. In a pool that includes eighth-place hitters, lineup slot is a
large, obvious effect.

So a wide pool isn't "more data." It's a different question, and a fairer
one.

HOW THE POOL IS CHOSEN
----------------------
Stratified sampling on OPS (on-base plus slugging -- the standard one-number
summary of a hitter). Hitters are ranked, split into equal tiers, and
sampled evenly from each. That deliberately over-samples the extremes
relative to a random draw, which is what you want: the tails are where a
model earns its AUC, and a purely random sample of MLB regulars clusters
in the middle.

Deterministic given a seed, and cached to `cache/player_pool_{season}.csv`,
so the pool is reproducible rather than a one-off.

WHY MLB'S STATS API AND NOT PYBASEBALL
--------------------------------------
Two reasons, one of them a genuine fix:

  1. pybaseball's FanGraphs-backed functions (`batting_stats`,
     `team_batting`) all return 403 -- a known upstream issue, documented
     in this project's CONTEXT.md. `batting_stats_bref` also fails.
  2. More importantly, this endpoint returns the MLBAM player ID
     DIRECTLY. That removes `playerid_lookup` from the pipeline entirely,
     and with it the whole accented-name problem that needed a fuzzy-match
     workaround in data/loader.py. Yordan Alvarez is just id 670541 here.

Anything that deletes a workaround instead of maintaining it is worth
taking.
"""
import os
import json
import time
import urllib.request
import urllib.error

import pandas as pd

from data.cache import cache_path

STATS_URL = (
    "https://statsapi.mlb.com/api/v1/stats"
    "?stats=season&group=hitting&season={season}&sportId=1&limit=1000"
)
TIMEOUT_SEC = 30

# A hitter needs enough plate appearances for his rate stats to mean
# anything at all. 250 PA is roughly a half-season of regular play --
# below that, a .900 OPS is as likely to be luck as ability, and adding
# such a player adds noise to the pool rather than range.
MIN_PA = 250


def fetch_season_hitters(season: int, verbose: bool = True) -> pd.DataFrame:
    """
    Every hitter's season line, with MLBAM ids. One API call.

    Returns columns: player_id, name, team, pa, ops, avg, hr, so, bb, iso.
    """
    request = urllib.request.Request(
        STATS_URL.format(season=season),
        headers={"User-Agent": "baseball_predictor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
        payload = json.load(response)

    stat_blocks = payload.get("stats") or []
    if not stat_blocks:
        raise RuntimeError(
            f"MLB Stats API returned no hitting stats for {season}. "
            f"If the season hasn't started, try the previous one."
        )

    rows = []
    for split in stat_blocks[0].get("splits", []):
        player = split.get("player") or {}
        stat = split.get("stat") or {}
        team = split.get("team") or {}

        def num(key):
            try:
                return float(stat.get(key))
            except (TypeError, ValueError):
                return None

        slg, obp = num("slg"), num("obp")
        avg = num("avg")
        rows.append({
            "player_id": player.get("id"),
            "name": player.get("fullName"),
            "team": team.get("abbreviation") or team.get("name"),
            "pa": num("plateAppearances"),
            "ops": num("ops") if num("ops") is not None
                   else (None if slg is None or obp is None else slg + obp),
            "avg": avg,
            "hr": num("homeRuns"),
            "so": num("strikeOuts"),
            "bb": num("baseOnBalls"),
            # Isolated power = slugging minus average: extra bases per at
            # bat, with singles stripped out. The cleanest single measure
            # of raw power, and what matters for the HR side of the model.
            "iso": None if slg is None or avg is None else slg - avg,
        })

    df = pd.DataFrame(rows).dropna(subset=["player_id", "pa", "ops"])
    df["player_id"] = df["player_id"].astype(int)
    if verbose:
        print(f"  MLB Stats API returned {len(df)} hitters for {season}.")
    return df


def build_stratified_pool(season: int, n_players: int = 60, n_tiers: int = 3,
                          min_pa: int = MIN_PA, rank_by: str = "ops",
                          random_state: int = 7, verbose: bool = True) -> pd.DataFrame:
    """
    Sample `n_players` hitters spread evenly across `n_tiers` ability tiers.

    `rank_by` is "ops" for overall hitting quality (the default -- it's what
    determines lineup slot, and therefore plate appearances) or "iso" to
    stratify on raw power instead, which spreads the pool along the axis
    the home-run model cares about.
    """
    hitters = fetch_season_hitters(season, verbose=verbose)
    eligible = hitters[hitters["pa"] >= min_pa].copy()

    if len(eligible) < n_players:
        raise ValueError(
            f"Only {len(eligible)} hitters cleared {min_pa} PA in {season}, "
            f"but {n_players} were requested. Lower min_pa or n_players."
        )

    eligible = eligible.sort_values(rank_by, ascending=False).reset_index(drop=True)
    eligible["tier"] = pd.qcut(
        eligible[rank_by].rank(method="first"), n_tiers,
        labels=[f"tier{i + 1}" for i in range(n_tiers)],
    )
    # tier1 = lowest rank value = weakest. Relabel so tier names read
    # top-down, which is how anyone will actually think about them.
    tier_names = {f"tier{i + 1}": name for i, name in
                  enumerate(reversed(["elite", "middle", "replacement"][:n_tiers]))} \
        if n_tiers == 3 else None

    per_tier = n_players // n_tiers
    picks = []
    for tier, group in eligible.groupby("tier", observed=True):
        take = min(per_tier, len(group))
        picks.append(group.sample(take, random_state=random_state))
    pool = pd.concat(picks, ignore_index=True)

    if tier_names:
        pool["tier"] = pool["tier"].map(tier_names)

    pool = pool.sort_values(rank_by, ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n  Stratified pool: {len(pool)} hitters, {n_tiers} tiers by {rank_by}")
        summary = pool.groupby("tier", observed=True).agg(
            n=("player_id", "size"),
            ops_min=("ops", "min"), ops_max=("ops", "max"),
            iso_mean=("iso", "mean"), hr_mean=("hr", "mean"),
        )
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
        print("\n  Spread is the point: a model can only rank hitters apart")
        print("  if the pool actually contains hitters who differ.")

    return pool


def get_player_pool(season: int, n_players: int = 60, refresh: bool = False,
                    **kwargs) -> pd.DataFrame:
    """Cached wrapper. Delete cache/player_pool_{season}.csv or pass
    refresh=True to rebuild."""
    key = f"player_pool_{season}_{n_players}"
    path = cache_path(key)
    if os.path.exists(path) and not refresh:
        pool = pd.read_csv(path)
        print(f"  Loaded pool from cache/{key}.csv ({len(pool)} hitters).")
        return pool

    pool = build_stratified_pool(season, n_players=n_players, **kwargs)
    pool.to_csv(path, index=False)
    print(f"  Cached pool to cache/{key}.csv")
    return pool


def load_pool_statcast(pool: pd.DataFrame, start_date: str, end_date: str,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Statcast pulls for every hitter in the pool, by MLBAM id.

    Deliberately does NOT go through data/loader.get_player_id: the pool
    already carries real ids, so there is no name to look up and no accent
    to mishandle. Each player is cached individually, so an interrupted
    run resumes cheaply and adding one player later costs one pull.

    This is the slow step -- roughly one rate-limited request per player
    per season. Expect a first run over 60 players to take a while; every
    run after that reads from disk.
    """
    from pybaseball import statcast_batter
    from data.cache import load_cached, save_cache

    frames, failed = [], []
    for i, row in enumerate(pool.itertuples(), start=1):
        key = f"statcast_id{int(row.player_id)}_{start_date}_{end_date}"
        cached = load_cached(key)
        if cached is not None and not cached.empty:
            frames.append(cached)
            continue

        if verbose:
            print(f"  [{i}/{len(pool)}] pulling {row.name} (id {int(row.player_id)})...")
        try:
            df = statcast_batter(start_date, end_date, int(row.player_id))
        except Exception as e:
            print(f"      failed: {type(e).__name__} -- skipping")
            failed.append(row.name)
            continue

        if df is None or df.empty:
            failed.append(row.name)
            continue
        save_cache(key, df)
        frames.append(df)
        time.sleep(0.3)

    if not frames:
        raise RuntimeError("No Statcast data pulled for any pool player.")
    if failed and verbose:
        print(f"\n  {len(failed)} players returned nothing: {', '.join(failed[:8])}"
              f"{'...' if len(failed) > 8 else ''}")

    combined = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  Combined: {len(combined):,} pitches, "
              f"{combined['batter'].nunique()} batters.")
    return combined
