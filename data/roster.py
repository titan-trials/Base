"""
Who is actually going to bat, and where in the order.

TWO SOURCES, USED IN ORDER OF TRUST
-----------------------------------
1. The CONFIRMED lineup from the schedule feed, when it's posted. The
   list arrives in batting order, so position in the list is the slot.

2. A PROJECTED lineup built from each team's recent boxscores: for every
   hitter, the slot he most often occupied in his recent starts. Used when
   the official lineup isn't out yet.

Both come from MLB's own data. Neither is a guess about who is "good" --
projection here means "where has this player actually been batting
lately," which is a strong predictor precisely because managers reuse
lineups.

WHY THE DISTINCTION IS TRACKED, NOT SMOOTHED OVER
-------------------------------------------------
Slot drives plate appearances, and plate appearances drive every
probability this system prints. Measured on the pool: 4.40 PA at leadoff
versus 3.51 at ninth, and 47.9% of leadoff games reach five plate
appearances against 7.7% of ninth-slot games. Guessing a hitter into the
wrong slot moves his numbers materially.

So `lineup_status` rides along on every row and reaches the dashboard. A
projected row is still worth showing -- it's the best available answer in
the morning -- but it should not look identical to a confirmed one.

THE BENCH PROBLEM
-----------------
An active roster carries more hitters than start. Without a confirmed
lineup there's no certainty about which nine, so projection keeps only
players with a recent start and ranks them by how regularly they've
played, taking the top nine. A hitter who has started once in three weeks
is not projected into tonight's lineup.
"""
import json
import urllib.request
import urllib.error

import pandas as pd

ROSTER_URL = (
    "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
    "?rosterType=active&hydrate=person"
)
TIMEOUT_SEC = 30

# Pitchers bat essentially never under the universal designated hitter.
EXCLUDED_POSITIONS = {"P"}

# How far back to look when projecting a lineup. Long enough to see a
# regular's pattern, short enough that a slot change from three weeks ago
# doesn't outvote last night's.
PROJECTION_LOOKBACK_GAMES = 15


def _get(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "baseball_predictor/1.0"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
        return json.load(response)


def get_active_hitters(team_id: int) -> pd.DataFrame:
    """Active-roster position players. Columns: player_id, name, position."""
    try:
        payload = _get(ROSTER_URL.format(team_id=int(team_id)))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=["player_id", "name", "position"])

    rows = []
    for entry in payload.get("roster", []):
        person = entry.get("person") or {}
        position = (entry.get("position") or {}).get("abbreviation")
        if position in EXCLUDED_POSITIONS:
            continue
        if not person.get("id"):
            continue
        rows.append({
            "player_id": int(person["id"]),
            "name": person.get("fullName"),
            "position": position,
        })
    return pd.DataFrame(rows)


def project_lineup(team_id: int, recent_slots: pd.DataFrame,
                   roster: pd.DataFrame) -> pd.DataFrame:
    """
    Best guess at tonight's nine, from recent starts.

    `recent_slots` is expected to carry one row per (game_pk, batter) with
    `lineup_slot`, `is_starter` and `game_date` -- the output of
    data.lineup_slots joined to game dates, filtered to this team's recent
    games by the caller.

    Ranking is by how OFTEN a player started recently, not by how good he
    is. The model has opinions about quality; this function only needs to
    know who the manager keeps writing down.
    """
    if recent_slots.empty or roster.empty:
        return pd.DataFrame(columns=["player_id", "lineup_slot", "starts"])

    starters = recent_slots[recent_slots.get("is_starter", 1) == 1]
    starters = starters[starters["batter"].isin(roster["player_id"])]
    if starters.empty:
        return pd.DataFrame(columns=["player_id", "lineup_slot", "starts"])

    summary = starters.groupby("batter").agg(
        starts=("lineup_slot", "size"),
        # Modal slot, not mean: a hitter who bats 2nd and 8th on alternate
        # days bats 2nd or 8th, never 5th. Averaging would invent a slot he
        # has never occupied and hand it to the PA model.
        lineup_slot=("lineup_slot", lambda s: int(s.mode().iloc[0])),
    ).reset_index().rename(columns={"batter": "player_id"})

    summary = summary.sort_values("starts", ascending=False).head(9)

    # A batting order is a PERMUTATION of 1-9: exactly one hitter per slot.
    # Taking each player's modal slot independently does not guarantee that
    # -- two regulars who have both batted leadoff recently both come back
    # as slot 1, and some slot ends up empty. Measured on a real slate that
    # produced 47 hitters at slot 1 and 17 at slot 6 across 30 teams, where
    # every slot should have had exactly 30.
    #
    # That mattered twice over:
    #   - the extra slot-1 hitters were handed 4.51 expected plate
    #     appearances when some of them bat sixth and get 3.75
    #   - features/rbi_features.py builds a slot -> on-base-rate dict, so
    #     duplicates silently overwrote each other and empty slots fell
    #     back to league average
    #
    # Fix: keep the ORDER implied by the modal slots (with starts as the
    # tie-break, so a regular outranks a fill-in who happened to bat there
    # once) and reassign 1..9 densely over it. Relative position is what
    # the modal slot actually tells us; the exact integer is not.
    summary = summary.sort_values(
        ["lineup_slot", "starts"], ascending=[True, False]
    ).reset_index(drop=True)
    summary["lineup_slot"] = range(1, len(summary) + 1)
    return summary


def build_slate_hitters(slate: pd.DataFrame, slot_history: pd.DataFrame,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Every hitter expected to play on this slate, with slot, opponent and
    context.

    Returns one row per (game, hitter):
        game_pk, player_id, name, team_id, team, opponent, is_home,
        opposing_pitcher_id, opposing_pitcher, venue_id, venue_name,
        lineup_slot, lineup_status
    """
    rows = []

    for game in slate.itertuples():
        for side in ("home", "away"):
            team_id = getattr(game, f"{side}_team_id")
            team = getattr(game, f"{side}_team")
            other = "away" if side == "home" else "home"
            opponent = getattr(game, f"{other}_team")
            pitcher_id = getattr(game, f"{other}_probable_id")
            pitcher_name = getattr(game, f"{other}_probable")
            confirmed = getattr(game, f"{side}_lineup") or []

            roster = get_active_hitters(team_id)
            names = dict(zip(roster["player_id"], roster["name"])) if not roster.empty else {}

            if confirmed:
                # The feed returns the lineup already in batting order.
                entries = [
                    {"player_id": int(pid), "lineup_slot": slot,
                     "lineup_status": "confirmed"}
                    for slot, pid in enumerate(confirmed[:9], start=1)
                ]
            else:
                team_slots = slot_history[slot_history["team_id"] == team_id] \
                    if "team_id" in slot_history.columns else slot_history
                projected = project_lineup(team_id, team_slots, roster)
                if projected.empty:
                    entries = [
                        {"player_id": int(pid), "lineup_slot": None,
                         "lineup_status": "unknown"}
                        for pid in roster["player_id"].head(9)
                    ]
                else:
                    entries = [
                        {"player_id": int(r.player_id),
                         "lineup_slot": int(r.lineup_slot),
                         "lineup_status": "projected"}
                        for r in projected.itertuples()
                    ]

            for entry in entries:
                rows.append({
                    "game_pk": game.game_pk,
                    "team_id": team_id,
                    "team": team,
                    "opponent": opponent,
                    "is_home": int(side == "home"),
                    "opposing_pitcher_id": pitcher_id,
                    "opposing_pitcher": pitcher_name,
                    "venue_id": game.venue_id,
                    "venue_name": game.venue_name,
                    "start_time_utc": game.start_time_utc,
                    "name": names.get(entry["player_id"], str(entry["player_id"])),
                    **entry,
                })

    hitters = pd.DataFrame(rows)
    if verbose and not hitters.empty:
        counts = hitters["lineup_status"].value_counts().to_dict()
        print(f"  {len(hitters)} hitters across the slate: {counts}")
    return hitters
