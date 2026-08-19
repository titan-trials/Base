"""
Slate dashboard.

    python predict_slate.py 2026-08-18
    streamlit run dashboard.py

Reads cache/slate_{date}.csv and shows, per game, every hitter's projected
probabilities for a home run, a hit, a walk, a strikeout, and each
hits+runs+RBI over/under line.

DESIGN NOTES
------------
One series per view, so one hue. Each bar chart shows a single measure
(the prop you selected) across hitters, which makes this a MAGNITUDE
comparison, not a categorical one -- so every bar is the same colour.
Colouring bars darker-where-longer would double-encode length as hue and
spend the only free channel restating what the bar already says.

Lineup status is shown as a text badge, never as colour alone -- a
projected slot and a confirmed one produce visibly different plate
appearance estimates, and that distinction has to survive a greyscale
screenshot.

Probabilities are shown to one decimal. The engine's measured edge over
"just quote the base rate" is a Brier skill of roughly +0.018 to +0.028,
which is real but small; a second decimal would imply a precision the
model does not have.
"""
import glob
import os

import pandas as pd
import streamlit as st
import altair as alt

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Slot 1 of the reference categorical palette, used as the single hue for
# magnitude. Light and dark steps are chosen for their own surfaces.
SERIES_LIGHT = "#2a78d6"
SERIES_DARK = "#3987e5"
INK_SECONDARY = "#52514e"

PROPS = {
    "prob_hr": "Home run",
    "prob_hit": "Hit (over 0.5)",
    "prob_walk": "Walk",
    "prob_k": "Strikeout",
    "prob_hrr_over_0.5": "H+R+RBI over 0.5",
    "prob_hrr_over_1.5": "H+R+RBI over 1.5",
    "prob_hrr_over_2.5": "H+R+RBI over 2.5",
    "prob_hrr_over_3.5": "H+R+RBI over 3.5",
    "prob_tb_over_0.5": "Total bases over 0.5",
    "prob_tb_over_1.5": "Total bases over 1.5",
    "prob_tb_over_2.5": "Total bases over 2.5",
    "prob_tb_over_3.5": "Total bases over 3.5",
    "prob_hits_over_1.5": "Hits over 1.5",
    "prob_hits_over_2.5": "Hits over 2.5",
}


st.set_page_config(page_title="Slate props", layout="wide")


def fmt_clock(ts) -> str:
    """
    '7:10 PM ET', built without strftime.

    strftime's zero-stripping flag is platform-specific: POSIX spells it
    %-I, Windows spells it %#I, and each raises ValueError on the other.
    Since this project is developed on one and run on the other, the
    portable answer is to not use strftime for this at all -- the
    arithmetic is two lines and works everywhere.
    """
    if ts is None or pd.isna(ts):
        return ""
    hour = ts.hour % 12 or 12
    meridiem = "AM" if ts.hour < 12 else "PM"
    return f"{hour}:{ts.minute:02d} {meridiem} ET"


def fmt_datetime(ts) -> str:
    """'Aug 18, 7:10 PM ET' -- same portability reasoning as fmt_clock."""
    if ts is None or pd.isna(ts):
        return "unknown time"
    return f"{ts.strftime('%b')} {ts.day}, {fmt_clock(ts)}"


def available_slates():
    """Not cached: re-running the predictor can create a new date's file,
    and a cached listing would hide it until the app restarted."""
    paths = sorted(glob.glob(os.path.join(CACHE_DIR, "slate_*.csv")), reverse=True)
    return {os.path.basename(p)[len("slate_"):-len(".csv")]: p for p in paths}


@st.cache_data
def load_slate(path, file_mtime):
    """
    `file_mtime` is not used inside the function -- it's part of the CACHE
    KEY, and that is the whole point.

    Streamlit caches on arguments. Keyed on `path` alone, re-running
    predict_slate.py for the SAME date (which is exactly what you do when
    lineups flip from projected to confirmed) writes new numbers to the
    same filename, and the dashboard keeps serving the old ones with no
    indication anything is stale. Adding the modification time to the key
    means a rewritten file is automatically a cache miss.
    """
    df = pd.read_csv(path)
    if "start_time_utc" in df.columns:
        df["start_time"] = pd.to_datetime(
            df["start_time_utc"], errors="coerce", utc=True
        ).dt.tz_convert("America/New_York")
    return df


slates = available_slates()
if not slates:
    st.title("No slate found")
    st.write("Run the predictor first, then reload this page:")
    st.code("python predict_slate.py 2026-08-18", language="bash")
    st.stop()

# --- Controls, in one row above the content ----------------------------
st.title("Player prop probabilities")

col_date, col_prop, col_min = st.columns([1, 1.4, 1])
with col_date:
    slate_date = st.selectbox("Slate", list(slates))
with col_prop:
    prop_col = st.selectbox("Prop", list(PROPS), format_func=lambda c: PROPS[c])
with col_min:
    min_prob = st.slider("Hide below", 0.0, 1.0, 0.0, 0.05,
                         help="Filter out players under this probability.")

slate_path = slates[slate_date]
df = load_slate(slate_path, os.path.getmtime(slate_path))
if prop_col not in df.columns:
    st.error(f"'{PROPS[prop_col]}' is not in this slate file. Re-run the predictor.")
    st.stop()

is_dark = st.get_option("theme.base") == "dark"
series_colour = SERIES_DARK if is_dark else SERIES_LIGHT

# --- Headline ----------------------------------------------------------
shown = df[df[prop_col] >= min_prob]
projected = int((df["lineup_status"] == "projected").sum()) if "lineup_status" in df else 0
confirmed = int((df["lineup_status"] == "confirmed").sum()) if "lineup_status" in df else 0

a, b, c, d = st.columns(4)
a.metric("Games", df["game_pk"].nunique())
b.metric("Hitters", len(df))
c.metric("Confirmed lineups", confirmed)
d.metric("Projected lineups", projected)

if projected and not confirmed:
    st.info(
        "No official lineups posted yet — every slot is projected from recent "
        "starts. Plate-appearance estimates move by up to ~0.9 PA between the "
        "leadoff and ninth slots, so these will sharpen once lineups are out."
    )

written = pd.Timestamp(os.path.getmtime(slate_path), unit="s", tz="UTC") \
    .tz_convert("America/New_York")
st.caption(f"Predictions written {fmt_datetime(written)}. "
           f"Re-run `python predict_slate.py {slate_date}` after lineups post, "
           f"then reload this page.")

st.caption(
    "Calibrated probabilities, not confident calls. On held-out data this engine "
    "beats simply quoting the base rate by a Brier skill of about +0.018 to "
    "+0.028 — a real edge, and a small one."
)

with st.expander("What do these columns mean?"):
    st.markdown("""
**Baseball shorthand**

- **PA — plate appearance.** One complete turn at bat, however it ends:
  hit, out, walk, hit-by-pitch. Not the same as an *at bat* (AB), which
  excludes walks and sacrifices. A starter gets roughly 3.5 to 4.5 PA a
  game depending on where he bats in the order.
- **K — strikeout.** Standard scorekeeping shorthand, in use since the
  1860s.
- **BB — base on balls**, i.e. a walk. Hit-by-pitch is grouped with it
  here, since for betting purposes "reached base without swinging" is one
  category.
- **RBI — runs batted in.** Runs that scored because of your at bat.
- **R — runs scored.** You crossed the plate yourself. Note this usually
  happens two or three hitters AFTER the at bat where you reached base —
  it's the one part of the prop you don't finish on your own.
- **TB — total bases.** Bases gained on hits: a single is 1, a double 2,
  a triple 3, a homer 4. A walk counts **zero** — it puts you on first but
  you didn't earn a base with the bat. That's why total bases and H+R+RBI
  reward different hitters: a patient singles hitter is good at one and
  poor at the other.
- **H+R+RBI** — hits plus runs scored plus RBI, added up across the whole
  game. The standard combined prop. Walks are **not** in it; a walk helps
  only by putting you on base to score later. A three-run homer is 1 hit
  + 3 RBI + your own run = 5.

**Columns**

| Column | Meaning |
|---|---|
| Slot | Where he bats in the order, 1 through 9 |
| Lineup | `confirmed` = official lineup posted. `projected` = inferred from his recent starts |
| Facing | The opposing starting pitcher |
| His PA | How many career plate appearances back **this hitter's** rates. Under ~150 and the number is shrunk most of the way to league average — it's "a typical hitter in this spot", not a read on him |
| Pitcher PA | How many plate appearances back that pitcher's rates. **0 means we have no data on him** and fell back to league average — treat those matchups as batter-only |
| Exp. PA | Expected plate appearances tonight, from lineup slot and home/away |
| Home run / Hit / Walk / Strikeout | Chance of **at least one** in the game |
| Total bases over 1.5 | Chance of **2 or more** total bases. Bases from hits only: single 1, double 2, triple 3, homer 4. A walk is worth **zero** here |
| H+R+RBI over 1.5 | Chance the combined total is **2 or more** (lines are half-numbers so there's no tie) |

**Reading the numbers.** These are probabilities, not predictions. A 30%
home run chance means it happens roughly 3 times in 10 — so *not*
happening is still the normal outcome. The value is in the ordering and in
the number being honest, not in any single call.
""")

# --- Whole-slate leaderboard -------------------------------------------
st.subheader(f"Top of the slate — {PROPS[prop_col]}")

top = shown.nlargest(20, prop_col).copy()
# Build the label defensively. Plain string concatenation propagates NaN:
# if `team` is missing for one row, `name + " (" + team + ")"` makes the
# ENTIRE label NaN, and Altair draws the bar with no name beside it. The
# bar looks fine, the row looks broken, and nothing errors.
top["label"] = (top["name"].fillna("(unknown)").astype(str)
                + "  (" + top["team"].fillna("?").astype(str) + ")")

# Disambiguate any repeated label. Altair collapses identical y-values into
# one row, so two players sharing a name and team would silently merge into
# a single bar.
duplicated = top["label"].duplicated(keep=False)
if duplicated.any():
    top.loc[duplicated, "label"] = (
        top.loc[duplicated, "label"] + " #"
        + (top.loc[duplicated].groupby("label").cumcount() + 1).astype(str)
    )

# Copy the selected measure into a fixed, dot-free column name.
#
# Vega-Lite reads a dot in a field name as NESTED FIELD ACCESS, so
# "prob_hrr_over_1.5" is interpreted as object `prob_hrr_over_1`,
# property `5`. That object doesn't exist, every value resolves to
# undefined, and the bars render with zero length -- silently, with no
# error. It's why the combined props showed labels but no bars while the
# single-word props were fine.
#
# Escaping the dot in every encoding would also work, but is easy to miss
# in one place. Renaming once means the chart never sees the dot at all.
top["value"] = top[prop_col]

# Shared scales and axes, with NO .properties() on it.
#
# `padding` and `height` are properties of a whole view, not of one layer,
# and Altair v6 enforces that: layering two charts that each carry padding
# raises "Objects with 'padding' attribute cannot be used within
# LayerChart". The previous version set them on the bar chart and then
# derived the labels from it with .mark_text(), so BOTH layers inherited
# the padding and the layering failed.
#
# It worked locally and broke on Streamlit Cloud because the two run
# different Altair majors. Building the layers off a bare base and
# applying view properties to the finished LayerChart is correct on both,
# so this isn't a version pin -- it's the shape the API always wanted.
base = alt.Chart(top).encode(
    x=alt.X("value:Q",
            title=PROPS[prop_col],
            axis=alt.Axis(format=".0%", grid=True, gridOpacity=0.15,
                          domain=False, tickSize=0)),
    y=alt.Y("label:N", sort="-x", title=None,
            # labelLimit default is 180px, which clips at roughly 26
            # characters -- "Christian Encarnacion-Strand (BAL)" is 34.
            # Names were being truncated to "Yordan Alvarez (..." while
            # shorter ones rendered fine.
            # labelLimit default is 180px, which clips at roughly 26
            # characters. 220 fits "Christian Encarnacion-Strand (BAL)"
            # without reserving so much width that the plot area gets
            # squeezed off the right edge of the container.
            axis=alt.Axis(domain=False, tickSize=0, labelLimit=220)),
)

bars = base.mark_bar(
    cornerRadiusEnd=4, height=14, color=series_colour
).encode(
    tooltip=[
        alt.Tooltip("name:N", title="Player"),
        alt.Tooltip("team:N", title="Team"),
        alt.Tooltip("opponent:N", title="Opponent"),
        alt.Tooltip("opposing_pitcher:N", title="Facing"),
        alt.Tooltip("lineup_slot:Q", title="Slot"),
        alt.Tooltip("lineup_status:N", title="Lineup"),
        alt.Tooltip("expected_pa:Q", title="Expected PA", format=".2f"),
        alt.Tooltip("value:Q", title=PROPS[prop_col], format=".1%"),
    ],
)

labels = base.mark_text(
    align="left", dx=6, fontSize=11, color=INK_SECONDARY
).encode(text=alt.Text("value:Q", format=".1%"))

# `autosize` is the whole ballgame when the container sets the width.
#
# By default Vega-Lite treats `width` as the PLOT area and then adds the
# y-axis labels and padding OUTSIDE it. Streamlit's use_container_width
# hands Vega the container width as that plot width, so a 220px block of
# player names plus padding pushes the right-hand end of the plot past the
# edge of the container and it is simply cut off. The visible result is a
# chart whose bars all appear to run full width against an axis that stops
# early -- the axis was 0-70%, but only the first 16% of it fit on screen.
#
# "fit" with contains="padding" inverts that: the container width is the
# TOTAL, and axis labels and padding are subtracted from it to size the
# plot. Everything lands inside the box.
#
# The explicit padding dict is gone too. It bought an 8px cosmetic nudge
# and cost two separate bugs -- it is what Altair v6 refused to layer, and
# it is part of what overflowed the container here. Vega's default padding
# is fine.
chart = alt.layer(bars, labels).properties(
    # 26px a row, not 22. At 22 the labels for a 20-row leaderboard don't
    # fit vertically, and Vega silently drops every SECOND one rather than
    # shrinking or erroring -- which reads as "half my players lost their
    # names" and sent us looking for a data bug that wasn't there.
    height=max(220, 26 * len(top)),
    autosize={"type": "fit", "contains": "padding"},
)

st.altair_chart(chart, use_container_width=True)

# --- Per game ----------------------------------------------------------
st.subheader("By game")

display_cols = ["name", "lineup_slot", "lineup_status", "opposing_pitcher",
                "batter_pa_seen", "pitcher_pa_seen", "expected_pa"] + list(PROPS)
display_cols = [c for c in display_cols if c in df.columns]
column_config = {
    "name": st.column_config.TextColumn("Player"),
    "lineup_slot": st.column_config.NumberColumn("Slot", format="%d"),
    "lineup_status": st.column_config.TextColumn("Lineup"),
    "opposing_pitcher": st.column_config.TextColumn("Facing"),
    "batter_pa_seen": st.column_config.NumberColumn(
        "His PA", format="%d",
        help="How many career plate appearances back THIS HITTER's rates. "
             "Under ~150 and his numbers are shrunk most of the way to league "
             "average -- read them as 'a typical hitter in this spot' rather "
             "than as a read on the player."),
    "pitcher_pa_seen": st.column_config.NumberColumn(
        "Pitcher PA", format="%d",
        help="How many plate appearances back the opposing pitcher's rates. "
             "0 means we had no data on him and fell back to league average."),
    "expected_pa": st.column_config.NumberColumn("Exp. PA", format="%.2f"),
    **{
        col: st.column_config.ProgressColumn(
            label, format="%.1f%%", min_value=0.0, max_value=1.0
        )
        for col, label in PROPS.items()
    },
}

# Order games by FIRST PITCH, not by game_pk. game_pk is an internal MLB
# identifier with no relationship to start time -- grouping on it put an
# 8:05 game above a 6:40 one, which is not how anyone reads a slate.
if "start_time" in df.columns:
    game_order = (df.groupby("game_pk")["start_time"].min()
                    .sort_values().index.tolist())
else:
    game_order = df["game_pk"].unique().tolist()

for game_pk in game_order:
    game = df[df["game_pk"] == game_pk]
    away = game[game["is_home"] == 0]["team"].iloc[0] if (game["is_home"] == 0).any() else "?"
    home = game[game["is_home"] == 1]["team"].iloc[0] if (game["is_home"] == 1).any() else "?"
    venue = game["venue_name"].iloc[0] if "venue_name" in game else ""
    when = ""
    if "start_time" in game.columns and pd.notna(game["start_time"].iloc[0]):
        when = fmt_clock(game["start_time"].iloc[0])

    header = f"{away} @ {home}"
    if when:
        header += f"  ·  {when}"
    if venue:
        header += f"  ·  {venue}"

    with st.expander(header, expanded=False):
        table = game.sort_values("lineup_slot", na_position="last")
        st.dataframe(
            table[[c for c in display_cols if c in table.columns]],
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
        )

# --- Table view (accessibility: never colour or bar-length alone) ------
with st.expander("Full table (all players, all props)"):
    numeric = [c for c in PROPS if c in df.columns]
    plain = df.copy()
    for col in numeric:
        plain[col] = (plain[col] * 100).round(1)
    st.dataframe(
        plain[[c for c in ["name", "team", "opponent", "lineup_slot",
                           "lineup_status", "opposing_pitcher", "venue_name",
                           "expected_pa"] + numeric if c in plain.columns]],
        hide_index=True, use_container_width=True,
    )
    st.caption("Probabilities as percentages. Same numbers as the bars above, "
               "for anyone reading without colour or wanting to copy them out.")
