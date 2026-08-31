"""
Slate dashboard.

    python predict_slate.py 2026-08-19
    streamlit run dashboard.py

Five views: the whole slate ranked, one game at a time, tonight's starting
pitchers, every hitter in a sortable table, and how the model has actually
done.

WHY THIS RENDERS ITS OWN HTML
-----------------------------
Streamlit's native table can't produce the two things this page is built
around -- the ranked hero cards and the banded hitter grid -- so both are
built as HTML strings and injected. That sounds heavier than it is, and it
removed a dependency rather than adding one: the previous version drew its
bar chart in Altair and hit two separate Altair bugs (a `padding` attribute
that could not be layered in v6, then that same padding overflowing the
container so the axis was clipped at 16% of its range). A bar here is a div
with a width. There is no version of Altair that can get that wrong.

This file deliberately imports NOTHING from the project -- only glob, os,
pandas and streamlit. The deployed app therefore needs three packages and
one CSV, not the whole modelling stack.

THE ONE PLACE THAT DOES *NOT* RENDER ITS OWN HTML
-------------------------------------------------
"All hitters" uses Streamlit's native table on purpose. Click-to-sort is
the entire point of that view, and a div cannot sort itself -- reproducing
it would mean rebuilding sorting, column resizing and CSV export by hand.
Everywhere else the layout is the product and the native table cannot
express it; there, the sorting is the product and the layout is ordinary.

COLOUR: FOUR QUARTILE BANDS, AND WHY NOT FIVE
---------------------------------------------
Cells are shaded by which quarter of the slate the hitter falls into FOR
THAT PROP, so a 30% home run reads as strong even though 30% is a small
number in absolute terms. The printed percentage is always the real
probability.

The bands are red / amber / yellow / emerald. Every part of that was
measured rather than chosen:

  - Five bands is not available. A five-step red-to-green ramp puts lime
    beside amber at deltaE 2.2 under protanopia -- indistinguishable -- and
    at 12.6 for NORMAL vision, below the 15 floor.
  - Green is emerald (#34d399), not pure green (#0ca30c). Against red,
    pure green measures deltaE 4.1 under deuteranopia: the BEST and WORST
    bands look identical to roughly 8% of men. Emerald moves that to 19.2
    and still reads unmistakably green.
  - Amber against yellow sits at 14.5, marginally under the floor. That
    one is accepted knowingly: confusing "25-50%" with "50-75%" costs
    almost nothing, and the two fills differ in lightness so the ordering
    survives regardless.

Quartiles rather than thirds because thirds put a 60% chance on an
over-0.5 line into the red band, which reads as "bad" when it is merely
below the middle of the slate.

BACKWARD COMPATIBILITY
----------------------
Slate files written before total bases and hits existed do not carry those
columns. Every prop is filtered against what the loaded file actually has,
so an older slate renders its own props and simply shows fewer of them.
"""
import glob
import os

import pandas as pd
import streamlit as st

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

st.set_page_config(page_title="Slate props", layout="wide",
                   initial_sidebar_state="collapsed")

# Prop -> (label, family). Order here is the order everything renders in.
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
FAMILY_ORDER = ["Power", "Contact", "Combined", "Disc."]
DEFAULT_PROP = "prob_hrr_over_1.5"
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
           6: "6th", 7: "7th", 8: "8th", 9: "9th"}

CSS = """
<style>
:root{
 --bg:#121212; --card:#232322; --card2:#2b2b29; --line:#383734;
 --ink:#fff; --ink2:#c3c2b7; --ink3:#8a8880; --accent:#3987e5;
 --b1:#d03b3b; --b2:#fab219; --b3:#ffed29; --b4:#34d399;
 --b1bg:#2b1414; --b2bg:#31260a; --b3bg:#454011; --b4bg:#103831;
 --b1ink:#ff8a8a; --b2ink:#ffd679; --b3ink:#fff389; --b4ink:#6ee6c4;
 --good:#0ca30c; --warn:#fab219;

 /* ---- FONT SIZES -- change these, nothing else ------------------
    Every size below is used in several rules, so editing one value
    here moves everything that belongs together. Bump all five by the
    same amount to scale the whole page. */
 --fs-grid:16px;        /* the numbers in the game grid, hitter names */
 --fs-grid-head:15px;   /* column headers: HR, H+R+RBI 1.5, Walk      */
 --fs-grid-grp:14px;    /* family headers: POWER, CONTACT, COMBINED   */
 --fs-list:14px;        /* leaderboard names and percentages          */
 --fs-meta:12.5px;      /* secondary text: "bats 1st · 4.79 PA"       */
}
.stApp{background:var(--bg)}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden}
.block-container{padding-top:1.6rem;max-width:1300px}

/* ---- "All hitters" breaks out of the 1300px column -----------------
   Twelve prop columns do not fit in the width the rest of the page is
   tuned for, and the overflow lands on Walk -- a column most people
   would never discover, because a table that scrolls sideways does not
   look like a table that scrolls sideways.

   Only this one tab widens. The Slate and Games views are laid out for
   a comfortable reading measure and stretching them to 1900px would
   pull the leaderboard rows apart for no gain.

   Matched on a marker div rather than :nth-of-type(4), so inserting or
   reordering a tab later cannot silently widen the wrong one. If a
   future Streamlit renames the panel attribute the rule stops matching
   and the tab goes back to 1300px -- visibly plain, not subtly broken.

   The attribute is data-testid="stTabPanel". It is NOT the
   data-baseweb="tab-panel" that older write-ups give; that one silently
   matches nothing, which looks exactly like "the CSS had no effect".

   Only the TABLE breaks out, not the whole panel. Widening the panel
   drags the heading, the search box and the footnotes out with it, and
   they end up starting 120px left of the tab bar directly above them --
   which reads as a layout bug rather than as a wide table. */
div[data-testid="stTabPanel"]:has(.sp-wide)
 div[data-testid="stElementContainer"]:has(div[data-testid="stDataFrame"]){
 width:min(96vw,1900px);
 max-width:none;
 margin-left:calc(50% - min(48vw,950px));
}
.sp-head{display:flex;align-items:center;gap:13px;margin-bottom:6px}
.sp-mark{width:30px;height:30px;border-radius:8px;flex:none;
 background:linear-gradient(140deg,var(--accent),#1c5cab);position:relative}
.sp-mark:after{content:"";position:absolute;inset:9px;border-radius:50%;
 border:2px solid rgba(255,255,255,.85);border-top-color:transparent}
.sp-h1{font-size:18px;font-weight:660;color:var(--ink);letter-spacing:-.01em;margin:0}
.sp-sub{font-size:12px;color:var(--ink3);margin-top:1px}
/* NOTE the selectors below use div, not p. Streamlit styles `.stMarkdown p`
   with a font-size, and that beats a bare `.sp-date` class on specificity
   -- so a <p class="sp-date"> silently renders at 16px no matter what this
   rule asks for, while still picking up the weight and colour. The bug
   looks like "my CSS did nothing" when in fact only one property lost.
   The date IS the identity of this page. Every number on it belongs to
   one specific day, and reading Tuesday's slate believing it is Monday's
   makes all of them wrong in a way nothing else on screen would reveal.
   So it is the headline, not a caption under one. */
.sp-date{font-size:26px;font-weight:680;color:var(--ink);letter-spacing:-.02em;
 margin:0;line-height:1.15}
.sp-when{display:inline-block;padding:3px 10px;border-radius:999px;
 font-size:11px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
 vertical-align:6px;margin-left:11px;border:1px solid}
.sp-when.now{color:var(--b4);border-color:rgba(52,211,153,.55);
 background:rgba(52,211,153,.10)}
.sp-when.past{color:var(--ink3);border-color:var(--line)}
.sp-when.future{color:var(--b2);border-color:rgba(250,178,25,.5);
 background:rgba(250,178,25,.08)}
.sp-stale{margin-top:9px;padding:9px 13px;border-radius:9px;font-size:12.5px;
 border:1px solid var(--line);background:var(--card);color:var(--ink2)}
.sp-status{display:inline-flex;gap:9px;align-items:center;padding:7px 13px;
 border-radius:999px;border:1px solid var(--line);background:var(--card);
 font-size:var(--fs-meta);color:var(--ink)}
.sp-dot{width:8px;height:8px;border-radius:50%;flex:none}
.sp-heroes{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:2px 0 22px}
.sp-hero{background:linear-gradient(150deg,var(--card2),var(--card));
 border:1px solid var(--line);border-radius:13px;padding:16px 17px;
 position:relative;overflow:hidden}
.sp-hero:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;
 background:var(--accent)}
.sp-rk{font-size:10.5px;color:var(--ink3);letter-spacing:.1em;text-transform:uppercase}
.sp-nm{font-size:17px;font-weight:640;margin:5px 0 1px;color:var(--ink)}
.sp-mt{font-size:var(--fs-meta);color:var(--ink3)}
.sp-big{font-size:40px;font-weight:680;letter-spacing:-.03em;margin:11px 0 0;
 font-variant-numeric:tabular-nums;line-height:1;color:var(--ink)}
.sp-rows{display:grid;grid-template-columns:26px 1.1fr 1fr 200px 56px;
 gap:0 15px;align-items:center}
.sp-rows>div{padding:9px 0;border-bottom:1px solid var(--line)}
.sp-r{color:var(--ink3);font-size:var(--fs-meta);text-align:right;font-variant-numeric:tabular-nums}
.sp-pn{font-size:var(--fs-list);color:var(--ink)}
.sp-pn i{color:var(--ink3);font-style:normal;font-size:var(--fs-meta)}
.sp-ctx{color:var(--ink3);font-size:var(--fs-meta)}
/* Form marker. Deliberately quiet -- an outline chip, not a filled badge.
   It is an observation the model does not use, and dressing it like a
   prediction would invite reading it as one. */
.sp-form{display:inline-block;padding:1px 6px;border-radius:5px;font-size:11px;
 font-weight:600;letter-spacing:.02em;border:1px solid;margin-left:6px;
 vertical-align:1px}
.sp-form.hot{color:var(--b4);border-color:rgba(52,211,153,.5)}
.sp-form.cold{color:var(--b1ink);border-color:rgba(208,59,59,.55)}
.sp-track{height:7px;background:var(--line);border-radius:4px}
.sp-fill{height:7px;background:var(--accent);border-radius:4px}
.sp-pv{text-align:right;font-variant-numeric:tabular-nums;font-size:var(--fs-list);color:var(--ink)}
table.sp{border-collapse:separate;border-spacing:0;width:100%;font-size:var(--fs-grid)}
table.sp th{font-weight:560;color:var(--ink3);font-size:var(--fs-grid-head);text-align:center;
 padding:5px 6px;white-space:nowrap}
table.sp th.grp{font-size:var(--fs-grid-grp);letter-spacing:.11em;text-transform:uppercase;
 color:var(--ink2);padding-bottom:3px}
table.sp th.nm,table.sp td.nm{text-align:left;white-space:nowrap;padding-left:2px;
 color:var(--ink)}
table.sp td{padding:3px}
table.sp td.meta{color:var(--ink3);text-align:center;font-variant-numeric:tabular-nums}
table.sp td.cell{text-align:center;font-variant-numeric:tabular-nums;border-radius:6px;
 padding:9px 6px;font-weight:560;border:1px solid transparent}
table.sp .gut{width:18px;padding:0!important;border:none!important;background:none!important}
table.sp tr.team td{padding:18px 0 7px;font-size:var(--fs-grid);color:var(--ink2)}
table.sp tr.team .tn{font-weight:660;font-size:15px;color:var(--ink)}
.pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:11.5px;
 border:1px solid var(--line);color:var(--ink3);margin-left:7px;vertical-align:1px}
.pill.ok{border-color:rgba(12,163,12,.5);color:var(--good)}
.pill.wait{border-color:rgba(250,178,25,.5);color:var(--warn)}
.sp-legend{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap;
 font-size:var(--fs-meta);color:var(--ink3)}
.sp-key{display:inline-flex;gap:6px;align-items:center}
.sp-sw{width:13px;height:13px;border-radius:4px;border:1px solid transparent}
.sp-gh{display:flex;gap:15px;align-items:baseline;flex-wrap:wrap;margin:0 0 6px}
.sp-gh b{font-size:19px;color:var(--ink);letter-spacing:-.01em}
.sp-empty{border:1px dashed var(--line);border-radius:11px;padding:30px;
 text-align:center;color:var(--ink2);font-size:13px}
.sp-kpi{display:flex;gap:32px;flex-wrap:wrap;margin-bottom:18px}
.sp-kpi .v{font-size:27px;font-weight:660;color:var(--ink);display:block;
 font-variant-numeric:tabular-nums}
.sp-kpi .k{font-size:11.5px;color:var(--ink3)}
table.plain{width:100%;border-collapse:collapse;font-size:13px}
table.plain th{text-align:left;color:var(--ink3);font-size:11px;letter-spacing:.06em;
 text-transform:uppercase;padding:0 10px 8px;border-bottom:1px solid var(--line)}
table.plain td{padding:11px 10px;border-bottom:1px solid var(--line);color:var(--ink)}
div[data-testid="stHorizontalBlock"] .stButton>button{
 background:var(--card);border:1px solid var(--line);color:var(--ink2);
 border-radius:9px;padding:8px 11px;font-size:var(--fs-meta);font-weight:520;
 white-space:nowrap;min-height:0}
div[data-testid="stHorizontalBlock"] .stButton>button:hover{
 border-color:var(--ink3);color:var(--ink)}
div[data-testid="stHorizontalBlock"] .stButton>button[kind="primary"]{
 background:var(--card2);border-color:var(--accent);color:var(--ink);font-weight:640}
div[data-testid="stHorizontalBlock"]{gap:7px;margin-bottom:6px}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{color:var(--ink3);font-size:13.5px;font-weight:540}
.stTabs [aria-selected="true"]{color:var(--ink)!important}
</style>
"""

# An unbalanced comment marker in the block above does not fail loudly. The
# browser swallows everything from the stray token to the next `*/` it can
# find, the page still renders, and the only symptom is that some rule "had
# no effect" -- which reads as a selector that did not match and sends you
# looking in entirely the wrong place. Cost of finding that by hand once:
# about forty minutes. Cost of this check: one line.
if CSS.count("/*") != CSS.count("*/"):
    raise ValueError(
        f"dashboard CSS has {CSS.count('/*')} '/*' against "
        f"{CSS.count('*/')} '*/' -- an unbalanced comment will silently "
        f"disable the rules around it.")


def fmt_clock(ts) -> str:
    """
    '7:10 PM', without strftime.

    strftime's zero-stripping flag is platform-specific -- %-I on POSIX,
    %#I on Windows, each raising ValueError on the other. This project is
    written on Windows and deployed on Linux, so the portable answer is to
    not use strftime for this at all.
    """
    if ts is None or pd.isna(ts):
        return ""
    hour = ts.hour % 12 or 12
    return f"{hour}:{ts.minute:02d} {'AM' if ts.hour < 12 else 'PM'}"


def fmt_dt(ts) -> str:
    if ts is None or pd.isna(ts):
        return "unknown time"
    return f"{ts.strftime('%b')} {ts.day}, {fmt_clock(ts)} ET"


def available_slates():
    """Not cached: a fresh predict_slate run creates a new file, and a
    cached listing would hide it until the app restarted."""
    paths = sorted(glob.glob(os.path.join(CACHE_DIR, "slate_*.csv")), reverse=True)
    return {os.path.basename(p)[len("slate_"):-len(".csv")]: p for p in paths}


@st.cache_data(show_spinner=False)
def load_slate(path, mtime):
    """
    `mtime` is unused inside the function and that is the entire point --
    it is part of the CACHE KEY.

    Streamlit caches on arguments. Keyed on path alone, re-running
    predict_slate for the same date (exactly what you do when lineups flip
    from projected to confirmed) writes new numbers to the same filename
    and the dashboard keeps serving the old ones, silently.
    """
    df = pd.read_csv(path)
    if "start_time_utc" in df.columns:
        df["start"] = pd.to_datetime(df["start_time_utc"], utc=True,
                                     errors="coerce").dt.tz_convert("America/New_York")
    return df


def pct(v) -> str:
    return "—" if pd.isna(v) else f"{v * 100:.1f}%"


def band_of(value, cuts) -> int:
    """Which quartile of the slate this value falls into, 1-4."""
    if pd.isna(value):
        return 0
    q1, q2, q3 = cuts
    return 1 if value < q1 else 2 if value < q2 else 3 if value < q3 else 4


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def _css_var(name: str) -> str:
    """
    Read a colour back out of the stylesheet above.

    The native table in the "All hitters" tab draws to a canvas and never
    sees the page's CSS custom properties, so its band colours have to
    arrive as literal hex. Writing that hex out a second time is how the
    grid and the table drift a shade apart six months from now; parsing it
    from the one declaration keeps a single source of truth, and raises
    loudly here at import if a variable is ever renamed.
    """
    marker = f"--{name}:"
    start = CSS.index(marker) + len(marker)
    return CSS[start:CSS.index(";", start)].strip()


# band number -> (background, text), matching the game grid exactly.
BANDS = {b: (_css_var(f"b{b}bg"), _css_var(f"b{b}ink")) for b in (1, 2, 3, 4)}


# ----------------------------------------------------------------- load
html(CSS)
slates = available_slates()
if not slates:
    html('<div class="sp-head"><div class="sp-mark"></div>'
         '<div><p class="sp-h1">Slate props</p></div></div>')
    st.warning("No slate file found. Run the predictor, then reload.")
    st.code("python predict_slate.py", language="bash")
    st.stop()

top_l, top_r = st.columns([3, 2])
with top_l:
    slate_date = st.selectbox("Slate", list(slates), label_visibility="collapsed")
path = slates[slate_date]
df = load_slate(path, os.path.getmtime(path))

# Only the props this particular file actually carries.
props = {k: v for k, v in PROPS.items() if k in df.columns}
if not props:
    st.error("This slate file has no probability columns. Re-run predict_slate.py.")
    st.stop()
cuts = {k: [float(df[k].quantile(q)) for q in (.25, .50, .75)] for k in props}

confirmed = int((df.get("lineup_status") == "confirmed").sum())
projected = int((df.get("lineup_status") == "projected").sum())
written = pd.Timestamp(os.path.getmtime(path), unit="s", tz="UTC") \
            .tz_convert("America/New_York")

with top_r:
    all_in = projected == 0
    html(f'<div style="text-align:right"><span class="sp-status">'
         f'<span class="sp-dot" style="background:'
         f'{"var(--good)" if all_in else "var(--warn)"}"></span>'
         f'{"All lineups confirmed" if all_in else f"{confirmed} confirmed · {projected} projected"}'
         f'<span style="color:var(--ink3)"> · written {fmt_dt(written)}</span>'
         f'</span></div>')

def when_label(date_string):
    """
    (text, css class) for how tonight's slate relates to right now.

    Anchored to US Eastern rather than the viewer's clock, because that is
    the calendar baseball is scheduled on -- a 10pm Eastern game is still
    Monday's game to someone on the west coast at 7pm, and to the
    deployed app running in UTC it would otherwise already be Tuesday.
    """
    try:
        slate = pd.Timestamp(date_string).date()
    except Exception:
        return "", ""
    today = pd.Timestamp.now(tz="America/New_York").date()
    delta = (slate - today).days
    if delta == 0:
        return "Today", "now"
    if delta == -1:
        return "Yesterday", "past"
    if delta == 1:
        return "Tomorrow", "future"
    if delta < 0:
        return f"{-delta} days ago", "past"
    return f"in {delta} days", "future"


_when, _when_class = when_label(slate_date)
# Built by hand rather than with %-d, which is POSIX-only and crashes on
# Windows -- this project has already hit that once, in fmt_clock.
_ts = pd.Timestamp(slate_date)
_pretty = f"{_ts.strftime('%A, %B')} {_ts.day}"
_badge = (f'<span class="sp-when {_when_class}">{_when}</span>'
          if _when else "")

html(f'<div class="sp-head"><div class="sp-mark"></div><div>'
     f'<div class="sp-date">{_pretty}{_badge}</div>'
     f'<div class="sp-sub">{slate_date} · {df["game_pk"].nunique()} games · '
     f'{len(df)} hitters</div></div></div>')

# Looking at an old slate is a legitimate thing to do and a very easy
# thing to do by accident, since the picker defaults to the newest file
# and the newest file is not necessarily today's. Say so plainly rather
# than letting yesterday's probabilities read as tonight's.
if _when_class == "past":
    html(f'<div class="sp-stale">These games have already been played. '
         f'You are looking at what the model said before '
         f'{_when.lower()}\'s slate, not at tonight\'s.</div>')

tab_slate, tab_games, tab_pitch, tab_all, tab_res = st.tabs(
    ["Slate", "Games", "Pitchers", "All hitters", "Results"])


def form_chip(row) -> str:
    """
    Hot / Cold marker, or nothing at all.

    Only the two extremes render. "Normal" and "Unknown" produce an empty
    string rather than a grey chip, because a marker shown on every hitter
    is not a marker -- it is a column, and it would compete with the
    probability that is actually the point of the row.
    """
    state = row.get("form_state")
    if state not in ("Hot", "Cold"):
        return ""
    z = row.get("form_z")
    title = (f"Running {abs(z):.1f} standard deviations "
             f"{'above' if state == 'Hot' else 'below'} his own baseline "
             f"(strikeout, hit and on-base rates over his last 25 and 75 "
             f"plate appearances). Not used by the model."
             if pd.notna(z) else "")
    return (f'<span class="sp-form {state.lower()}" title="{title}">'
            f'{state.upper()}</span>')


def context_line(row) -> str:
    bits = []
    slot = row.get("lineup_slot")
    if pd.notna(slot):
        bits.append(f"bats {ORDINAL.get(int(slot), int(slot))}")
    if pd.notna(row.get("expected_pa")):
        bits.append(f"{row['expected_pa']:.2f} PA")
    if isinstance(row.get("venue_name"), str):
        bits.append(row["venue_name"])
    return " · ".join(bits)


# ---------------------------------------------------------------- slate
with tab_slate:
    left, right = st.columns([3, 1])
    with left:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink)">'
             'Top of the slate</div><div style="color:var(--ink3);font-size:12.5px;'
             'margin-bottom:6px">Every hitter, every game, ranked.</div>')
    with right:
        prop = st.selectbox("Prop", list(props),
                            index=list(props).index(DEFAULT_PROP)
                            if DEFAULT_PROP in props else 0,
                            format_func=lambda c: props[c][0],
                            label_visibility="collapsed")

    top = df.nlargest(20, prop)
    heroes = "".join(
        f'<div class="sp-hero"><div class="sp-rk">#{i} on the slate</div>'
        f'<div class="sp-nm">{r["name"]}{form_chip(r)}</div>'
        f'<div class="sp-mt">{r.get("team","")} vs {r.get("opponent","")}'
        f' · {r.get("opposing_pitcher","")}</div>'
        f'<div class="sp-big">{pct(r[prop])}</div>'
        f'<div class="sp-mt" style="margin-top:2px">{context_line(r)}</div></div>'
        for i, (_, r) in enumerate(top.head(3).iterrows(), start=1))
    html(f'<div class="sp-heroes">{heroes}</div>')

    # Bars run to 100%, not to the leader. Scaling to the maximum makes the
    # best hitter's bar fill the track whatever his number is, so 65% and
    # 25% both look like "the top of the scale".
    rows = "".join(
        f'<div class="sp-r">{i}</div>'
        f'<div class="sp-pn">{r["name"]} <i>{r.get("team","")}</i>'
        f'{form_chip(r)}</div>'
        f'<div class="sp-ctx">{context_line(r)}</div>'
        f'<div class="sp-track"><div class="sp-fill" '
        f'style="width:{min(r[prop] * 100, 100):.1f}%"></div></div>'
        f'<div class="sp-pv">{pct(r[prop])}</div>'
        for i, (_, r) in enumerate(top.iloc[3:].iterrows(), start=4))
    html(f'<div class="sp-rows">{rows}</div>')


# ---------------------------------------------------------------- games
with tab_games:
    games = (df.groupby("game_pk")
               .agg(start=("start", "first"), home=("is_home", "sum"))
               .reset_index().sort_values("start"))

    # Clickable chips rather than a dropdown. A dropdown hides fourteen of
    # the fifteen games behind an interaction; the whole slate laid out at
    # once is the point of this view.
    #
    # Selection is held in session_state and set from an on_click CALLBACK,
    # not from the button's return value. The return value is only True
    # during the rerun the click triggers -- by which point the chips above
    # have already drawn themselves using the OLD selection, so the
    # highlight would lag one click behind. A callback runs before anything
    # re-renders.
    def pick_game(pk):
        st.session_state.sp_game = int(pk)

    game_pks = games.game_pk.tolist()
    if st.session_state.get("sp_game") not in game_pks:
        # Also covers switching to a different slate date, whose games have
        # entirely different ids.
        st.session_state.sp_game = int(game_pks[0])

    PER_ROW = 5
    for start in range(0, len(game_pks), PER_ROW):
        chunk = game_pks[start:start + PER_ROW]
        for col, pk in zip(st.columns(PER_ROW), chunk):
            gg = df[df.game_pk == pk]
            away = gg[gg.is_home == 0]["team"]
            home = gg[gg.is_home == 1]["team"]
            posted = (gg["lineup_status"] == "confirmed").all() \
                if "lineup_status" in gg else False
            dot = ":green[●]" if posted else ":orange[●]"
            label = (f'{dot} {away.iloc[0] if len(away) else "?"} @ '
                     f'{home.iloc[0] if len(home) else "?"}  '
                     f'{fmt_clock(gg["start"].iloc[0])}')
            col.button(label, key=f"sp_g{pk}", use_container_width=True,
                       on_click=pick_game, args=(pk,),
                       type="primary" if pk == st.session_state.sp_game
                       else "secondary")

    chosen = st.session_state.sp_game
    g = df[df.game_pk == chosen]

    families = [f for f in FAMILY_ORDER
                if any(props[k][1] == f for k in props)]
    head_grp = '<tr><th class="grp" colspan="3"></th>'
    head_col = '<tr><th class="nm">Hitter</th><th>Slot</th><th>PA</th>'
    for fam in families:
        cols = [k for k in props if props[k][1] == fam]
        head_grp += f'<th class="gut"></th><th class="grp" colspan="{len(cols)}">{fam}</th>'
        head_col += '<th class="gut"></th>' + "".join(
            f'<th>{props[k][0]}</th>' for k in cols)
    ncols = 3 + len(props) + len(families)

    body = ""
    for is_home in (0, 1):
        side = g[g.is_home == is_home].sort_values("lineup_slot")
        if side.empty:
            continue
        team = side["team"].iloc[0]
        facing = side["opposing_pitcher"].iloc[0]
        ok = (side["lineup_status"] == "confirmed").all() \
            if "lineup_status" in side else False
        body += (f'<tr class="team"><td colspan="{ncols}">'
                 f'<span class="tn">{team}</span> batting vs {facing}'
                 f'<span class="pill {"ok" if ok else "wait"}">'
                 f'{"lineup confirmed" if ok else "projected"}</span></td></tr>')
        for _, r in side.iterrows():
            slot = "" if pd.isna(r.get("lineup_slot")) else int(r["lineup_slot"])
            pa = "" if pd.isna(r.get("expected_pa")) else f'{r["expected_pa"]:.2f}'
            body += (f'<tr><td class="nm">{r["name"]}</td>'
                     f'<td class="meta">{slot}</td><td class="meta">{pa}</td>')
            for fam in families:
                body += '<td class="gut"></td>'
                for k in [c for c in props if props[c][1] == fam]:
                    b = band_of(r[k], cuts[k])
                    style = (f'background:var(--b{b}bg);border-color:var(--b{b});'
                             f'color:var(--b{b}ink)') if b else 'color:var(--ink3)'
                    body += f'<td class="cell" style="{style}">{pct(r[k])}</td>'
            body += "</tr>"

    venue = g["venue_name"].iloc[0] if "venue_name" in g else ""
    away_t = g[g.is_home == 0]["team"]
    home_t = g[g.is_home == 1]["team"]
    html(f'<div class="sp-gh"><b>{away_t.iloc[0] if len(away_t) else "?"} @ '
         f'{home_t.iloc[0] if len(home_t) else "?"}</b>'
         f'<span style="color:var(--ink2)">{fmt_clock(g["start"].iloc[0])} · {venue}</span></div>'
         f'<table class="sp">{head_grp}</tr>{head_col}</tr>{body}</table>')

    keys = "".join(
        f'<span class="sp-key"><span class="sp-sw" style="background:var(--b{i}bg);'
        f'border-color:var(--b{i})"></span>{lab}</span>'
        for i, lab in enumerate(["bottom 25%", "25-50%", "50-75%", "top 25%"], start=1))
    html(f'<div class="sp-legend"><span>Shaded by where the hitter falls across '
         f'the whole slate for that prop:</span>{keys}'
         f'<span>· the number shown is the probability</span></div>')


# ------------------------------------------------------------- pitchers
with tab_pitch:
    html('<div style="font-size:15px;font-weight:640;color:var(--ink)">'
         'Starting pitchers</div><div style="color:var(--ink3);font-size:12.5px;'
         'margin-bottom:14px">Strikeout totals for tonight\'s starters.</div>')
    # The pitcher props live in their own file. A slate row is one hitter,
    # and a starter is not a hitter -- carrying nine pitcher columns on
    # every batter row to describe fourteen pitchers would be worse than a
    # second table.
    pit_path = os.path.join(CACHE_DIR, f"pitchers_{slate_date}.csv")
    pit = None
    if os.path.exists(pit_path):
        try:
            pit = pd.read_csv(pit_path)
        except Exception:
            pit = None

    # The book's line, when compare_market.py has been run for this date.
    #
    # Read from the file that script writes rather than joining the odds
    # feed here. The join needs accent-stripping and suffix handling, and
    # a second copy of that logic living in the dashboard is how the two
    # quietly disagree about who "Ronald Acuna Jr." is. compare_market
    # owns the matching; this just displays the answer.
    #
    # It also keeps this file's no-project-imports rule intact, which is
    # what lets the deployed app run on three packages.
    mkt = None
    mkt_path = os.path.join(CACHE_DIR, f"market_compare_{slate_date}.csv")
    if os.path.exists(mkt_path):
        try:
            mkt = pd.read_csv(mkt_path)[
                ["pitcher", "line", "model_prob", "market_prob", "edge"]]
        except Exception:
            mkt = None

    if pit is not None and not pit.empty and "prob_k_over_5.5" in pit.columns:
        if mkt is not None and not mkt.empty:
            pit = pit.merge(mkt, on="pitcher", how="left")
        pit = pit.sort_values("expected_k", ascending=False)
        # k_cuts, NOT cuts. This is a flat script: Streamlit runs it top to
        # bottom every rerun, so every `with tab_x:` block shares one
        # namespace. Naming this `cuts` overwrote the hitter quartiles
        # defined near the top, and the All hitters tab -- which runs after
        # this one and reads them -- died with KeyError: 'prob_hr'.
        #
        # The tab that broke was not the tab with the bug, which is what
        # makes this shape of error expensive to chase. Anything defined
        # inside a tab block gets a name that could only belong to it.
        k_cuts = {c: [float(pit[c].quantile(q)) for q in (.25, .50, .75)]
                  for c in ("prob_k_over_5.5", "prob_k_over_6.5")}

        has_market = "market_prob" in pit.columns and pit["market_prob"].notna().any()

        def market_cells(r):
            """
            The book's line, the model read AT that line, and the gap.

            Blank when no book quoted this pitcher, which is most of them
            on a slate fetched early. An empty cell is the honest render:
            filling it with the model's 5.5 number beside a market price
            for some other line would be comparing two different questions.
            """
            if not has_market:
                return ''
            line, mp, kp = r.get("line"), r.get("market_prob"), r.get("model_prob")
            if pd.isna(line) or pd.isna(mp) or pd.isna(kp):
                return ('<td colspan="4" style="color:var(--ink3);'
                        'text-align:center;font-size:12px">no line posted</td>')
            edge = kp - mp
            # Green where the model is higher, red lower. This is a
            # DISAGREEMENT, not a recommendation -- on four pitchers the
            # model and the market agreed to within 2.3 points, and a gap
            # is a question about what one of them knows, not a bet.
            colour = ("var(--b4)" if edge > 0.03 else
                      "var(--b1ink)" if edge < -0.03 else "var(--ink2)")
            return (f'<td style="color:var(--ink);text-align:center;'
                    f'font-weight:560">{line:g}</td>'
                    f'<td style="color:var(--ink2);text-align:center">'
                    f'{pct(kp)}</td>'
                    f'<td style="color:var(--ink2);text-align:center">'
                    f'{pct(mp)}</td>'
                    f'<td style="color:{colour};text-align:center;'
                    f'font-weight:560">{edge:+.1%}</td>')

        def kcell(value, col):
            band = band_of(value, k_cuts[col])
            if band == 0:
                return '<td style="color:var(--ink3)">—</td>'
            bg, ink = BANDS[band]
            return (f'<td style="background:{bg};color:{ink};font-weight:560;'
                    f'text-align:center;border-radius:6px">{pct(value)}</td>')

        # Plain dicts, not itertuples: `prob_k_over_5.5` has a dot in it,
        # and itertuples silently renames any column that is not a valid
        # identifier to positional `_7`, `_8`. Reaching for those by
        # position is how the wrong column ends up under the wrong header.
        rows = ""
        for r in pit.to_dict("records"):
            starts = int(r.get("starts_seen") or 0)
            # Under five starts the expected batters faced is mostly the
            # league mean rather than a read on him, and saying so is
            # cheaper than having someone discover it the hard way.
            thin = starts < 5
            rows += (
                f'<tr><td style="font-weight:560">{r["pitcher"]}</td>'
                f'<td style="color:var(--ink2)">{r["team"]}</td>'
                f'<td style="color:var(--ink2)">vs {r["opponent"]}</td>'
                f'<td style="color:{"var(--warn)" if thin else "var(--ink2)"};'
                f'text-align:center">{starts}{"*" if thin else ""}</td>'
                f'<td style="color:var(--ink2);text-align:center">'
                f'{r["expected_bf"]:.1f}</td>'
                f'<td style="color:var(--ink);text-align:center;'
                f'font-weight:560">{r["expected_k"]:.1f}</td>'
                + kcell(r.get("prob_k_over_5.5"), "prob_k_over_5.5")
                + kcell(r.get("prob_k_over_6.5"), "prob_k_over_6.5")
                + market_cells(r)
                + '</tr>')
        market_head = ('<th style="text-align:center">Book line</th>'
                       '<th style="text-align:center">Model</th>'
                       '<th style="text-align:center">Market</th>'
                       '<th style="text-align:center">Edge</th>'
                       if has_market else '')
        html('<table class="plain"><thead><tr><th>Pitcher</th><th>Team</th>'
             '<th>Opponent</th><th style="text-align:center">Starts</th>'
             '<th style="text-align:center">Batters faced</th>'
             '<th style="text-align:center">Expected K</th>'
             '<th style="text-align:center">Over 5.5 K</th>'
             '<th style="text-align:center">Over 6.5 K</th>'
             + market_head +
             '</tr></thead><tbody>' + rows + '</tbody></table>')
        thin_n = int((pit["starts_seen"] < 5).sum()) if "starts_seen" in pit else 0
        note = (f' · {thin_n} marked * have under five starts of history, so '
                f'their batters faced is close to the league average'
                if thin_n else '')
        html(f'<div style="margin-top:10px;font-size:12px;color:var(--ink3);'
             f'line-height:1.55">Colour is which quarter of tonight\'s '
             f'starters he falls into{note}. Batters faced is what drives '
             f'everything else — a pitcher pulled in the fourth cannot '
             f'reach six strikeouts however good his rate is.</div>')

    elif "opposing_pitcher" not in df.columns:
        st.info("This slate file has no pitcher information.")
    else:
        # A slate predicted before the strikeout model existed. Show who is
        # starting, and say plainly why the numbers are missing rather than
        # printing dashes that look like a failure.
        legacy = (df.groupby("opposing_pitcher")
                    .agg(faces=("team", "first"), team=("opponent", "first"),
                         pa_seen=("pitcher_pa_seen", "first"))
                    .reset_index())
        rows = "".join(
            f'<tr><td style="font-weight:560">{r.opposing_pitcher}</td>'
            f'<td style="color:var(--ink2)">{r.team}</td>'
            f'<td style="color:var(--ink2)">vs {r.faces}</td>'
            f'<td style="color:{"var(--ink2)" if r.pa_seen else "var(--warn)"}">'
            f'{f"{int(r.pa_seen):,}" if r.pa_seen else "no data"}</td></tr>'
            for r in legacy.itertuples())
        html('<table class="plain"><thead><tr><th>Pitcher</th><th>Team</th>'
             '<th>Opponent</th><th>PA of history</th>'
             '</tr></thead><tbody>' + rows + '</tbody></table>')
        html('<div style="margin-top:10px;font-size:12px;color:var(--ink3)">'
             'This slate was predicted before the strikeout model existed. '
             'Re-run the predictor for this date to fill in the numbers.</div>')


# ---------------------------------------------------------- all hitters
BASE_LABELS = {"name": "Hitter", "team": "Team", "opponent": "Opp",
               "lineup_slot": "Slot", "expected_pa": "PA",
               # Sortable here, which the chips on the Slate tab are not --
               # this is the view for "show me everyone running hot".
               "form_z": "Form"}

# Hover text for the prop columns. "TB 1.5" is unreadable to anyone who has
# not been staring at this for a week, and the header itself has no room to
# say more -- that is the whole reason the columns fit now.
PROP_HELP = {
    "prob_hr":            "Chance he hits at least one home run",
    "prob_tb_over_1.5":   "Chance of 2+ total bases — a double, or two singles",
    "prob_tb_over_2.5":   "Chance of 3+ total bases",
    "prob_tb_over_3.5":   "Chance of 4+ total bases",
    "prob_hit":           "Chance of at least one hit",
    "prob_hits_over_1.5": "Chance of 2+ hits",
    "prob_hits_over_2.5": "Chance of 3+ hits",
    "prob_hrr_over_0.5":  "Chance of at least one hit, run scored or RBI. "
                          "Walks do not count.",
    "prob_hrr_over_1.5":  "Chance of 2+ hits, runs and RBI combined",
    "prob_hrr_over_2.5":  "Chance of 3+ hits, runs and RBI combined",
    "prob_hrr_over_3.5":  "Chance of 4+ hits, runs and RBI combined",
    "prob_walk":          "Chance he draws at least one walk",
}

with tab_all:
    # Marker the CSS above keys on to widen this panel and only this one.
    html('<div class="sp-wide"></div>'
         '<div style="font-size:15px;font-weight:640;color:var(--ink)">'
         'All hitters</div><div style="color:var(--ink3);font-size:12.5px;'
         'margin-bottom:8px">Every hitter on the slate, every prop. '
         'Hover a column header and use its ⋮ menu to sort, pin or hide it.'
         '</div>')

    query = st.text_input("Search", "", label_visibility="collapsed",
                          placeholder="Filter by hitter, team or opponent")

    base = [c for c in BASE_LABELS if c in df.columns]
    view = df[base + list(props)].copy()

    if query.strip():
        # Plain substring, regex=False. A hitter's name is not a pattern,
        # and someone typing "O'Neill" or "Jr." should not get a regex
        # error for their trouble.
        needle = query.strip().lower()
        hit = pd.Series(False, index=view.index)
        for col in ("name", "team", "opponent"):
            if col in view.columns:
                hit |= (view[col].astype(str).str.lower()
                        .str.contains(needle, regex=False, na=False))
        view = view[hit]

    label_of = {**{c: BASE_LABELS[c] for c in base},
                **{k: props[k][0] for k in props}}
    view = view.rename(columns=label_of)
    prop_labels = [props[k][0] for k in props]
    # Bands are computed against the WHOLE slate, not against whatever the
    # search box left behind. Filtering to one team and having its best
    # hitter turn green would say "good for a Dodger", which is not the
    # question anyone is asking.
    cuts_of = {props[k][0]: cuts[k] for k in props}

    default_label = (props[DEFAULT_PROP][0] if DEFAULT_PROP in props
                     else prop_labels[0])
    view = view.sort_values(default_label, ascending=False)

    if view.empty:
        html('<div class="sp-empty">No hitter matches that.</div>')
    else:
        def band_fill(series, column_cuts):
            out = []
            for value in series:
                band = band_of(value, column_cuts)
                if band == 0:
                    out.append("")
                else:
                    bg, ink = BANDS[band]
                    out.append(f"background-color:{bg};color:{ink}")
            return out

        styled = view.style
        for lab in prop_labels:
            styled = styled.apply(band_fill, column_cuts=cuts_of[lab],
                                  subset=[lab])
        fmt = {lab: "{:.1%}" for lab in prop_labels}
        if "PA" in view.columns:
            fmt["PA"] = "{:.2f}"
        if "Slot" in view.columns:
            fmt["Slot"] = "{:.0f}"
        if "Form" in view.columns:
            # Signed, because the sign is the whole message.
            fmt["Form"] = "{:+.1f}"
        styled = styled.format(fmt, na_rep="—")

        # Explicit pixel widths on the five identity columns. Left to size
        # themselves they take 75px each -- a 75px column for a one-digit
        # batting slot -- and those wasted pixels are exactly what pushed
        # Walk off the right edge on a 1440px screen.
        config = {
            # Pinned, so the name stays put if the table does end up
            # scrolling on a narrow window.
            "Hitter": st.column_config.Column(pinned=True, width=170),
            "Team": st.column_config.Column(width=58),
            "Opp": st.column_config.Column(width=58,
                                           help="Team he is facing tonight"),
            "Slot": st.column_config.Column(
                width=48, help="Where he bats in the order, 1 through 9"),
            "PA": st.column_config.Column(
                width=58,
                help="Plate appearances the model expects him to get. "
                     "Everything else scales off this -- a leadoff hitter "
                     "gets roughly one more trip than the number nine."),
            "Form": st.column_config.Column(
                width=58,
                help="How far he is running from his OWN baseline, in "
                     "standard deviations, over his last 25 and 75 plate "
                     "appearances. Positive is hot. Built on strikeout, "
                     "hit and on-base rates. Beyond ±2.0 he is flagged. "
                     "The model does not use this -- it is being tracked "
                     "to find out whether it predicts anything."),
        }
        for key, (lab, _fam) in props.items():
            if lab in view.columns:
                config[lab] = st.column_config.Column(help=PROP_HELP.get(key))

        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=620, column_config=config)

        shown = len(view)
        html(f'<div style="margin-top:8px;font-size:12px;color:var(--ink3)">'
             f'{shown} of {len(df)} hitters'
             f'{" matching your search" if query.strip() else ""} · '
             f'colour is which quarter of the whole slate the hitter falls '
             f'into for that prop · hover the table for a toolbar with '
             f'search, fullscreen and CSV download</div>'
             f'<div style="margin-top:4px;font-size:12px;color:var(--ink3)">'
             f'Hover any column header for what it means. Dragging a header '
             f'moves that column — reload the page to put them back in '
             f'order.</div>')


# -------------------------------------------------------------- results
with tab_res:
    html('<div style="font-size:15px;font-weight:640;color:var(--ink);'
         'margin-bottom:14px">How the model has done</div>')
    log_path = os.path.join(CACHE_DIR, "scoring_log.csv")
    if not os.path.exists(log_path):
        html('<div class="sp-kpi">'
             '<div><span class="v">—</span><span class="k">Slates scored</span></div>'
             '<div><span class="v">—</span><span class="k">Hitters graded</span></div>'
             '<div><span class="v">—</span><span class="k">Model said vs actually happened</span></div>'
             '</div><div class="sp-empty">No slates scored yet.<br>'
             '<span style="color:var(--ink3);font-size:12px">Fills in once games '
             'have been played and graded.</span></div>')
    else:
        raw = pd.read_csv(log_path)

        # Read the CLEAN columns -- hitters whose prediction was written
        # before their own game started. Lineups get posted at different
        # times, so a night usually involves re-running the predictor, and
        # rows for games already underway are graded against outcomes the
        # rolling rates may already contain. Those rows stay in the log and
        # are still there to look at; they just do not set this number.
        #
        # Rows written before score_slate recorded the distinction have no
        # clean_* columns. Fall back to their plain values rather than
        # dropping the slate -- the distinction did not exist to record.
        log = raw.copy()
        for col in ("n", "base_rate", "mean_pred", "brier"):
            clean_col = f"clean_{col}"
            if clean_col in log.columns:
                log[col] = log[clean_col].fillna(log[col])
        log = log[log["n"].fillna(0) > 0]

        # If nothing anywhere was written pre-game, show the full-slate
        # numbers rather than an empty page -- but say plainly what they
        # are. st.stop() would have been the obvious move here and is the
        # wrong one: it halts the whole script, not just this tab.
        full_graded = int(raw.groupby("game_date")["n"].max().sum())

        showing_tainted = log.empty
        if showing_tainted:
            log = raw.copy()
            html('<div style="border:1px solid var(--warn);border-radius:9px;'
                 'padding:11px 13px;margin-bottom:14px;font-size:12.5px;'
                 'color:var(--ink2)"><b style="color:var(--warn)">'
                 'Not out-of-sample.</b> No scored slate has predictions '
                 'written before first pitch, so the numbers below are '
                 'measuring hindsight, not forecasting. Run the predictor '
                 'before games start.</div>')

        n_slates = log["game_date"].nunique()

        # Every prop is graded against the SAME hitters, so summing `n`
        # across the seven rows of a slate counts each hitter seven times.
        # One slate of 270 would read "1,890 hitters graded", which is
        # flattering and false. Take the per-slate figure, then add slates.
        graded = int(log.groupby("game_date")["n"].max().sum())

        # Pool each prop across slates, weighting by how many hitters that
        # slate contributed. Averaging the rates unweighted would let a
        # rained-out four-game night count as much as a full slate.
        def pooled(g):
            w = g["n"].sum()
            said = (g["mean_pred"] * g["n"]).sum() / w
            did = (g["base_rate"] * g["n"]).sum() / w
            brier = (g["brier"] * g["n"]).sum() / w
            # Rebuild skill from the pooled numbers rather than averaging
            # the per-slate skills: the reference variance is p(1-p) at the
            # POOLED base rate, and an average of ratios is not the ratio
            # of the averages.
            ref = did * (1.0 - did)
            skill = 1.0 - brier / ref if ref > 0 else float("nan")
            # How many standard errors the miss is worth. A 2-point gap on
            # 270 hitters is noise; the same gap on 5,000 is a bias.
            se = (did * (1.0 - did) / w) ** 0.5 if w else float("nan")
            sigma = (said - did) / se if se else float("nan")
            return pd.Series({"n": w, "said": said, "did": did,
                              "skill": skill, "sigma": sigma})

        # Column list is explicit: `groupby(...).apply()` over the whole
        # frame warns (and in newer pandas will error) about operating on
        # the grouping column itself.
        by_prop = (log.groupby("label", sort=False)
                      [["n", "mean_pred", "base_rate", "brier"]]
                      .apply(pooled).reset_index())

        # The headline is the edge, not the average probability: averaging
        # "11% of hitters homer" against "61% get a hit" produces a number
        # that describes nothing.
        head_skill = float((by_prop["skill"] * by_prop["n"]).sum()
                           / by_prop["n"].sum())
        worst = float(by_prop["sigma"].abs().max())

        html(f'<div class="sp-kpi">'
             f'<div><span class="v">{n_slates}</span>'
             f'<span class="k">Slates scored</span></div>'
             f'<div><span class="v">{graded:,}</span>'
             f'<span class="k">'
             f'{"Hitters graded" if graded >= full_graded else "Clean hitters graded"}'
             f'</span></div>'
             f'<div><span class="v">{head_skill * 100:+.2f}%</span>'
             f'<span class="k">Edge over guessing the base rate</span></div>'
             f'<div><span class="v">{worst:.1f}σ</span>'
             f'<span class="k">Largest calibration miss</span></div></div>')

        def skill_color(s):
            if pd.isna(s):
                return "var(--ink3)"
            return "var(--b4)" if s > 0.005 else \
                   "var(--b3)" if s > 0 else "var(--b1)"

        rows = "".join(
            f'<tr><td style="font-weight:560">{r.label}</td>'
            f'<td style="color:var(--ink2)">{int(r.n):,}</td>'
            f'<td>{pct(r.said)}</td>'
            f'<td>{pct(r.did)}</td>'
            f'<td style="color:{"var(--ink2)" if abs(r.sigma) < 2 else "var(--warn)"}">'
            f'{r.sigma:+.1f}σ</td>'
            f'<td style="color:{skill_color(r.skill)};font-weight:560">'
            f'{r.skill * 100:+.2f}%</td></tr>'
            for r in by_prop.itertuples())
        html('<table class="plain"><thead><tr><th>Prop</th><th>Graded</th>'
             '<th>Model said</th><th>Actually happened</th>'
             '<th>Miss</th><th>Edge</th></tr></thead><tbody>'
             + rows + '</tbody></table>')

        html('<div style="margin-top:14px;font-size:12px;color:var(--ink3);'
             'line-height:1.55">'
             '<b style="color:var(--ink2)">Edge</b> is how much of the '
             'guesswork the model removes versus just quoting the league '
             'rate for everyone. Positive is good; anything above about '
             '+1% is real skill. '
             '<b style="color:var(--ink2)">Miss</b> is the gap between what '
             'the model expected and what happened, measured in standard '
             'errors — under 2σ is the sample being small, over 2σ is the '
             'model being wrong.</div>')

        if not showing_tainted and graded < full_graded:
            html(f'<div style="margin-top:8px;font-size:12px;color:var(--ink3);'
                 f'line-height:1.55">Counts only the {graded:,} hitters whose '
                 f'prediction was written before their own game started. The '
                 f'other {full_graded - graded:,} were predicted after first '
                 f'pitch — usually a re-run to pick up late lineups — and are '
                 f'kept in the log but left out of these numbers.</div>')

        if n_slates < 10:
            html(f'<div style="margin-top:12px;font-size:12px;'
                 f'color:var(--ink3)">Based on {n_slates} '
                 f'slate{"" if n_slates == 1 else "s"}. Roughly ten are '
                 f'needed before these numbers stop moving around.</div>')
