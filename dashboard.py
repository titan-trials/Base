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

import numpy as np
import pandas as pd
import streamlit as st
# The documented entry point for real JavaScript. `st.components` is not a
# guaranteed attribute of the streamlit module -- it only exists once this
# submodule has been imported somewhere -- so it is imported explicitly
# rather than reached through `st.`, which would work on one version and
# raise AttributeError on another.
import streamlit.components.v1 as components

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
/* Google Fonts by @import rather than a <link> tag: Streamlit's markdown
   renderer does not reliably pass a bare <link> through, and a silently
   dropped font falls back without saying so. Fraunces carries the date and
   the display figures; Archivo does everything operational. */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Archivo:wght@400;500;600;700&display=swap');
:root{
 --bg:#121212; --card:#232322; --card2:#2b2b29; --line:#383734;
 --ink:#fff; --ink2:#c3c2b7; --ink3:#8a8880;
 /* Warmed from #3987e5. The neutrals above lean warm on purpose and the
    old blue was the one colour on the page arguing with them. Still
    nowhere near the b1-b4 band scale, which has to stay semantic. */
 --accent:#7d93d8;
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
/* ---- game cards (Game Lines) ------------------------------------
   Two per row on a wide screen, one on a narrow one. auto-fill with a
   minimum rather than a fixed repeat(2), so the cards reflow instead of
   squeezing when the window is small. */
.sp-gl{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
 gap:14px;margin:2px 0 20px}
.sp-gc{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:14px 16px 13px}
.sp-gc .hd{display:flex;justify-content:space-between;align-items:baseline;
 font-size:11px;color:var(--ink3);letter-spacing:.06em;text-transform:uppercase;
 margin-bottom:9px}
.sp-gc .sd{display:flex;justify-content:space-between;align-items:flex-end}
.sp-gc .tm{font-size:16.5px;font-weight:660;color:var(--ink)}
.sp-gc .rn{font-size:25px;font-weight:680;color:var(--ink);
 font-variant-numeric:tabular-nums;line-height:1.05}
.sp-gc .rn.dim{color:var(--ink2)}
/* Win probability as one split bar rather than two numbers. The whole
   point of "LAD 61 COL 39" is that the two are one quantity. */
.sp-bar{display:flex;height:8px;border-radius:5px;overflow:hidden;
 background:var(--line);margin:10px 0 5px}
.sp-bar i{display:block;height:100%}
.sp-bar .a{background:var(--accent)}
.sp-bar .h{background:var(--b4)}
.sp-wp{display:flex;justify-content:space-between;font-size:12.5px;
 color:var(--ink2);font-variant-numeric:tabular-nums}
.sp-wp b{color:var(--ink);font-weight:620}
.sp-sc{margin-top:11px;padding-top:10px;border-top:1px solid var(--line);
 font-size:var(--fs-meta);line-height:1.75;color:var(--ink3)}
.sp-sc .who{color:var(--ink2)}
.sp-sc .who b{color:var(--ink);font-weight:560}
.sp-thin{color:var(--b2ink);font-size:11.5px;margin-top:7px}

/* The team block inside a game's drill-down.
   Deliberately NOT the old .sp-gc card: that card led with a win-
   probability bar, and the win probability was measured to have no
   skill (home-win Brier 0.237-0.265 against 0.25 for a coin flip,
   because the split only ever spanned 45-57%). Same numbers behind it,
   but laid out so the thing in the biggest type is the thing that
   holds up -- projected runs -- and there is no winner anywhere. */
.sp-tb{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:14px 16px 13px;margin:6px 0 16px}
.sp-tb .sd{display:flex;align-items:flex-end;gap:18px;flex-wrap:wrap}
.sp-tb .sd>div{display:flex;align-items:baseline;gap:8px}
.sp-tb .tm{font-size:14.5px;font-weight:640;color:var(--ink2);
 letter-spacing:.01em}
.sp-tb .rn{font-family:"Fraunces",Georgia,serif;font-size:27px;
 font-weight:600;color:var(--ink);line-height:1}
.sp-tb .at{font-size:11.5px;color:var(--ink3);font-style:italic;
 padding-bottom:3px}
.sp-tb .tot{margin-left:auto;font-size:12px;color:var(--ink3);
 padding-bottom:4px}
.sp-tb .tot b{font-family:"Fraunces",Georgia,serif;font-size:15px;
 font-weight:600;color:var(--ink2)}
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

/* ==================================================== slate page v2 ===
   The front page used to be a ranked list of twenty hitters, which
   answers "who is highest" and looks identical every night. It now opens
   with what is true about TONIGHT and puts the fifteen games -- which the
   old page never mentioned once -- at the centre. */
.sp-display{font-family:"Fraunces","Iowan Old Style",Georgia,serif;
 font-optical-sizing:auto;font-weight:600;letter-spacing:-.02em;color:var(--ink)}
.sp-summary{margin:9px 0 0;font-size:15px;color:var(--ink2);max-width:62ch;
 line-height:1.55}
.sp-summary b{color:var(--ink);font-weight:600}
.sp-health{font-size:12px;color:var(--ink2);line-height:1.6;text-align:right}
.sp-health .k{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--ink3);font-weight:600;display:block;margin-bottom:4px}
.sp-health b{color:var(--ink);font-variant-numeric:tabular-nums}

/* Findings. Drawn from a pool of checks -- see slate_findings(). */
.sp-read{display:grid;grid-template-columns:repeat(auto-fit,minmax(252px,1fr));
 gap:12px;margin:20px 0 30px}
.sp-find{background:var(--card);border:1px solid var(--line);border-radius:13px;
 padding:14px 16px 15px;display:flex;flex-direction:column}
.sp-find .kind{font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;
 font-weight:700}
.sp-find .fig{font-family:"Fraunces",Georgia,serif;font-size:31px;font-weight:600;
 letter-spacing:-.03em;color:var(--ink);line-height:1.05;margin:8px 0 1px;
 font-variant-numeric:tabular-nums}
/* Every figure says what it counts. Without this the reader infers the unit
   from the sentence, and a bare "18.0%" beside a pitcher's name reads as
   easily as his chance of something as it does his rate. */
.sp-find .unit{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
 color:var(--ink3);font-weight:600;margin-bottom:7px}
.sp-find .txt{font-size:12.5px;color:var(--ink3);line-height:1.55}
.sp-find .txt b{color:var(--ink2);font-weight:600}

/* The games list. Each row is a link that sets ?game=<pk>, which is how a
   drill-down is done in Streamlit without fighting the tab model -- there
   is no API to switch tabs, so the detail opens in place instead. */
.sp-games{border-top:1px solid var(--line);margin-top:4px}
a.sp-g{display:grid;grid-template-columns:64px 116px minmax(0,1.5fr) minmax(0,1.15fr) 62px;
 gap:0 16px;align-items:center;padding:10px 2px;border-bottom:1px solid var(--line);
 text-decoration:none;color:inherit;transition:background .12s ease}
a.sp-g:hover{background:rgba(255,255,255,.03)}
/* The same row minus its first cell, for the button-driven version: the
   time is a real st.button in its own column so opening a game reruns
   over the websocket instead of navigating. A <a href="?game="> is a
   genuine browser navigation -- Streamlit does not intercept it -- which
   is the page reload Nolan saw. Everything right of the time stays HTML
   so the columns keep lining up across rows. */
.sp-g2{display:grid;
 grid-template-columns:116px minmax(0,1.5fr) minmax(0,1.15fr) 62px;
 gap:0 16px;align-items:center;min-height:38px;color:inherit;
 /* The separator the old <a> row carried. Without it fifteen rows of
    numbers run together, which is most of why the list read as a table
    rather than a paragraph. */
 border-bottom:1px solid var(--line)}
.sp-g2.on{box-shadow:inset 2px 0 0 var(--accent);
 background:rgba(125,147,216,.09);margin-left:-8px;padding-left:8px}
.sp-g2 .m{font-size:15px;color:var(--ink);white-space:nowrap}
.sp-g2 .m i{color:var(--ink3);font-size:12.5px;font-style:normal;margin:0 4px}
.sp-g2 .p{font-size:12.5px;color:var(--ink3);min-width:0;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.sp-g2 .p b{color:var(--ink2);font-weight:500}
.sp-g2 .b{font-size:12.5px;color:var(--ink2);min-width:0;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.sp-g2 .b i{color:var(--ink3);font-style:normal}
.sp-g2 .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);
 font-size:14px;font-weight:500}
@media (max-width:900px){
 .sp-g2{grid-template-columns:minmax(0,1fr) 56px;row-gap:3px}
 .sp-g2 .p{grid-column:1 / -1;order:4}
 .sp-g2 .b{grid-column:1 / -1;order:5}
}
a.sp-g.on{background:rgba(125,147,216,.09);box-shadow:inset 2px 0 0 var(--accent)}
a.sp-g .t{font-size:12.5px;color:var(--ink3);font-variant-numeric:tabular-nums}
a.sp-g .m{font-size:15px;color:var(--ink);white-space:nowrap}
a.sp-g .m i{color:var(--ink3);font-size:12.5px;font-style:normal;margin:0 4px}
a.sp-g .p{font-size:12.5px;color:var(--ink3);min-width:0;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
a.sp-g .p b{color:var(--ink2);font-weight:500}
a.sp-g .b{font-size:12.5px;color:var(--ink2);min-width:0;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
a.sp-g .b i{color:var(--ink3);font-style:normal}
a.sp-g .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);
 font-size:14px;font-weight:500}
.sp-dot{display:inline-block;width:5px;height:5px;border-radius:50%;margin-right:7px;
 vertical-align:1.5px}
.sp-dot.ok{background:var(--b4)}
.sp-dot.wait{background:var(--b2)}
@media (max-width:900px){
 a.sp-g{grid-template-columns:58px minmax(0,1fr) 56px;row-gap:3px}
 a.sp-g .p{grid-column:2 / -1;order:4}
 a.sp-g .b{grid-column:2 / -1;order:5}
}

/* Leaderboard, demoted to a panel. The median tick was a 1px hairline at
   80% opacity on a 6px bar -- invisible exactly where it matters. */
.sp-lb{border-top:1px solid var(--line)}
.sp-lb .r{display:grid;grid-template-columns:24px minmax(0,1fr) 128px 54px;
 gap:0 12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}
.sp-lb .n{color:var(--ink3);font-size:12.5px;text-align:right;
 font-variant-numeric:tabular-nums}
.sp-lb .w{color:var(--ink);font-size:13.5px;min-width:0;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.sp-lb .w i{color:var(--ink3);font-style:normal;font-size:12.5px;margin-left:5px}
.sp-lb .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);
 font-size:13.5px}
.sp-tw{position:relative}
.sp-med{position:absolute;top:-6px;bottom:-6px;width:2px;background:var(--ink);
 border-radius:1px;box-shadow:0 0 0 2px var(--bg)}
.sp-watch{border-top:1px solid var(--line)}
.sp-w{padding:10px 0;border-bottom:1px solid var(--line)}
.sp-w .t{font-size:13.5px;color:var(--ink);font-weight:500}
.sp-w .d{font-size:12.5px;color:var(--ink3);margin-top:2px;line-height:1.5}
.sp-w .d b{color:var(--ink2);font-weight:600;font-variant-numeric:tabular-nums}
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


def _int_or(value, default: int = 0) -> int:
    """
    int(), but NaN-safe -- which `int(x or 0)` is NOT.

    NaN is truthy in Python, so `float("nan") or 0` evaluates to NaN and
    int() then raises. That guard reads like it handles missing values and
    handles only None and zero.

    It matters because a slate file legitimately mixes rows: re-running
    the predictor preserves rows for games already underway, so a column
    added today is present on the refreshed rows and absent on the
    preserved ones. On 2026-09-08, 10 of 30 pitcher rows had no
    career_starts and the Pitchers tab died on the first of them.
    """
    return int(value) if pd.notna(value) else default


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

# ---------------------------------------------------------------- picks
#
# The saved-hitter list, held in the URL rather than in a file.
#
# A file was the obvious choice and is the wrong one: the deployed app is
# ONE process serving everybody, so a file would be a single global list.
# Two people looking at the site would overwrite each other's picks and
# neither would understand why. The URL is per-viewer for free, survives a
# refresh, and makes the list shareable as a side effect -- send someone
# the address and they see your eight.
#
# Stored as player_id, not row position. Row positions are meaningless the
# moment the slate changes or the search box filters anything.
MAX_PICKS = 8
PICKS_PARAM = "picks"

# Starts-in-window thresholds for the two pitcher views.
#
# They differ on purpose. The Pitchers table is a reading aid: at or under
# ten starts the shrinkage prior is still doing most of the work (the K
# prior is 250 batters faced, about ten starts), so the number is flagged.
# The market table is a decision aid, and there the bar is higher -- a
# disagreement with five books means nothing if the model has only a
# partial read on the pitcher, so anything under twenty is called out.
THIN_STARTS = 10
THIN_STARTS_MARKET = 20

if "sp_picks" not in st.session_state:
    raw = st.query_params.get(PICKS_PARAM, "")
    # Truncated here rather than trusting the table to do it. A hand-edited
    # or stale link can carry any number of ids, and the enforcement in the
    # editor only runs when the editor does -- filter the search box down to
    # nothing and it does not.
    st.session_state.sp_picks = [
        int(x) for x in str(raw).split(",")
        if x.strip().lstrip("-").isdigit()][:MAX_PICKS]


def _write_picks(picks):
    """Session state and the URL, kept in step."""
    st.session_state.sp_picks = list(picks)
    if picks:
        st.query_params[PICKS_PARAM] = ",".join(str(p) for p in picks)
    elif PICKS_PARAM in st.query_params:
        del st.query_params[PICKS_PARAM]


def render_picks_panel(frame, prop_map, band_cuts):
    """
    The saved hitters, in Streamlit's own sidebar.

    The sidebar rather than a floating div because it already IS what was
    asked for -- a panel with an arrow that collapses it, present on every
    tab. A custom overlay would look the same and could not write back to
    Python, so a tick inside it would do nothing.

    Each hitter collapses to a line. Eight hitters times fourteen props is
    over a hundred numbers, which is a wall if it is all open at once and
    is exactly what an expander is for.
    """
    side = st.sidebar
    picks = list(st.session_state.sp_picks)

    side.markdown(
        f'<div style="font-size:14px;font-weight:640;color:var(--ink)">'
        f'My team</div>'
        f'<div style="font-size:12px;color:var(--ink3);margin-bottom:10px">'
        f'{len(picks)} of {MAX_PICKS} · tick <b>Save</b> on the All hitters '
        f'tab</div>', unsafe_allow_html=True)

    if st.session_state.pop("sp_full_msg", False):
        side.warning(f"Full at {MAX_PICKS}. Remove someone first.")

    if not picks:
        side.markdown(
            '<div style="font-size:12px;color:var(--ink3);line-height:1.6">'
            'Nobody saved yet. Your list is kept in the page address, so it '
            'survives a refresh and you can send the link to someone.</div>',
            unsafe_allow_html=True)
        return

    key_col = "player_id" if "player_id" in frame.columns else None
    missing = 0
    for pid in picks:
        rows = (frame[frame[key_col] == pid] if key_col
                else frame.loc[frame.index == pid])
        if rows.empty:
            missing += 1
            continue
        first = rows.iloc[0]
        # A doubleheader means two rows for one man, against two different
        # starters, with two different sets of numbers. Both are shown --
        # picking the first would quietly hide half of his night.
        title = f'{first.get("name", pid)} · {first.get("team", "")}'
        if len(rows) > 1:
            title += f' ({len(rows)} games)'
        with side.expander(title):
            for n, (_, r) in enumerate(rows.iterrows()):
                # Built as a list rather than nested inside one f-string:
                # quoting a dict key with the same quote character as the
                # surrounding f-string is a SyntaxError before Python 3.12,
                # and this file runs on whatever the deploy host provides.
                bits = [f'vs {r.get("opposing_pitcher", "?")}']
                slot = r.get("lineup_slot")
                if pd.notna(slot):
                    bits.append(f"bats {ORDINAL.get(int(slot), int(slot))}")
                pa = r.get("expected_pa")
                if pd.notna(pa):
                    bits.append(f"{float(pa):.2f} PA")
                rule = ("border-top:1px solid var(--line);padding-top:7px;"
                        "margin-top:9px;" if n else "")
                head = f"Game {n + 1} · " if len(rows) > 1 else ""
                st.markdown(
                    f'<div style="{rule}font-size:11.5px;color:var(--ink3);'
                    f'margin-bottom:6px">{head}{" · ".join(bits)}</div>',
                    unsafe_allow_html=True)
                # Every prop the slate carries, coloured on the same
                # slate-wide quartiles as the big table so a number means
                # the same thing in both places.
                cells = ""
                for key, (label, _fam) in prop_map.items():
                    value = r.get(key)
                    if pd.isna(value):
                        continue
                    b = band_of(value, band_cuts[key])
                    colour = BANDS[b][1] if b else "var(--ink2)"
                    cells += (
                        f'<div style="display:flex;'
                        f'justify-content:space-between;gap:8px;'
                        f'padding:2px 0;font-size:12px">'
                        f'<span style="color:var(--ink3)">{label}</span>'
                        f'<b style="color:{colour};'
                        f'font-variant-numeric:tabular-nums">'
                        f'{pct(value)}</b></div>')
                st.markdown(cells, unsafe_allow_html=True)
            if st.button("Remove", key=f"sp_rm_{pid}",
                         width="stretch"):
                _write_picks([p for p in picks if p != pid])
                st.session_state.sp_grid_nonce += 1
                st.rerun()

    if missing:
        side.markdown(
            f'<div style="font-size:11.5px;color:var(--warn);margin-top:8px">'
            f'{missing} saved hitter{"" if missing == 1 else "s"} '
            f'{"is" if missing == 1 else "are"} not on this slate. Still in '
            f'your link — they will reappear on a slate they play.</div>',
            unsafe_allow_html=True)

    if side.button("Clear all", width="stretch"):
        _write_picks([])
        st.session_state.sp_grid_nonce += 1
        st.rerun()

confirmed = int((df.get("lineup_status") == "confirmed").sum())
projected = int((df.get("lineup_status") == "projected").sum())
written = pd.Timestamp(os.path.getmtime(path), unit="s", tz="UTC") \
            .tz_convert("America/New_York")

# ---- the live clock ---------------------------------------------------
#
# WHY A COMPONENT AND NOT MARKDOWN
# --------------------------------
# Streamlit strips <script> out of st.markdown, so a clock written that way
# renders the time the server happened to send and then sits there, wrong,
# until something else triggers a rerun. st.components.v1.html renders into
# an iframe where the browser runs the script normally, so this ticks on
# its own and costs the server nothing -- no reruns, no polling, no
# st.rerun loop burning a websocket message a second.
#
# WHY THE BROWSER'S CLOCK AND NOT THE SERVER'S
# --------------------------------------------
# Everything time-shaped in this app is already anchored to
# America/New_York -- fmt_dt prints ET, when_label asks what day it is in
# ET, and start_time_utc is converted to ET for every game row. That is the
# calendar baseball is scheduled on. But the machine running this is not
# necessarily on either clock: deployed it runs in UTC, and "your time"
# means the person looking at the screen.
#
# So both halves come from the viewer's own browser. Intl.DateTimeFormat
# with a timeZone does the ET conversion, which means daylight saving is
# handled by the browser's tz database rather than by a hardcoded three
# hours that would silently go wrong twice a year -- and the offset line
# is computed from the two rendered times rather than assumed, so it reads
# correctly from anywhere.
_TZ_BALLPARK = "America/New_York"
components.html(
    """
<style>
  body{margin:0;font:500 12px/1.25 Archivo,system-ui,-apple-system,sans-serif;
       color:#8a8880;background:transparent}
  .w{display:flex;justify-content:flex-end;gap:14px;align-items:baseline}
  .b{white-space:nowrap}
  .t{font-variant-numeric:tabular-nums;color:#c3c2b7;font-weight:600}
  .l{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em}
  .o{color:#8a8880;font-size:10.5px}
</style>
<div class="w">
  <span class="b"><span class="l">you</span> <span class="t" id="a">--:--</span></span>
  <span class="b"><span class="l">ballpark</span> <span class="t" id="b">--:--</span></span>
  <span class="o" id="o"></span>
</div>
<script>
const ZONE = "__ZONE__";
// Same options for both, so the two strings are comparable and the only
// difference is the zone.
const opt = {hour:"numeric", minute:"2-digit", second:"2-digit", hour12:true};
const here = new Intl.DateTimeFormat([], opt);
const park = new Intl.DateTimeFormat([], Object.assign({timeZone: ZONE}, opt));
// Offset from the two zones' actual UTC offsets right now, so it survives
// daylight saving on either side and reads 0 when they are the same zone.
function offsetHours(d){
  const f = z => new Date(new Intl.DateTimeFormat("en-US", {timeZone:z,
      year:"numeric",month:"2-digit",day:"2-digit",
      // hourCycle h23, not hour12:false. Some engines (Safari
      // historically) render midnight as "24:00:00" under hour12:false,
      // which Date() then refuses to parse -- so the offset would go NaN
      // for exactly one hour a night and the label would blank out.
      hour:"2-digit",minute:"2-digit",second:"2-digit",hourCycle:"h23"})
      .format(d).replace(",", ""));
  const mine = Intl.DateTimeFormat().resolvedOptions().timeZone;
  return Math.round((f(ZONE) - f(mine)) / 36e5);
}
function tick(){
  const d = new Date();
  document.getElementById("a").textContent = here.format(d);
  document.getElementById("b").textContent = park.format(d);
  const h = offsetHours(d);
  document.getElementById("o").textContent =
    h === 0 ? "same zone" : (h > 0 ? "+" : "") + h + "h";
}
tick();
setInterval(tick, 1000);
</script>
""".replace("__ZONE__", _TZ_BALLPARK),
    height=24)

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


# Hoisted above the tabs. This is a flat script -- every `with tab_x:`
# block shares one namespace and runs top to bottom -- so a constant used
# by the FIRST tab cannot be defined next to the fifth. The drill-down on
# the Slate page needs these labels and died with NameError until they
# moved up here. Same shape of bug as the `cuts`/`k_cuts` collision.
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

# --------------------------------------------------------------- slate v2
WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
         7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven",
         12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen",
         16: "Sixteen"}


def _word(n, cap=False):
    w = WORDS.get(int(n), str(int(n)))
    return w if cap else w.lower()


def slate_summary(frame) -> str:
    """
    The line under the date, in the words a person would use.

    A template with branches, not prose someone typed once. The branches
    are the work: a sentence that reads well at fifteen games has to still
    read well at two, and at "every lineup is posted" as well as "none is".
    Nothing here is generated by a model -- a template cannot invent a
    first-pitch time, which is the whole argument for using one.
    """
    n_games = frame["game_pk"].nunique()
    bits = [f"{_word(n_games, cap=True)} game{'' if n_games == 1 else 's'}"]
    if "start" in frame.columns and frame["start"].notna().any():
        first, last = frame["start"].min(), frame["start"].max()
        part = "morning" if first.hour < 12 else "afternoon" if first.hour < 17 \
            else "evening"
        if n_games == 1:
            bits.append(f"first pitch {fmt_clock(first)}")
        else:
            venue = ""
            tail = frame[frame["start"] == last]
            if "venue_name" in tail.columns and len(tail):
                venue = f" in {tail['venue_name'].iloc[0]}"
            bits.append(f"running {fmt_clock(first)} this {part} through a "
                        f"{fmt_clock(last)} finish{venue}")
    line = ", ".join(bits) + ". "

    if "lineup_status" in frame.columns:
        by_game = frame.groupby("game_pk")["lineup_status"].apply(
            lambda c: (c == "confirmed").all())
        conf = int(by_game.sum())
        if conf == n_games:
            line += "Every lineup is posted."
        elif conf == 0:
            line += ("No lineups are posted yet, so every batting slot below "
                     "is the model's guess.")
        else:
            line += (f"Lineups are in for {_word(conf)} — the rest are still "
                     f"projected, so slots below the top of the order can move.")
    return line


def slate_findings(frame, pitchers, prop, prop_label, limit=4):
    """
    The card pool. Every entry is a check; the ones that fire get a card.

    This is the difference between a page that regenerates and a page that
    was written once. A market disagreement needs odds captured that day; a
    hot bat needs one to exist. Rather than leave a blank card, each check
    reports whether it fired, and the highest-priority hits are shown.

    Priority is lowest-first. The three or four that survive are the cards.
    """
    out = []

    def add(pri, kind, tone, fig, unit, txt):
        out.append(dict(pri=pri, kind=kind, tone=tone, fig=fig,
                        unit=unit, txt=txt))

    # 1. Always fires: the top of the board for the selected prop.
    if len(frame):
        b = frame.nlargest(1, prop).iloc[0]
        also = ""
        if "prob_hr" in frame.columns and prop != "prob_hr":
            if frame.nlargest(1, "prob_hr").iloc[0]["name"] == b["name"]:
                also = (f" He owns the slate's best home-run number too, at "
                        f"{pct(b['prob_hr'])} — the only hitter tonight "
                        f"leading both.")
        add(0, "Strongest on the board", "var(--accent)", pct(b[prop]),
            prop_label,
            f"<b>{b['name']}</b> against {b.get('opposing_pitcher', '—')}."
            + also)

    # 2 and 3. The two ends of tonight's pitching. Showing only the soft
    #    arm tells a hitter where to look and never where to avoid.
    if pitchers is not None and len(pitchers) >= 2 \
            and {"expected_k", "expected_bf"}.issubset(pitchers.columns):
        pit = pitchers.dropna(subset=["expected_k", "expected_bf"]).copy()
        pit = pit[pit["expected_bf"] > 0]
        if len(pit) >= 2:
            pit["kr"] = pit["expected_k"] / pit["expected_bf"]
            lg = float(pit["kr"].mean())
            hi, lo = pit.nlargest(1, "kr").iloc[0], pit.nsmallest(1, "kr").iloc[0]
            add(1, "Softest arm", "var(--b2)", f"{lo['kr']:.1%}",
                f"of batters struck out · slate {lg:.0%}",
                f"<b>{lo['pitcher']}</b> misses fewer bats than any starter "
                f"tonight, so <b>{lo.get('opponent', 'the')}</b> hitters facing "
                f"him put the ball in play more than anyone on the board. Good "
                f"for them, not for him.")
            # Deliberately the mirror image of the card above, sentence for
            # sentence. The first version ended "The worst spot on the
            # board for a hitter." -- a floating fragment that never said
            # WHOSE hitters, so you had to re-read it to work out that
            # "worst spot" meant worst for the other team and not for him.
            # The soft-arm card already had the right shape: name the
            # lineup, say what it means for them, close on who it favours.
            add(2, "Strongest arm", "var(--b1ink)", f"{hi['kr']:.1%}",
                f"of batters struck out · slate {lg:.0%}",
                f"<b>{hi['pitcher']}</b> misses more bats than any starter "
                f"tonight — <b>{hi['expected_k']:.1f}</b> strikeouts over "
                f"{hi['expected_bf']:.0f} batters. "
                f"<b>{hi.get('opponent', 'The')}</b> hitters facing him are "
                f"in the toughest spot on the board. Good for him, not for "
                f"them.")

    # 4. Fires whenever anything is still projected. On a fully confirmed
    #    slate it says so instead of hiding, which is the better message.
    if "lineup_status" in frame.columns:
        by_game = frame.groupby("game_pk")["lineup_status"].apply(
            lambda c: (c == "confirmed").all())
        n_games, waiting = len(by_game), int((~by_game).sum())
        if waiting:
            add(3, "Trust this less", "var(--b4)", f"{waiting} of {n_games}",
                "lineups not posted",
                "Batting slots in those games are the model's guess. Slot "
                "drives plate appearances and plate appearances drive every "
                "number below — re-run once they post.")

    # 5. Form marker, when one lands near the top. Often it does not: on
    #    2026-09-12 all ten of the top ten were Normal.
    if "form_state" in frame.columns and len(frame) >= 10:
        top = frame.nlargest(10, prop)
        for state, tone, verb in (("Hot", "var(--b4)", "running hot"),
                                  ("Cold", "var(--b1ink)", "running cold")):
            hit = top[top["form_state"] == state]
            if len(hit):
                who = hit.iloc[0]
                add(4 if state == "Hot" else 5, f"{state} bat near the top",
                    tone, f"#{int(top.reset_index().index[top.reset_index()['name'] == who['name']][0]) + 1}",
                    "on the board",
                    f"<b>{who['name']}</b> is {verb} against his own baseline. "
                    f"The model does not use the marker — it is tracked to find "
                    f"out whether it predicts anything.")
                break

    # 6. A probable who is not a starter. Needs the role column, which
    #    slates predicted before 2026-09-12 do not carry.
    if pitchers is not None and "role" in (pitchers.columns if pitchers is not None else []):
        odd = pitchers[pitchers["role"].isin(["opener", "reliever"])]
        if len(odd):
            o = odd.iloc[0]
            add(6, "Not really a starter", "var(--b2)",
                f"{o['expected_bf']:.0f}", "batters faced",
                f"<b>{o['pitcher']}</b> is listed as starting but the model has "
                f"him as {'an opener' if o['role'] == 'opener' else 'a reliever'}. "
                f"His numbers come from his own short outings, not a starter's "
                f"prior.")

    # 7. Games with only one probable listed.
    if pitchers is not None and len(pitchers) and "game_pk" in pitchers.columns:
        per = pitchers.groupby("game_pk").size()
        short = int((per < 2).sum())
        if short:
            add(7, "Missing a probable", "var(--ink3)", str(short),
                "games with one starter named",
                "Hitters in those games are projected against a league-average "
                "opponent until the second starter is announced.")

    out.sort(key=lambda d: d["pri"])
    return out[:limit]


# Loaded once, before the tabs: the Slate page reads it for the pitching
# findings and the Pitchers tab reads it for the table. Two tabs loading the
# same file twice is how they drift.
pit = None
_pit_path = os.path.join(CACHE_DIR, f"pitchers_{slate_date}.csv")
if os.path.exists(_pit_path):
    try:
        pit = pd.read_csv(_pit_path)
    except Exception:
        pit = None

# The team model's output, one row per game: projected runs a side, total,
# and how many hitters each lineup had when the prediction was made.
#
# The Game Lines TAB is off (see SHOW_GAME_LINES below for why) but the
# file it read is still written every night, and the half of it that works
# -- runs -- belongs on a game you have just clicked into. Absent file
# means no block, not an empty state: a slate predicted before the team
# model existed should look like a slate, not like something broke.
teams = None
_team_path = os.path.join(CACHE_DIR, f"teams_{slate_date}.csv")
if os.path.exists(_team_path):
    try:
        teams = pd.read_csv(_team_path)
    except Exception:
        teams = None


# --------------------------------------------------------------------
# PER-PITCHER STRIKEOUT LINES
#
# The Pitchers tab used to show two fixed columns, over 5.5 and over 6.5,
# for everyone. Nolan: "sometimes there's a line for over five point five
# for all the people on the slate... his betting line's for like eight
# eight point five. So it doesn't make sense to have a five point five
# line for someone that's gonna be over eight."
#
# Right, and the fix costs nothing, because predict_slate already writes
# the WHOLE distribution to the `k_dist` column -- a comma-separated pmf
# indexed from zero strikeouts. Verified against Tyler Glasnow on
# 2026-09-12: summing k_dist[6:] reproduces the stored prob_k_over_5.5 to
# seven decimals. So any line is available for any pitcher on any slate
# already on disk, with nothing re-run and no history lost.
#
# WHICH line to show is the interesting half. The answer is the one the
# book is actually offering, because that is the bet that exists; his own
# distribution is the fallback for a night when the odds fetch was late,
# missed, or skipped a game that had already started.
# --------------------------------------------------------------------
def parse_pmf(text):
    """A stored `k_dist` / `outs_dist` string as a numpy array, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        pmf = np.array([float(x) for x in text.split(",") if x.strip()])
    except ValueError:
        return None
    return pmf if pmf.size and pmf.sum() > 0 else None


def prob_over(pmf, line):
    """
    P(count > line) for a half-integer line.

    Over 5.5 means six or more, so the sum starts at ceil(line). Getting
    this off by one would shift every probability on the page by a whole
    strikeout, silently and in the direction that flatters the model.
    """
    if pmf is None:
        return float("nan")
    start = int(np.ceil(line))
    return float(pmf[start:].sum()) if start < len(pmf) else 0.0


def own_line(pmf):
    """
    The half-integer line closest to a coin flip for this pitcher.

    Not the mean. A book sets the line where the over and under are close
    to even money, and for a skewed count distribution the median is
    nearer that point than the mean is. Walking the candidate lines and
    taking the one whose over probability is closest to 0.5 finds it
    directly and needs no assumption about the shape.
    """
    if pmf is None:
        return float("nan")
    cands = np.arange(0.5, min(len(pmf), 16) + 0.5, 1.0)
    if not len(cands):
        return float("nan")
    return float(min(cands, key=lambda L: abs(prob_over(pmf, L) - 0.5)))


def market_lines(date):
    """
    {player: (line, market_prob)} from the captured odds for this slate.

    The line closest to even money is the one the book is surest about,
    and the one worth putting in a single column -- books offer several
    alternates per pitcher and the outer ones carry most of the hold.

    Reads odds_{date}.csv rather than market_compare_{date}.csv because
    the odds file exists whenever a fetch happened, while the compare
    file additionally requires compare_market.py to have run afterwards.
    """
    path = os.path.join(CACHE_DIR, f"odds_{date}.csv")
    if not os.path.exists(path):
        return {}
    try:
        odds = pd.read_csv(path)
    except Exception:
        return {}
    need = {"player", "line", "prob_over"}
    if not need <= set(odds.columns):
        return {}
    if "market" in odds.columns:
        odds = odds[odds["market"] == "pitcher_strikeouts"]
    odds = odds.dropna(subset=["player", "line", "prob_over"])
    if odds.empty:
        return {}
    odds = odds.assign(_d=(odds["prob_over"] - 0.5).abs())
    best = odds.sort_values("_d").groupby("player").first()
    # How many books quoted it comes back too. It matters more than it
    # looks: on 2026-09-13 Joe Ryan's captured line was 3.5 from THREE
    # books, and the model -- which projects him at 5.4 strikeouts --
    # read that as a +21 point edge. A 132-start starter does not have a
    # 3.5 line unless something is going on (a pitch limit, a bullpen
    # game) that three books know about and the model does not. Treating
    # a thin alternate as "the market" manufactures exactly the edges
    # that are least real, so the count is shown and read as a warning.
    return {str(k): (float(v["line"]), float(v["prob_over"]),
                     int(v["n_books"]) if "n_books" in best.columns
                     and pd.notna(v.get("n_books")) else 0)
            for k, v in best.iterrows()}


_when, _when_class = when_label(slate_date)
# Built by hand rather than with %-d, which is POSIX-only and crashes on
# Windows -- this project has already hit that once, in fmt_clock.
_ts = pd.Timestamp(slate_date)
_pretty = f"{_ts.strftime('%A, %B')} {_ts.day}"
_badge = (f'<span class="sp-when {_when_class}">{_when}</span>'
          if _when else "")

# The 30px gradient square is gone. It was a shape where information
# should have been, which is why it read as a placeholder -- it was one.
# The date is what you need to be certain about, so the date is the mark.
_hl, _hr = st.columns([3, 1])
with _hl:
    html(f'<div class="sp-date sp-display" style="font-size:clamp(28px,4vw,40px);'
         f'line-height:1.04">{_pretty}{_badge}</div>'
         f'<div class="sp-summary">{slate_summary(df)}</div>')
with _hr:
    # Model health, permanently visible. A page that shows probabilities
    # should show how the probabilities have been doing, without a trip to
    # the Results tab to find out.
    _hs = ""
    _hp = os.path.join(CACHE_DIR, "pitcher_scoring_log.csv")
    if os.path.exists(_hp):
        try:
            _hlog = pd.read_csv(_hp)
            _hlog = _hlog[pd.to_datetime(_hlog["game_date"], errors="coerce")
                          >= pd.Timestamp("2026-09-05")]
            _k = _hlog[_hlog["line"] == 5.5]
            if len(_k):
                _w = _k["n"].sum()
                _sk = float((_k["brier_skill"] * _k["n"]).sum() / _w)
                _hs += (f'Strikeouts <b>{_sk * 100:+.1f}%</b> over 5.5<br>')
        except Exception:
            pass
    _sp = os.path.join(CACHE_DIR, "scoring_log.csv")
    if os.path.exists(_sp):
        try:
            _slog = pd.read_csv(_sp)
            for _c in ("n", "base_rate", "brier"):
                if f"clean_{_c}" in _slog.columns:
                    _slog[_c] = _slog[f"clean_{_c}"].fillna(_slog[_c])
            _g = _slog[_slog["label"] == "at least 1 walk"]
            if len(_g):
                _w = _g["n"].sum()
                _did = float((_g["base_rate"] * _g["n"]).sum() / _w)
                _br = float((_g["brier"] * _g["n"]).sum() / _w)
                _ref = _did * (1 - _did)
                if _ref > 0:
                    _hs += (f'Best hitter prop, walks '
                            f'<b>{(1 - _br / _ref) * 100:+.1f}%</b>')
        except Exception:
            pass
    if _hs:
        html(f'<div class="sp-health"><span class="k">Model, scored slates'
             f'</span>{_hs}</div>')

# Looking at an old slate is a legitimate thing to do and a very easy
# thing to do by accident, since the picker defaults to the newest file
# and the newest file is not necessarily today's. Say so plainly rather
# than letting yesterday's probabilities read as tonight's.
if _when_class == "past":
    html(f'<div class="sp-stale">These games have already been played. '
         f'You are looking at what the model said before '
         f'{_when.lower()}\'s slate, not at tonight\'s.</div>')

# The Game Lines tab is retired, not deleted.
#
# It showed the team model: projected score, win probability, and who the
# model expected to do the scoring. Graded over four slates it had no
# skill at picking winners -- home-win Brier 0.237 to 0.265 against 0.25
# for a coin flip -- because the win probabilities only ever spanned 0.45
# to 0.57. It also projected total runs about half a run worse than the
# market. A tab that cannot separate tonight's games is a tab that costs
# attention and returns nothing.
#
# Everything behind it keeps running: predict_slate still writes
# cache/teams_{date}.csv and score_slate.score_teams still grades it
# against final scores into team_scoring_log.csv. If the model ever gets
# a real home-field term or a wider spread, flip this back to True and
# the tab returns exactly as it was.
SHOW_GAME_LINES = False

_TAB_LABELS = ["Slate", "Games", "Pitchers", "All hitters", "Bet ready",
               "Results"]
if SHOW_GAME_LINES:
    _TAB_LABELS.insert(3, "Game Lines")
_TABS = dict(zip(_TAB_LABELS, st.tabs(_TAB_LABELS)))
tab_slate = _TABS["Slate"]
tab_games = _TABS["Games"]
tab_pitch = _TABS["Pitchers"]
tab_all = _TABS["All hitters"]
tab_bet = _TABS["Bet ready"]
tab_res = _TABS["Results"]
tab_lines = _TABS.get("Game Lines")


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
    # ---- what is true about tonight --------------------------------
    _prop_label = props[DEFAULT_PROP][0] if DEFAULT_PROP in props \
        else list(props.values())[0][0]
    _findings = slate_findings(df, pit, DEFAULT_PROP, _prop_label)
    if _findings:
        html('<div class="sp-read">' + "".join(
            f'<div class="sp-find">'
            f'<div class="kind" style="color:{f["tone"]}">{f["kind"]}</div>'
            f'<div class="fig">{f["fig"]}</div>'
            f'<div class="unit">{f["unit"]}</div>'
            f'<div class="txt">{f["txt"]}</div></div>'
            for f in _findings) + '</div>')

    # ---- the slate ---------------------------------------------------
    #
    # The page's centre of gravity. The old front page ranked twenty
    # hitters and never mentioned a game; this is every game in first-pitch
    # order, and each row is a link.
    #
    # A link rather than a button because Streamlit has no API to switch
    # tabs -- setting ?game=<pk> reruns the app, the detail opens in place
    # below, and the Games tab agrees because the same session key is set.
    _open = st.query_params.get("game")
    _open = int(_open) if str(_open).isdigit() else None
    if _open is not None:
        st.session_state.sp_game = _open

    _g = (df.assign(_c=(df.get("lineup_status") == "confirmed"))
            .groupby("game_pk")
            .agg(start=("start", "first"),
                 venue=("venue_name", "first"),
                 conf=("_c", "all"))
            .reset_index().sort_values("start", na_position="last"))

    # Built into a list rather than one HTML string, because each row is
    # now a Streamlit column pair: a real button for the time, and the
    # rest of the row as HTML. See the .sp-g2 note in the CSS -- a
    # <a href="?game="> is a browser navigation and reloads the page.
    _rowdata = []
    for _r in _g.itertuples():
        _gg = df[df.game_pk == _r.game_pk]
        _away = _gg[_gg.is_home == 0]["team"]
        _home = _gg[_gg.is_home == 1]["team"]
        _a = _away.iloc[0] if len(_away) else "?"
        _h = _home.iloc[0] if len(_home) else "?"
        _arms = ""
        if pit is not None and "expected_k" in pit.columns:
            _pg = pit[pit.game_pk == _r.game_pk]
            _arms = " · ".join(
                f'<b>{str(x.pitcher).split()[-1]}</b> {x.expected_k:.1f}'
                for x in _pg.itertuples() if pd.notna(x.expected_k))
        _best = _gg.nlargest(1, DEFAULT_PROP)
        _bn = _bv = ""
        if len(_best):
            _b = _best.iloc[0]
            _slot = _b.get("lineup_slot")
            _bn = (f'{_b["name"]} <i>· bats '
                   f'{ORDINAL.get(int(_slot), int(_slot))}</i>'
                   if pd.notna(_slot) else str(_b["name"]))
            _bv = pct(_b[DEFAULT_PROP])
        _rowdata.append({
            "pk": int(_r.game_pk), "clock": fmt_clock(_r.start),
            "conf": bool(_r.conf),
            # The confirmed/projected dot used to live in the time cell.
            # A button label cannot carry it, so it moves to the front of
            # the matchup -- same information, same row, one cell right.
            "html": (f'<span class="sp-dot {"ok" if _r.conf else "wait"}">'
                     f'</span>{_a}<i>at</i>{_h}'),
            "arms": _arms, "best": _bn, "value": _bv})

    _l, _r2 = st.columns([3, 1])
    with _l:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink)">'
             'The slate</div>')
    with _r2:
        html('<div style="font-size:12px;color:var(--ink3);text-align:right;'
             'padding-top:4px"><span class="sp-dot ok"></span>confirmed'
             '<span class="sp-dot wait" style="margin-left:12px"></span>'
             'projected</div>')
    # One column pair per game: the clock as a real button, everything
    # right of it as HTML on a shared grid so the columns line up down
    # the page. Setting st.query_params rather than following a link
    # updates the address bar WITHOUT a navigation, so the game stays
    # shareable and the page no longer flashes or loses your scroll.
    html('<div class="sp-games" style="margin-bottom:2px"></div>')
    for _row in _rowdata:
        _cb, _ch = st.columns([1, 8.4])
        with _cb:
            if st.button(_row["clock"], key=f'gbtn_{_row["pk"]}',
                         width="stretch",
                         type=("primary" if _open == _row["pk"]
                               else "secondary")):
                # Clicking the open game closes it, so the button is a
                # toggle rather than a one-way trip that needs the
                # "all games" link to undo.
                if _open == _row["pk"]:
                    # del rather than pop: QueryParams is a MutableMapping
                    # but which dict methods it exposes has moved between
                    # Streamlit versions. Falling back to an empty value
                    # is safe because _open only accepts a digit string,
                    # so "" reads as closed either way.
                    try:
                        del st.query_params["game"]
                    except Exception:
                        st.query_params["game"] = ""
                    st.session_state.pop("sp_game", None)
                else:
                    # Assigning one key leaves the others alone, so the
                    # saved-hitter list in ?picks= survives on its own.
                    # The old link had to re-append it by hand or opening
                    # a game silently emptied the sidebar.
                    st.query_params["game"] = str(_row["pk"])
                st.rerun()
        with _ch:
            html(f'<div class="sp-g2{" on" if _open == _row["pk"] else ""}">'
                 f'<div class="m">{_row["html"]}</div>'
                 f'<div class="p">{_row["arms"]}</div>'
                 f'<div class="b">{_row["best"]}</div>'
                 f'<div class="v">{_row["value"]}</div></div>')

    # ---- drill-down, in place ---------------------------------------
    if _open is not None and _open in set(df.game_pk):
        _sel = df[df.game_pk == _open]
        _sa = _sel[_sel.is_home == 0]["team"]
        _sh = _sel[_sel.is_home == 1]["team"]
        # A button, not a link, for the same reason the clock is: an
        # <a href="?"> navigates and reloads. The clock button already
        # toggles the game shut, so this is the second way out rather
        # than the only one -- but a reload on either would be the same
        # flash Nolan saw.
        _dh, _db = st.columns([5, 1])
        with _dh:
            html(f'<div class="sp-display" style="font-size:19px;'
                 f'margin:22px 0 4px">'
                 f'{_sa.iloc[0] if len(_sa) else "?"} at '
                 f'{_sh.iloc[0] if len(_sh) else "?"}</div>')
        with _db:
            if st.button("← all games", key="sp_back", width="stretch"):
                try:
                    del st.query_params["game"]
                except Exception:
                    st.query_params["game"] = ""
                st.session_state.pop("sp_game", None)
                st.rerun()

        # ---- the team numbers for this game --------------------------
        #
        # Runs and who scores them. No winner and no margin: both were
        # measured and both were noise, and a number that is shown gets
        # believed regardless of what the caption says about it.
        #
        # exp_runs is p(run) x expected_pa, written by predict_slate --
        # the same per-hitter terms that were summed to make the team
        # total, so the three names under a side really are the biggest
        # contributors to the figure above them, not a separate ranking.
        _tm = None
        if teams is not None and "game_pk" in teams.columns:
            _tm = teams[teams.game_pk == _open]
            _tm = _tm.iloc[0] if len(_tm) else None
        if _tm is not None:
            _ar = pd.to_numeric(_tm.get("away_runs"), errors="coerce")
            _hr = pd.to_numeric(_tm.get("home_runs"), errors="coerce")
            if pd.notna(_ar) and pd.notna(_hr):
                _an = _tm.get("away_team", "?")
                _hn = _tm.get("home_team", "?")

                def _scorers(team, n=3):
                    if "exp_runs" not in _sel.columns:
                        return ""
                    side = (_sel[_sel["team"] == team]
                            .dropna(subset=["exp_runs"]).nlargest(n, "exp_runs"))
                    return " · ".join(
                        f'<b>{str(r["name"]).split()[-1]}</b> {r["exp_runs"]:.2f}'
                        for _, r in side.iterrows() if pd.notna(r.get("name")))

                _body = ""
                _as, _hs = _scorers(_an), _scorers(_hn)
                if _as or _hs:
                    _body = (f'<div class="sp-sc">Expected to score, in runs'
                             f'<div class="who">{_an} &nbsp;{_as}</div>'
                             f'<div class="who">{_hn} &nbsp;{_hs}</div></div>')

                # A lineup the model could only partly fill scores low for
                # a reason that has nothing to do with the teams. Say so,
                # or the gap reads as a projection about the matchup.
                _short = [t for t, k in ((_an, "away_hitters"),
                                         (_hn, "home_hitters"))
                          if pd.notna(_tm.get(k)) and int(_tm[k]) < 9]
                _thin = (f'<div class="sp-thin">Lineup incomplete for '
                         f'{" and ".join(_short)} — that side\'s runs are low '
                         f'by roughly the missing share.</div>') if _short else ""

                html(f'<div class="sp-tb"><div class="sd">'
                     f'<div><span class="tm">{_an}</span>'
                     f'<span class="rn">{_ar:.1f}</span></div>'
                     f'<div class="at">at</div>'
                     f'<div><span class="tm">{_hn}</span>'
                     f'<span class="rn">{_hr:.1f}</span></div>'
                     f'<div class="tot"><b>{_ar + _hr:.1f}</b> runs total</div>'
                     f'</div>{_body}{_thin}'
                     f'<div class="sp-thin" style="color:var(--ink3)">'
                     f'A projected score is the <b style="color:var(--ink2)">'
                     f'average</b> of how the game goes, not a guess at the '
                     f'final. No winner is shown because the model has not '
                     f'earned one — graded over four slates its win '
                     f'probabilities never left 45–57% and beat a coin flip '
                     f'on none of them.</div></div>')

        _cols = ["name", "team", "lineup_slot", "expected_pa",
                 "opposing_pitcher"] + list(props)
        _view = _sel[[c for c in _cols if c in _sel.columns]].copy()
        _view = _view.sort_values(["team", "lineup_slot"], na_position="last")
        _view = _view.rename(columns={
            **{c: BASE_LABELS[c] for c in BASE_LABELS if c in _view.columns},
            "opposing_pitcher": "Facing",
            **{k: props[k][0] for k in props if k in _view.columns}})
        _pl = [props[k][0] for k in props if props[k][0] in _view.columns]
        _st = _view.style
        for _lab in _pl:
            _key = next(k for k in props if props[k][0] == _lab)
            _st = _st.apply(
                lambda col, c=cuts[_key]: [
                    ("" if band_of(v, c) == 0 else
                     f"background-color:{BANDS[band_of(v, c)][0]};"
                     f"color:{BANDS[band_of(v, c)][1]}")
                    for v in col], subset=[_lab])
        _fmt = {l: "{:.1%}" for l in _pl}
        if "PA" in _view.columns:
            _fmt["PA"] = "{:.2f}"
        if "Slot" in _view.columns:
            _fmt["Slot"] = "{:.0f}"
        st.dataframe(_st.format(_fmt, na_rep="—"), width="stretch",
                     hide_index=True,
                     height=min(700, 40 + 35 * len(_view)))

    # ---- leaderboard and what to watch -------------------------------
    _lb, _wt = st.columns([1.55, 1])
    with _lb:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink)">'
             'Best on the board</div>')

        # Chips rather than a dropdown.
        #
        # A select box hides eleven of twelve options behind a click, which
        # is backwards for something you switch between constantly and
        # where the LIST is itself information -- seeing that TB 3.5 and
        # HR exist next to each other is half of knowing what to look at.
        #
        # Buttons also rerun over the websocket, so the page does not
        # reload the way a query-param link does.
        #
        # Six to a row: twelve props do not fit across one line at this
        # width, and st.columns does not wrap.
        _pkeys = list(props)
        if "lb_prop" not in st.session_state or \
                st.session_state.lb_prop not in props:
            st.session_state.lb_prop = (DEFAULT_PROP if DEFAULT_PROP in props
                                        else _pkeys[0])
        for _start in range(0, len(_pkeys), 6):
            _chunk = _pkeys[_start:_start + 6]
            # Pad to a full six so a short last row keeps the same chip
            # width as the row above it rather than stretching.
            _cols = st.columns(6)
            for _col, _key in zip(_cols, _chunk):
                with _col:
                    if st.button(
                            props[_key][0], key=f"chip_{_key}",
                            width="stretch",
                            type=("primary" if st.session_state.lb_prop == _key
                                  else "secondary")):
                        st.session_state.lb_prop = _key
                        st.rerun()
        prop = st.session_state.lb_prop
        _top = df.nlargest(8, prop)
        _med = float(df[prop].median())
        _lbr = "".join(
            f'<div class="r"><div class="n">{i}</div>'
            f'<div class="w">{r["name"]} <i>{r.get("team","")}</i>'
            f'{form_chip(r)}</div>'
            f'<div class="sp-tw"><div class="sp-track">'
            f'<div class="sp-fill" style="width:{min(r[prop]*100,100):.1f}%">'
            f'</div></div><div class="sp-med" style="left:{_med*100:.1f}%">'
            f'</div></div>'
            f'<div class="v">{pct(r[prop])}</div></div>'
            for i, (_, r) in enumerate(_top.iterrows(), start=1))
        html(f'<div class="sp-lb">{_lbr}</div>'
             f'<div style="margin-top:10px;font-size:12px;color:var(--ink3)">'
             f'The tick is the slate median, {pct(_med)} — everything here '
             f'beats a typical hitter tonight. All {len(df)} are on the '
             f'<b style="color:var(--ink2)">All hitters</b> tab.</div>')
    with _wt:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink);'
             'margin-bottom:7px">Watch tonight</div>')
        _w = ""
        for f in slate_findings(df, pit, prop, props[prop][0], limit=9)[3:]:
            _w += (f'<div class="sp-w"><div class="t">{f["kind"]}</div>'
                   f'<div class="d">{f["txt"]}</div></div>')
        html(f'<div class="sp-watch">{_w}</div>' if _w else
             '<div style="font-size:12.5px;color:var(--ink3)">'
             'Nothing unusual about tonight — every check came back clean.</div>')


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
            col.button(label, key=f"sp_g{pk}", width="stretch",
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
    # ---- the low-line warning, reserved here and written below --------
    #
    # It has to APPEAR at the top, next to the heading, and it can only be
    # COMPUTED after pit_view exists, which is two hundred lines down. A
    # flat script runs top to bottom, so the slot is claimed now and
    # filled later; st.empty() holds the position.
    #
    # Left unfilled it renders nothing at all -- no gap, no empty box. That
    # is the intended state on most nights, and it is the whole point of
    # the thing being conditional: a warning that is always on the page
    # stops being read within a week. This one is only there when tonight's
    # slate actually has rows in the band the model runs hot in.
    pit_warn_slot = st.empty()
    # The pitcher props live in their own file. A slate row is one hitter,
    # and a starter is not a hitter -- carrying nine pitcher columns on
    # every batter row to describe fourteen pitchers would be worse than a
    # second table.
    if pit is not None and not pit.empty and "prob_k_over_5.5" in pit.columns:
        # Sortable, like All hitters, rather than hand-built HTML.
        #
        # There are thirteen numbers a row now and the interesting reads
        # are comparisons: who goes deepest, who misses the most bats per
        # batter faced, which thin starter the model is least sure about.
        # A fixed sort by expected strikeouts answers one of those and
        # hides the rest.
        #
        # Two things that were crammed inside cells become columns,
        # because a column can be sorted and a suffix cannot:
        #   - innings, which was "(5.3 ip)" inside the Outs cell
        #   - days since his last start, which was a BACK chip and a
        #     tooltip. As a number it is better than a chip: normal rest
        #     is 4-6 days, so a 464 sorts straight to the top and a
        #     6-day and a 40-day layoff are no longer the same thing.
        #
        # Names are prefixed `pit_` throughout. This is a flat script and
        # every tab shares one namespace -- naming a local `cuts` here
        # once clobbered the hitter quartiles and killed a tab that runs
        # later. The tab that broke was not the tab with the bug.
        pit = pit.sort_values("expected_k", ascending=False)

        pit_view = pd.DataFrame({
            "Pitcher": pit["pitcher"],
            "Team": pit["team"],
            "Opp": pit["opponent"],
            "Starts": pd.to_numeric(pit.get("starts_seen"), errors="coerce"),
            # Days since he last PITCHED, not since he last started.
            #
            # Started life as days_since_last_start and was wrong for the
            # case it most needed to be right about: Sean Newcomb read 507
            # days on 2026-09-12 having pitched three days earlier. He had
            # not STARTED in 507 days -- he is a reliever -- and the column
            # said "coming back from something" about a man working every
            # third day. Falls back to the old field for slates predicted
            # before appearances were exported.
            "Rest": pd.to_numeric(
                pit.get("days_since_last_appearance",
                        pit.get("days_since_last_start")), errors="coerce"),
            "Role": pit.get("role"),
            # Fastball mph over his last three starts against his own
            # baseline. Written by pitcher_form.py, which can read the
            # statcast caches this app never will.
            "Velo": pd.to_numeric(pit.get("velo_drop"), errors="coerce"),
            # How he has been DOING, as against how hard he is throwing.
            # The two disagree often -- Landen Roupp on 2026-09-15 read
            # velocity Normal while his last four starts sat 2.15 sigma
            # under his own strikeout rate. Display only: recent K rate is
            # real but does not survive next to velocity, so it is not fed
            # to anything. See KFORM_WINDOW in pitcher_form.py.
            "Form": pd.to_numeric(pit.get("kform_z"), errors="coerce"),
            "BF": pd.to_numeric(pit["expected_bf"], errors="coerce"),
            "K": pd.to_numeric(pit["expected_k"], errors="coerce"),
        })
        # A slate predicted before the strikeout-form marker existed has no
        # kform_z, and an all-empty column is worse than no column. Velo
        # predates this guard and gets it too, so an old cache renders the
        # same way it always did.
        for _c in ("Form", "Velo"):
            if _c in pit_view and not pit_view[_c].notna().any():
                pit_view = pit_view.drop(columns=[_c])

        PIT_PROPS = {}

        # ---- the strikeout line, one per pitcher ---------------------
        #
        # Three columns, each meaning exactly one thing:
        #   Line   the number on the board, or his own if none was caught
        #   Over   the model's probability AT THAT LINE
        #   Edge   model minus market, blank when there is no market
        #
        # Over is deliberately NOT colour-banded. Every line sits near its
        # own pitcher's coin flip by construction, so the column clusters
        # around 50% and a quartile colour on it would be reading noise.
        # Edge is the decision-relevant number and is the one coloured.
        _mkt = market_lines(slate_date)
        if "k_dist" in pit.columns:
            _pmfs = [parse_pmf(v) for v in pit["k_dist"]]
            _lines, _over, _mp, _bk = [], [], [], []
            for _name, _pmf in zip(pit["pitcher"], _pmfs):
                _hit = _mkt.get(str(_name))
                _line = _hit[0] if _hit else own_line(_pmf)
                _lines.append(_line)
                _over.append(prob_over(_pmf, _line) if pd.notna(_line)
                             else float("nan"))
                _mp.append(_hit[1] if _hit else float("nan"))
                _bk.append(_hit[2] if _hit else 0)
            pit_view["Line"] = _lines
            pit_view["Over"] = _over
            pit_view["Mkt"] = _mp
            pit_view["Books"] = _bk
            pit_view["Edge"] = [
                (o - m) if pd.notna(m) and pd.notna(o) else float("nan")
                for o, m in zip(_over, _mp)]
            PIT_PROPS["Edge"] = "edge"
        else:
            # A slate predicted before k_dist was written still renders,
            # with the two fixed columns it was built with.
            for col, label in (("prob_k_over_5.5", "5.5 K"),
                               ("prob_k_over_6.5", "6.5 K")):
                if col in pit.columns:
                    pit_view[label] = pd.to_numeric(pit[col], errors="coerce")
                    PIT_PROPS[label] = col
        if "expected_outs" in pit.columns:
            outs = pd.to_numeric(pit["expected_outs"], errors="coerce")
            pit_view["Outs"] = outs
            pit_view["IP"] = outs / 3.0
        for col, label in (("prob_outs_over_14.5", "14.5 outs"),
                           ("prob_outs_over_17.5", "17.5 outs")):
            if col in pit.columns and pit[col].notna().any():
                pit_view[label] = pd.to_numeric(pit[col], errors="coerce")
                PIT_PROPS[label] = col

        # Quartiles measured on the whole slate, so the colour means the
        # same thing after the user sorts as it did before.
        #
        # Edge is excluded on purpose. A quartile always paints a top
        # quarter, so on a night when the model and the market agree
        # everywhere it would colour the largest of a set of trivial
        # disagreements and call it a find. Edge gets FIXED thresholds
        # instead -- the same 5 points the market block below uses -- so
        # no colour on the page is the correct and common answer.
        pit_cuts = {lab: [float(pit_view[lab].quantile(q))
                          for q in (.25, .50, .75)]
                    for lab in PIT_PROPS if lab != "Edge"}

        # Four books is the threshold for calling something a consensus.
        # Below it a "line" is one or two shops posting an alternate, and
        # a disagreement with it is not an edge.
        MIN_BOOKS = 4

        def pit_edge_fill(series):
            books = pit_view.get("Books")
            out = []
            for i, v in enumerate(series):
                thin = books is not None and books.iloc[i] < MIN_BOOKS
                if pd.isna(v) or abs(v) < 0.05 or thin:
                    out.append("")
                else:
                    bg, ink = BANDS[3] if v > 0 else BANDS[1]
                    out.append(f"background-color:{bg};color:{ink}")
            return out

        def pit_books_fill(series):
            out = []
            for v in series:
                if pd.isna(v) or v == 0:
                    out.append(f"color:{_css_var('ink3')}")   # no price
                elif v < MIN_BOOKS:
                    out.append(f"color:{_css_var('warn')}")   # thin
                else:
                    out.append("")
            return out

        def pit_band_fill(series, column_cuts):
            out = []
            for value in series:
                band = 0 if pd.isna(value) else band_of(value, column_cuts)
                if band == 0:
                    out.append("")
                else:
                    bg, ink = BANDS[band]
                    out.append(f"background-color:{bg};color:{ink}")
            return out

        def pit_thin_fill(series):
            # Same threshold the footnote explains, applied to the number
            # itself so it survives sorting.
            return [f"color:{_css_var('warn')}"
                    if pd.notna(v) and v <= THIN_STARTS else ""
                    for v in series]

        def pit_rest_fill(series):
            # A starter works on four to six days' rest. Past a month he
            # is coming back from something, and the model has no idea
            # what -- that is the Burnes case, 464 days.
            return [f"color:{_css_var('warn')}"
                    if pd.notna(v) and v > 30 else "" for v in series]

        pit_styled = pit_view.style
        for lab in pit_cuts:          # NOT PIT_PROPS -- Edge is excluded
            pit_styled = pit_styled.apply(pit_band_fill,
                                          column_cuts=pit_cuts[lab],
                                          subset=[lab])
        if "Starts" in pit_view:
            pit_styled = pit_styled.apply(pit_thin_fill, subset=["Starts"])
        if "Rest" in pit_view:
            pit_styled = pit_styled.apply(pit_rest_fill, subset=["Rest"])
        if "Role" in pit_view:
            pit_styled = pit_styled.apply(
                lambda col: [f"color:{_css_var('warn')}"
                             if v in ("opener", "reliever", "unknown") else ""
                             for v in col], subset=["Role"])
        if "Velo" in pit_view:
            # Green up, amber down, nothing in between. The cut is the
            # same 0.5 mph pitcher_form.py uses, which was chosen from
            # 4,599 cached starts rather than picked round.
            pit_styled = pit_styled.apply(
                lambda col: [
                    "" if pd.isna(v) or abs(v) < 0.5 else
                    f"color:{_css_var('b4ink')}" if v > 0 else
                    f"color:{_css_var('warn')}" for v in col],
                subset=["Velo"])
        if "Form" in pit_view:
            # Same treatment as Velo and the same logic: the cut is the
            # 1.25 sigma pitcher_form.py measured, not a round number.
            pit_styled = pit_styled.apply(
                lambda col: [
                    "" if pd.isna(v) or abs(v) < 1.25 else
                    f"color:{_css_var('b4ink')}" if v > 0 else
                    f"color:{_css_var('warn')}" for v in col],
                subset=["Form"])
        if "Edge" in pit_view:
            pit_styled = pit_styled.apply(pit_edge_fill, subset=["Edge"])
        if "Books" in pit_view:
            pit_styled = pit_styled.apply(pit_books_fill, subset=["Books"])
        pit_fmt = {lab: "{:.1%}" for lab in PIT_PROPS}
        pit_fmt.update({"BF": "{:.1f}", "K": "{:.1f}", "Starts": "{:.0f}",
                        "Rest": "{:.0f}"})
        if "Velo" in pit_view:
            pit_fmt["Velo"] = "{:+.1f}"
        if "Form" in pit_view:
            pit_fmt["Form"] = "{:+.1f}"
        for _c, _f in (("Line", "{:.1f}"), ("Over", "{:.1%}"),
                       ("Mkt", "{:.1%}"), ("Edge", "{:+.1%}"),
                       ("Books", "{:.0f}")):
            if _c in pit_view:
                pit_fmt[_c] = _f
        if "Outs" in pit_view:
            pit_fmt.update({"Outs": "{:.1f}", "IP": "{:.1f}"})
        pit_styled = pit_styled.format(pit_fmt, na_rep="—")

        pit_config = {
            "Pitcher": st.column_config.Column(pinned=True, width=150),
            "Team": st.column_config.Column(width=58),
            "Opp": st.column_config.Column(width=58,
                                           help="The lineup he faces tonight"),
            "Starts": st.column_config.Column(
                width=62,
                help=f"Starts in the last 12 months. At or under "
                     f"{THIN_STARTS} (amber) his batters faced and "
                     f"strikeout rate lean on the league prior rather "
                     f"than on him, so read those as 'a starter in this "
                     f"spot', not as a read on the man."),
            "Rest": st.column_config.Column(
                width=58,
                help="Days since he last pitched, in any role. Four to six "
                     "is a normal turn in the rotation. Amber past 30 means "
                     "he is coming back from something the model cannot "
                     "see — and a pitcher on a rehab leash goes shorter "
                     "than any prior expects."),
            "Role": st.column_config.Column(
                width=72,
                help="How the model has him. 'starter' has real starts in "
                     "the last 12 months. 'opener' has only opener-length "
                     "ones, and is projected from those rather than from "
                     "the starter prior — Sean Newcomb averages 7 batters "
                     "when he starts, against a league prior of 22.5. "
                     "'reliever' pitches regularly but has not started. "
                     "'unknown' is a debut or a long layoff, and gets the "
                     "new-pitcher prior."),
            "Form": st.column_config.Column(
                width=58,
                help="How his STRIKEOUTS have been going, not how hard he "
                     "is throwing — his last four starts against his own "
                     "rate, in standard errors. Velocity and this one "
                     "disagree most of the time, which is the point: an "
                     "arm can be at full speed and still not be missing "
                     "bats. Measured over ~10,500 starts, a starter "
                     "flagged Hot beats one flagged Cold by 1.8 points of "
                     "strikeout rate on his NEXT start (z = +5.1) — real, "
                     "but a fifth of what it looks like before the "
                     "shared-baseline artifact is removed. Shown, never "
                     "fed to the model: recent form does not survive "
                     "alongside velocity, so using both would count the "
                     "same signal twice."),
            "Velo": st.column_config.Column(
                width=58,
                help="Fastball mph over his last three starts against his "
                     "own baseline — four-seam, sinker and cutter only, "
                     "because a curveball's speed moves for different "
                     "reasons. Measured over 4,599 cached starts: a "
                     "starter in the bottom decile of this against one in "
                     "the top differs by 2.2 points of strikeout rate, "
                     "about half a strikeout over 23 batters (r = 0.052, "
                     "p = 0.001, seasonal arc removed). The obvious "
                     "version of this marker — recent strikeout RATE — was "
                     "tested at the same time and is noise (p = 0.21).\n\n"
                     "Shown only. The model does not use it, and will not "
                     "until it has graded itself on live slates; the "
                     "Results tab is where that appears."),
            "BF": st.column_config.Column(
                width=58,
                help="Batters he is projected to face. Everything to the "
                     "right is built on this — a starter pulled in the "
                     "fourth cannot reach six strikeouts however good "
                     "his rate is."),
            "K": st.column_config.Column(
                width=58,
                help="Strikeouts projected, by walking the real batting "
                     "order: batter n is lineup slot n mod 9, so the "
                     "leadoff hitter is faced three times and the "
                     "nine-hitter twice, each at his own strikeout rate."),
            "Outs": st.column_config.Column(
                width=62,
                help="Outs recorded, fitted separately from batters faced "
                     "and graded against real innings pitched."),
            "IP": st.column_config.Column(
                width=52, help="The same number in innings — outs / 3."),
            "Line": st.column_config.Column(
                width=56,
                help="The strikeout line on the board for HIM, not a fixed "
                     "one shared by the slate. Taken from tonight's captured "
                     "odds where a price exists; where none was caught, it "
                     "is the line his own projected distribution puts "
                     "closest to a coin flip. The Src column says which."),
            "Over": st.column_config.Column(
                width=62,
                help="The model's chance he goes over THAT line. Not "
                     "colour-coded, and it should not be: every line sits "
                     "near its own pitcher's even-money point, so this "
                     "column clusters around 50% by construction and a "
                     "colour on it would be painting noise."),
            "Mkt": st.column_config.Column(
                width=58,
                help="What the book's price implies for the same line, "
                     "with the hold removed. Blank when no price was "
                     "captured for him."),
            "Edge": st.column_config.Column(
                width=62,
                help="Model minus market, at the same line. Coloured past "
                     "5 points either way — green where the model likes "
                     "the over, red the under. Most nights most rows are "
                     "blank or uncoloured, and that is the honest answer: "
                     "agreeing with the market is not an edge, it means "
                     "the model is reproducing public information "
                     "competently. Uncoloured whenever Books is under 4 — "
                     "the biggest numbers in this column are usually the "
                     "least real, because a thin alternate line is where "
                     "the market is pricing something the model cannot "
                     "see."),
            "Books": st.column_config.Column(
                width=58,
                help="How many sportsbooks were quoting this line when the "
                     "odds were captured. Dimmed zero means no price at all "
                     "— the line beside it is the model's own and there is "
                     "nothing here to bet against. Amber under 4 means one "
                     "or two shops posting an alternate, which is not a "
                     "consensus: Edge is left uncoloured on those rows "
                     "however large it looks. On 2026-09-13 Joe Ryan's only "
                     "captured line was 3.5 from three books, and the model "
                     "read it as a 21-point edge. A 132-start starter does "
                     "not get a 3.5 line unless three books know something "
                     "the model does not."),
        }
        for lab, col in PIT_PROPS.items():
            if lab == "Edge":
                continue          # configured by hand above
            pit_config[lab] = st.column_config.Column(
                width=76,
                help=("Chance he records more than "
                      f"{lab.split()[0]} "
                      + ("strikeouts" if lab.endswith("K") else
                         f"outs ({float(lab.split()[0]) / 3:.1f} innings)")))

        st.dataframe(pit_styled, width="stretch", hide_index=True,
                     height=min(560, 40 + 35 * len(pit_view)),
                     column_config=pit_config)

        thin_n = int((pit["starts_seen"] <= THIN_STARTS).sum()) \
            if "starts_seen" in pit else 0
        back_n = int(((pit.get("starts_seen", 0) == 0)
                      & (pit.get("career_starts", 0) >= 10)).sum()) \
            if "career_starts" in pit else 0
        note = (f' · {thin_n} sit at {THIN_STARTS} starts or fewer (amber), '
                f'so their numbers lean on the prior rather than on them'
                if thin_n else '')
        if back_n:
            note += (f' · {back_n} have a real career but no start in a '
                     f'year — sort by Rest to find them')
        # The bias warning, and it is measured rather than a hunch: across
        # 202 pitcher-nights on ten captured slates the model moves 0.67
        # strikeouts for every 1 the market moves (lab_k_spread.py). It
        # runs high on short-outing arms and low on aces, so its biggest
        # OVERs cluster at the lowest lines -- which is exactly where they
        # look most attractive. Counted from tonight's own rows rather
        # than asserted, so on a night with no such rows it says nothing.
        _low = 0
        if "Edge" in pit_view and "Books" in pit_view:
            _shown = (pit_view["Edge"].abs() >= 0.05) & (pit_view["Books"] >= MIN_BOOKS)
            _low = int((_shown & (pit_view["Edge"] > 0)
                        & (pit_view["Line"] <= 4.5)).sum())
        # What the running record says tonight's OWN probabilities are
        # worth. Computed from pitcher_row_log rather than asserted, and
        # attached to the rows it applies to rather than left in Results
        # where nobody reads it while looking at an edge.
        _band = ""
        _bpath = os.path.join(CACHE_DIR, "pitcher_row_log.csv")
        if "Over" in pit_view and os.path.exists(_bpath):
            try:
                _bl = pd.read_csv(_bpath).dropna(subset=["k_dist",
                                                         "strikeouts"])
            except Exception:
                _bl = pd.DataFrame()
            if len(_bl) and "k_dist" in _bl.columns:
                _pts = []
                for _, _r in _bl.iterrows():
                    _p = parse_pmf(_r["k_dist"])
                    if _p is None:
                        continue
                    for _ln in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
                        _pts.append((prob_over(_p, _ln),
                                     int(_r["strikeouts"] > _ln), _ln))
                # The line is carried through because the warning above the
                # table is specifically about LOW lines. Grading every
                # stored distribution at every line is what makes that
                # possible at all -- the slate was only ever priced at a
                # couple of them, and 2.5 was never one.
                _cd = pd.DataFrame(_pts, columns=["said", "hit", "line"])
                # Tonight's rows, binned the same way, so the note counts
                # real rows on this page rather than describing history.
                _tn = pit_view["Over"].dropna()
                _hot = _cd[(_cd.said >= .65) & (_cd.said < .8)]
                _n_tonight = int(((_tn >= .65) & (_tn < .8)).sum())
                if len(_hot) >= 40 and _n_tonight:
                    _gap = float(_hot.hit.mean() - _hot.said.mean())
                    _se = float(((_hot.said * (1 - _hot.said)).sum()) ** .5) \
                        / len(_hot)
                    if _gap < -2 * _se:
                        _band = (
                            f'<br><b style="color:var(--ink2)">'
                            f'{_n_tonight} row(s) tonight sit between 65% '
                            f'and 80%.</b> Over {len(_hot)} graded '
                            f'probabilities in that band the model has said '
                            f'{_hot.said.mean():.0%} and it has happened '
                            f'{_hot.hit.mean():.0%} — about '
                            f'{abs(_gap) * 100:.0f} points hot. An Edge '
                            f'computed from an overstated probability is '
                            f'overstated by the same amount, so treat a big '
                            f'number on one of those rows as roughly '
                            f'{abs(_gap) * 100:.0f} points smaller than it '
                            f'reads. Results has the full curve.')

                # ---- the low-line pill, written into the slot on top ----
                #
                # Same measurement as the footnote, a different slice of it,
                # and deliberately a different place on the page. The
                # footnote answers "what is this Edge actually worth" for
                # the rows it sits under. This answers "should I be careful
                # tonight" before he has scrolled anywhere.
                #
                # Restricted to lines at 3.5 and below AND to probabilities
                # the model likes, because that pair is what produces a big
                # printed OVER edge -- and it is the pair the model is
                # furthest wrong about: the 2026-09-14 calibration curve has
                # the model saying 73.8% and the thing happening 60.3% on
                # 2.5 and 3.5.
                #
                # The CAUSE is still open, and this marker deliberately does
                # not name one. lab_tto.py measured a real times-through-the-
                # order effect (z = -8.84) and then priced its effect on this
                # model at 0.2 points of probability -- the rate the model
                # applies is learned on starts that already contain the
                # decay, so it is absorbed rather than missing. Reporting the
                # gap without explaining it is the only honest thing the page
                # can do until the bad-and-short coupling has been tested.
                #
                # Counted from tonight's own rows and from the running
                # record, never asserted. On a slate with no low-line rows,
                # or before the log is deep enough to say anything, the slot
                # stays empty and the page looks exactly as it did before.
                _lo_hist = _cd[(_cd.line <= 3.5) & (_cd.said >= 0.55)]
                _lo_n = 0
                if "Line" in pit_view and "Over" in pit_view:
                    _lo_n = int(((pit_view["Line"] <= 3.5)
                                 & (pit_view["Over"] >= 0.55)).sum())
                if len(_lo_hist) >= 40 and _lo_n:
                    _lgap = float(_lo_hist.hit.mean() - _lo_hist.said.mean())
                    # Poisson-binomial: the variance of a sum of independent
                    # indicators is the sum of p(1-p), not n p̄(1-p̄). Each
                    # graded probability is its own coin.
                    _lse = (float(((_lo_hist.said * (1 - _lo_hist.said))
                                   .sum()) ** .5) / len(_lo_hist))
                    if _lgap < -2 * _lse:
                        _tip = (f"Over {len(_lo_hist)} graded probabilities "
                                f"at lines of 3.5 and below, the model has "
                                f"said {_lo_hist.said.mean():.0%} and it has "
                                f"happened {_lo_hist.hit.mean():.0%}. "
                                f"{_lo_n} row(s) tonight sit there. "
                                f"Results has the full curve.")
                        pit_warn_slot.markdown(
                            f'<div style="display:flex;justify-content:'
                            f'flex-end;margin:-10px 0 12px">'
                            f'<div title="{_tip}" style="display:inline-flex;'
                            f'align-items:center;gap:8px;border:1px solid '
                            f'var(--b1);background:var(--b1bg);'
                            f'color:var(--b1ink);border-radius:999px;'
                            f'padding:5px 13px;font-size:12px;'
                            f'font-weight:600;cursor:help">'
                            f'<span style="width:7px;height:7px;'
                            f'border-radius:50%;background:var(--b1);'
                            f'display:inline-block"></span>'
                            f'{_lo_n} low-line OVER'
                            f'{"s" if _lo_n != 1 else ""} tonight · '
                            f'the model runs {abs(_lgap) * 100:.0f} pts hot '
                            f'at 3.5 and below</div></div>',
                            unsafe_allow_html=True)
        _bias = (f'{_band}<br><b style="color:var(--ink2)">{_low} of tonight\'s '
                 f'coloured edges are OVERs at 4.5 or below.</b> That is the '
                 f'shape of a known bias, not a find: measured over 202 '
                 f'pitcher-nights the model moves 0.67 strikeouts for every '
                 f'1 the market moves, so it reads high on short-outing arms '
                 f'and low on aces. A large OVER at a low line is usually '
                 f'the market pricing a pitch limit or a bullpen game that '
                 f'the model cannot see.') if _low else ''
        html(f'<div style="margin-top:10px;font-size:12px;color:var(--ink3);'
             f'line-height:1.55">Click any header to sort{note}. Colour is '
             f'which quarter of tonight\'s starters he falls into, measured '
             f'on the whole slate so it keeps its meaning after you sort.'
             f'{_bias}'
             f'<br><b style="color:var(--ink2)">Outs</b> is how deep he '
             f'goes: 14.5 is getting through five innings, 17.5 through '
             f'six. Model only — the outs market is charged per game and '
             f'would double the odds bill, so these are graded against '
             f'real innings pitched rather than against a price.<br>'
             f'Sorting by <b style="color:var(--ink2)">Outs</b> against '
             f'<b style="color:var(--ink2)">K</b> is the useful one: they '
             f'do not agree, and a pitcher high in one and low in the '
             f'other is telling you something a single column cannot.'
             f'</div>')

        # ---- recent form ------------------------------------------------
        #
        # Nolan: "I need to check how he's been recently and how he is in
        # home starts and how many pitches has he been throwing in all his
        # starts."
        #
        # All three come out of the statcast pitcher caches, which the
        # DEPLOYED app cannot read -- they are 1.2 GB and untracked on
        # purpose. pitcher_form.py computes this on the machine that has
        # them and writes ~25 KB a night; see the .gitignore note.
        #
        # There is no versus-opponent block here and there should not be.
        # It was measured (lab_opponent.py): real overdispersion of about
        # 2.3 points of strikeout rate, z = +5.3, that does NOT survive a
        # split-half test -- r = +0.019 against a null of -0.001 +/- 0.030
        # over 869 pairings. Drift and clustering, not a trait. Half the
        # reason to write a lab script is to know what not to build.
        _form_path = os.path.join(CACHE_DIR, f"pitcher_form_{slate_date}.csv")
        _form = None
        if os.path.exists(_form_path):
            try:
                _form = pd.read_csv(_form_path)
            except Exception:
                _form = None

        if _form is not None and not _form.empty and "pitcher_id" in pit:
            html('<div style="font-size:15px;font-weight:640;color:var(--ink);'
                 'margin-top:26px">Recent form</div>'
                 '<div style="color:var(--ink3);font-size:12.5px;'
                 'margin-bottom:10px">His last ten starts — how deep he went, '
                 'how many pitches it took, and how it split home and '
                 'away.</div>')
            _have = set(_form["pitcher_id"].dropna().astype(int))
            _opts = [(int(r["pitcher_id"]), str(r["pitcher"]))
                     for _, r in pit.iterrows()
                     if pd.notna(r.get("pitcher_id"))
                     and int(r["pitcher_id"]) in _have]
            if _opts:
                _pick = st.selectbox(
                    "Pitcher", _opts, format_func=lambda o: o[1],
                    key="pit_form_pick", label_visibility="collapsed")
                _his = _form[_form["pitcher_id"] == _pick[0]].copy()
                _his = _his.sort_values("game_date", ascending=False)

                # The split summary first: it is the question asked, and
                # it needs its sample size beside it or it is a number
                # without a reason to believe it.
                _cards = ""
                for _flag, _lab in ((True, "At home"), (False, "On the road")):
                    _side = _his[_his["home"].astype(bool) == _flag]
                    if _side.empty:
                        continue
                    _bf = float(_side["bf"].sum())
                    _kr = float(_side["k"].sum()) / _bf if _bf else float("nan")
                    _cards += (
                        f'<div class="sp-find" style="padding:12px 14px">'
                        f'<div class="kind" style="color:var(--ink3)">'
                        f'{_lab}</div>'
                        f'<div class="fig">{_side["pitches"].mean():.0f}</div>'
                        f'<div class="unit">pitches a start</div>'
                        f'<div class="txt">{len(_side)} of these ten · '
                        f'<b>{_side["bf"].mean():.1f}</b> batters · '
                        f'<b>{_kr:.1%}</b> struck out</div></div>')
                if _cards:
                    html(f'<div class="sp-read">{_cards}</div>')

                _tbl = _his[["game_date", "opp", "home", "pitches", "bf", "k",
                             "last_inning"]].copy()
                _tbl["home"] = np.where(_tbl["home"].astype(bool), "vs", "at")
                _tbl["K%"] = _tbl["k"] / _tbl["bf"].replace(0, pd.NA)
                _tbl = _tbl.rename(columns={
                    "game_date": "Date", "opp": "Opp", "home": "H/A",
                    "pitches": "Pitches", "bf": "BF", "k": "K",
                    "last_inning": "Thru inn"})
                st.dataframe(
                    _tbl.style.format({"K%": "{:.1%}"}, na_rep="—"),
                    width="stretch", hide_index=True,
                    height=min(420, 40 + 35 * len(_tbl)))

                html('<div style="margin-top:8px;font-size:12px;'
                     'color:var(--ink3);line-height:1.55">'
                     'The home/away split is shown as a <b '
                     'style="color:var(--ink2)">record</b>, not a signal. It '
                     'has real sample behind it — across the cached starters '
                     'the lighter of a pitcher\'s two halves still holds '
                     'about 46 starts — but whether it PREDICTS anything has '
                     'not been tested, and the model does not use it.<br>'
                     'There is deliberately no “against this opponent” '
                     'column. That one was tested: the residual after his '
                     'own rate and the opponent\'s own rate is real but does '
                     'not repeat (split-half r = +0.02 against a null of '
                     '−0.00 ± 0.03 over 869 pairings), so a past number '
                     'against tonight\'s team tells you nothing about '
                     'tonight.</div>')
        elif "k_dist" in pit.columns:
            html('<div style="margin-top:22px;font-size:12px;'
                 'color:var(--ink3);line-height:1.55">'
                 '<b style="color:var(--ink2)">Recent form</b> is not in this '
                 'slate. It comes from <code>pitcher_form.py</code>, which '
                 'reads the statcast caches — those are 1.2 GB and stay out '
                 'of git, so the summary has to be written on the machine '
                 'that has them and committed as its own small file. Run '
                 '<code>python pitcher_form.py</code> after a predict and it '
                 'appears here.</div>')

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


# ------------------------------------------------------------ game lines
if SHOW_GAME_LINES:
    with tab_lines:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink)">'
             'Game lines</div><div style="color:var(--ink3);font-size:12.5px;'
             'margin-bottom:14px">Who wins, by how much, and who does the '
             'scoring. Built by adding up the same hitters shown on every '
             'other tab — no separate team model to disagree with them.</div>')

        games_path = os.path.join(CACHE_DIR, f"teams_{slate_date}.csv")
        games_df = None
        if os.path.exists(games_path):
            try:
                games_df = pd.read_csv(games_path)
            except Exception:
                games_df = None

        if games_df is None or games_df.empty:
            html('<div class="sp-empty">No game predictions for this slate.<br>'
                 '<span style="color:var(--ink3);font-size:12px">'
                 'Written by <code>predict_slate.py</code> alongside the hitter '
                 'props. A slate predicted before this tab existed will not '
                 'have them.</span></div>')
        else:
            if "start_time_utc" in games_df:
                games_df["start"] = pd.to_datetime(
                    games_df["start_time_utc"], utc=True, errors="coerce"
                ).dt.tz_convert("America/New_York")
                games_df = games_df.sort_values("start", na_position="last")

            # Who the model expects to do the scoring. exp_runs is p_run x
            # expected_pa, written by predict_slate -- the same numbers that
            # were summed to produce the team total above, so the three names
            # under a card are literally the biggest terms in that card's score.
            # An older slate file predates the column; the card still draws,
            # just without the bottom half.
            have_scorers = "exp_runs" in df.columns

            def scorers(pk, team, n=3):
                if not have_scorers:
                    return ""
                side = df[(df["game_pk"] == pk) & (df["team"] == team)]
                side = side.dropna(subset=["exp_runs"]).nlargest(n, "exp_runs")
                if side.empty:
                    return ""
                return " · ".join(
                    f'<b>{str(r["name"]).split()[-1] if pd.notna(r["name"]) else "?"}</b>'
                    f' {r["exp_runs"]:.2f}'
                    for _, r in side.iterrows())

            cards = ""
            for r in games_df.to_dict("records"):
                pk = int(r["game_pk"])
                away, home = r.get("away_team", "?"), r.get("home_team", "?")
                ap, hp = float(r.get("away_win_prob", 0.5)), float(r.get("home_win_prob", 0.5))
                ar, hr_ = float(r.get("away_runs", 0)), float(r.get("home_runs", 0))
                clock = fmt_clock(r.get("start")) if "start" in r else ""

                # A lineup the model could only partly fill scores low for a
                # reason that has nothing to do with the teams. Say so on the
                # card rather than letting it read as a projection.
                thin = ""
                missing = [t for t, k in ((away, "away_hitters"), (home, "home_hitters"))
                           if pd.notna(r.get(k)) and int(r[k]) < 9]
                if missing:
                    thin = (f'<div class="sp-thin">Lineup incomplete for '
                            f'{" and ".join(missing)} — that side\'s runs are '
                            f'low by roughly the missing share.</div>')

                body = ""
                if have_scorers:
                    a_s, h_s = scorers(pk, away), scorers(pk, home)
                    if a_s or h_s:
                        body = (f'<div class="sp-sc">Expected to score'
                                f'<div class="who">{away} &nbsp;{a_s}</div>'
                                f'<div class="who">{home} &nbsp;{h_s}</div></div>')

                cards += (
                    f'<div class="sp-gc">'
                    f'<div class="hd"><span>{away} at {home}</span>'
                    f'<span>{clock}</span></div>'
                    f'<div class="sd">'
                    f'<div><div class="tm">{away}</div>'
                    f'<div class="rn{"" if ar >= hr_ else " dim"}">{ar:.1f}</div></div>'
                    f'<div style="text-align:right"><div class="tm">{home}</div>'
                    f'<div class="rn{"" if hr_ >= ar else " dim"}">{hr_:.1f}</div></div>'
                    f'</div>'
                    f'<div class="sp-bar"><i class="a" style="width:{ap * 100:.1f}%"></i>'
                    f'<i class="h" style="width:{hp * 100:.1f}%"></i></div>'
                    f'<div class="sp-wp"><span>{away} <b>{ap:.0%}</b></span>'
                    # The flanks already carry the split the bar is showing, so
                    # the middle carries the other number people want off a
                    # game card: how many runs total.
                    f'<span style="color:var(--ink3)">{ar + hr_:.1f} total</span>'
                    f'<span><b>{hp:.0%}</b> {home}</span></div>'
                    f'{body}{thin}</div>')
            html(f'<div class="sp-gl">{cards}</div>')

            totals = games_df["home_runs"] + games_df["away_runs"]
            edge = games_df[["home_win_prob", "away_win_prob"]].max(axis=1)
            html(f'<div class="sp-kpi">'
                 f'<div><span class="v">{len(games_df)}</span>'
                 f'<span class="k">Games projected</span></div>'
                 f'<div><span class="v">{totals.mean():.1f}</span>'
                 f'<span class="k">Runs per game, both sides</span></div>'
                 f'<div><span class="v">{edge.max():.0%}</span>'
                 f'<span class="k">Strongest lean on the slate</span></div>'
                 f'<div><span class="v">'
                 f'{int((edge > 0.60).sum())}</span>'
                 f'<span class="k">Games leaning past 60%</span></div></div>')

            html('<div style="font-size:12px;color:var(--ink3);line-height:1.6">'
                 'A projected score is the <b style="color:var(--ink2)">average</b> '
                 'of how the game goes, not a prediction of the final. The win '
                 'probability is what matters, and it is much closer to even than '
                 'the run gap looks: a team projected a full run better still '
                 'loses close to four times in ten, because baseball scoring is '
                 'lumpy — one big inning swings a game and the model knows it. '
                 'The names are the model\'s biggest expected run scorers, in '
                 'runs, and they add up to the team total above them.<br>'
                 'The starting pitcher is in these numbers through each hitter\'s '
                 'matchup; there is no separate pitcher term.</div>')


# --------------------------------------------------- pitchers vs market
def _render_market_block():
    """
    The strikeout market check, moved off the Game Lines tab.

    It was there because it was the only thing there; it is a comparison
    of PITCHER strikeout props, so it belongs under the pitchers. Kept as
    a function so it renders at the bottom of that tab, below the props
    it is checking, rather than above them.
    """
    html('<div style="font-size:15px;font-weight:640;color:var(--ink);'
         'margin-top:26px">Against the market</div>'
         '<div style="color:var(--ink3);font-size:12.5px;margin-bottom:14px">'
         'The same strikeout numbers, priced against the sportsbooks. '
         'Biggest disagreements first.</div>')

    # Written by compare_market.py, which owns the name matching between
    # the odds feed ("Ronald Acuna Jr.") and this project's tables. The
    # dashboard reads the answer rather than redoing the join, so the two
    # can never disagree about who a player is.
    lines_path = os.path.join(CACHE_DIR, f"market_compare_{slate_date}.csv")
    lines_df = None
    if os.path.exists(lines_path):
        try:
            lines_df = pd.read_csv(lines_path)
        except Exception:
            lines_df = None

    if lines_df is None or lines_df.empty:
        html('<div class="sp-empty">No market lines for this slate.<br>'
             '<span style="color:var(--ink3);font-size:12px">'
             'Capture them before first pitch with '
             '<code>python -m data.odds_lines ' + str(slate_date) + '</code>, '
             'then run <code>python compare_market.py ' + str(slate_date) +
             '</code>.<br>They cannot be fetched after the games.</span></div>')
    else:
        lines_df = lines_df.reindex(
            lines_df["edge"].abs().sort_values(ascending=False).index)
        n_books_total = int(lines_df["n_books"].max()) if "n_books" in lines_df else 0

        rows = ""
        for r in lines_df.to_dict("records"):
            edge = r.get("edge")
            # Coloured by which side the model prefers, and only once the
            # gap is big enough to be worth a second look. Books and models
            # differ by a point or two on noise alone.
            if pd.isna(edge):
                colour, label = "var(--ink3)", "—"
            elif edge > 0.05:
                colour, label = "var(--b4)", f"OVER by {edge:.1%}"
            elif edge < -0.05:
                colour, label = "var(--b1ink)", f"UNDER by {abs(edge):.1%}"
            else:
                colour, label = "var(--ink2)", "agree"
            # How much history is behind the model's side of the
            # disagreement. Under twenty starts the model is partly
            # quoting a prior, and a prior disagreeing with five books is
            # not an edge -- it is the model admitting it does not know.
            # The name carries the warning because the name is what gets
            # read; the two columns say how thin, and how many strikeouts
            # the model is actually projecting behind the percentage.
            seen = r.get("starts_seen")
            starts = int(seen) if pd.notna(seen) else None
            light = starts is not None and starts < THIN_STARTS_MARKET
            exp_k = r.get("expected_k")
            rows += (
                f'<tr><td style="font-weight:560;'
                f'color:{"var(--warn)" if light else "var(--ink)"}">'
                f'{r.get("pitcher", r.get("player"))}'
                f'{"*" if light else ""}</td>'
                f'<td style="color:var(--ink2)">{r.get("team","")} '
                f'<span style="color:var(--ink3)">vs {r.get("opponent","")}</span></td>'
                f'<td style="color:var(--ink);text-align:center;'
                f'font-weight:560">{r.get("line"):g}</td>'
                f'<td style="text-align:center">{pct(r.get("model_prob"))}</td>'
                f'<td style="text-align:center">{pct(r.get("market_prob"))}</td>'
                f'<td style="color:{colour};text-align:center;font-weight:560">'
                f'{label}</td>'
                f'<td style="color:var(--ink2);text-align:center;'
                f'font-variant-numeric:tabular-nums">'
                f'{f"{exp_k:.1f}" if pd.notna(exp_k) else "—"}</td>'
                f'<td style="text-align:center;font-variant-numeric:tabular-nums;'
                f'color:{"var(--warn)" if light else "var(--ink3)"}">'
                f'{starts if starts is not None else "—"}</td>'
                f'<td style="color:var(--ink3);text-align:center">'
                f'{_int_or(r.get("n_books"))}</td></tr>')

        html('<table class="plain"><thead><tr><th>Pitcher</th><th>Matchup</th>'
             '<th style="text-align:center">Line</th>'
             '<th style="text-align:center">Model</th>'
             '<th style="text-align:center">Market</th>'
             '<th style="text-align:center">Disagreement</th>'
             '<th style="text-align:center">Exp K</th>'
             '<th style="text-align:center">Starts</th>'
             '<th style="text-align:center">Books</th>'
             '</tr></thead><tbody>' + rows + '</tbody></table>')

        gap = lines_df["edge"].abs().mean()
        signed = lines_df["edge"].mean()
        html(f'<div class="sp-kpi" style="margin-top:18px">'
             f'<div><span class="v">{len(lines_df)}</span>'
             f'<span class="k">Lines priced</span></div>'
             f'<div><span class="v">{gap:.1%}</span>'
             f'<span class="k">Average disagreement</span></div>'
             f'<div><span class="v">{signed:+.1%}</span>'
             f'<span class="k">Signed — model vs market</span></div>'
             f'<div><span class="v">{n_books_total}</span>'
             f'<span class="k">Most books on a line</span></div></div>')

        note = ""
        if abs(signed) > 0.03:
            note = (f' The model sits systematically '
                    f'{"above" if signed > 0 else "below"} the market, which '
                    f'is a calibration difference rather than an edge.')
        html(f'<div style="margin-top:6px;font-size:12px;color:var(--ink3);'
             f'line-height:1.55">Model probabilities are read at the '
             f'<b style="color:var(--ink2)">book\'s</b> line, not at a fixed '
             f'5.5 — books price each pitcher differently.{note}<br>'
             f'Agreeing with the market is not an edge; it means the model '
             f'is reproducing public information competently. An edge is '
             f'disagreement that turns out to be right, which only '
             f'<code>score_slate.py</code> can tell you.<br>'
             f'<b style="color:var(--warn)">Amber names</b> have fewer than '
             f'{THIN_STARTS_MARKET} starts in the last 12 months. Their '
             f'model number is partly the shrinkage prior, so a big gap '
             f'there is the model saying it does not know this pitcher, '
             f'not that it has found something the books missed. '
             f'<b style="color:var(--ink2)">Exp K</b> is the strikeout '
             f'total behind the percentage.</div>')


# Re-entering a tab context appends to it. The market block has to be
# defined before it can be called, and it is defined below the pitcher
# tab because it used to live somewhere else -- so it is called here
# rather than moved, which keeps the diff to the part that changed.
with tab_pitch:
    _render_market_block()


# ---------------------------------------------------------- all hitters
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
    # A tick has to resolve to a PLAYER, not to "row 14 of whatever is on
    # screen". The obvious way to do that is to put player_id on the index
    # -- and it is wrong: on a doubleheader a player has TWO rows, the
    # index is no longer unique, and pandas refuses to style a frame with a
    # non-unique index at all. The table dies, not just the checkbox.
    #
    # So the frame keeps its unique row index and player_id travels
    # alongside it. Everything below maps through this.
    row_pid = (df["player_id"] if "player_id" in df.columns
               else pd.Series(df.index, index=df.index))

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

        # The Save column goes in BEFORE styling, and gets no style of its
        # own. st.data_editor applies pandas.Styler colours to columns that
        # are non-editable -- which is every column here except this one --
        # so the checkbox and the green/yellow/red banding coexist. That is
        # not obvious from the API and is the reason this tab did not have
        # to give anything up to gain a checkbox.
        view_pid = row_pid.reindex(view.index)
        view.insert(0, "Save",
                    view_pid.isin(st.session_state.sp_picks).to_numpy())

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
            "Save": st.column_config.CheckboxColumn(
                "Save", width=52, pinned=True,
                help=f"Tick to add this hitter to your team in the left "
                     f"panel. Up to {MAX_PICKS}."),
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

        # The widget key carries the slate AND the search text on purpose.
        #
        # st.data_editor stores a user's edits keyed by ROW POSITION, not by
        # index. Type into the search box and row 3 becomes a different
        # hitter, so a stale edit would silently tick the wrong man. Making
        # the key change with the row set throws that state away whenever it
        # could go stale. Nothing is lost: the ticks live in sp_picks and are
        # rebuilt into the Save column above on every run.
        #
        # The nonce on the end is how a refused tick gets un-ticked. A
        # widget's stored edits cannot be deleted while it exists, so the
        # move is to change the key -- that builds a NEW editor whose Save
        # column comes straight from sp_picks, and the orphaned state is
        # collected. Without it, ticking a ninth hitter would leave a box
        # ticked for a hitter who is not on the list, which is the worst
        # kind of bug: the screen disagreeing with the data.
        st.session_state.setdefault("sp_grid_nonce", 0)
        editor_key = (f"sp_grid_{slate_date}_{query.strip().lower()}"
                      f"_{st.session_state.sp_grid_nonce}")
        edited = st.data_editor(
            styled, width="stretch", hide_index=True, height=620,
            column_config=config, num_rows="fixed",
            disabled=[c for c in view.columns if c != "Save"],
            key=editor_key)

        # dict.fromkeys rather than set(): a doubleheader player is ticked
        # on both his rows and must come back once, in table order.
        edited_pid = row_pid.reindex(edited.index)
        ticked = list(dict.fromkeys(
            int(p) for p in edited_pid[edited["Save"].fillna(False).to_numpy()]))
        previous = list(st.session_state.sp_picks)
        # Hitters filtered out by the search box are not on screen and so
        # cannot have been unticked -- they must survive the round trip.
        on_screen = set(view_pid)
        off_screen = [p for p in previous if p not in on_screen]
        kept = [p for p in previous if p in ticked]
        added = [p for p in ticked if p not in previous]
        picks = off_screen + kept + added

        # Over the cap, the NEWEST tick is the one refused. Dropping an
        # earlier pick instead would mean a click quietly deleting a hitter
        # somewhere off screen.
        overflow = picks[MAX_PICKS:]
        picks = picks[:MAX_PICKS]
        if picks != previous:
            _write_picks(picks)
        if overflow:
            st.session_state.sp_grid_nonce += 1
            st.session_state.sp_full_msg = True
            st.rerun()

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

# Outside the tab block on purpose. st.sidebar writes to its own container
# wherever it is called, and every tab body runs on every rerun, so this
# renders whichever tab is on screen. Placed after the editor so it shows
# the ticks from THIS run rather than the previous one.
render_picks_panel(df, props, cuts)


# -------------------------------------------------------------- results
# ------------------------------------------------------------- bet ready
#
# This tab RENDERS slips; it no longer builds them. slips.py does that,
# before first pitch, and writes cache/slips_{date}.csv.
#
# The reason is Nolan's question "which slips hit last night", which could
# not be answered: the slips only ever existed as pixels. Rebuilding them
# the next morning would have graded whatever today's code and today's
# slate file produce, not what was on the board. That is the rule this
# project keeps re-learning and .gitignore states twice -- a number you
# want to grade later has to be committed before the game starts.
#
# Reading the committed file rather than rebuilding also means the slip on
# screen and the slip in slip_log.csv are the same object, and that there
# is exactly ONE implementation of what a slip is. Building it in two
# places is how the card and the copy text drifted apart once already.
#
# A slate predicted before slips.py existed has no file, so the tab falls
# back to building live and says so.
with tab_bet:
    try:
        import slips as _S
    except Exception as _e:            # pragma: no cover - deploy guard
        _S = None
        st.error(f"slips.py could not be imported ({_e}). The Bet ready tab "
                 f"needs it — it holds the definition of what a slip is.")

    if _S is not None:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink)">'
             'Bet ready</div><div style="color:var(--ink3);font-size:12.5px;'
             'margin-bottom:14px">Four slips per first-pitch window — one '
             'mixed, then one each of pure safe, medium and long shot. Props '
             'mix inside every slip; risk levels only mix in the first.</div>')

        _slip_path = os.path.join(CACHE_DIR, f"slips_{slate_date}.csv")
        _committed = os.path.exists(_slip_path)
        _sl = None
        if _committed:
            try:
                _sl = pd.read_csv(_slip_path)
            except Exception:
                _sl, _committed = None, False
        if _sl is None or _sl.empty:
            _committed = False
            _sl = _S.build_slips(df, pit, _S.market_lines(slate_date,
                                                          CACHE_DIR), legs=3)

        if _sl is None or _sl.empty:
            html('<div class="sp-empty">No slips for this slate.<br>'
                 '<span style="color:var(--ink3);font-size:12px">'
                 'Run <code>python slips.py</code> after a predict, or check '
                 'that the slate has start times and probability '
                 'columns.</span></div>')
        else:
            if not _committed:
                html('<div style="margin:-6px 0 12px;font-size:12px;'
                     'color:var(--b2ink);line-height:1.5">These are built '
                     'live from the slate, not committed. Nothing graded '
                     'them and nothing will — run '
                     '<code>python slips.py</code> before first pitch and '
                     'the slips become a record that '
                     '<code>score_slips.py</code> can score afterwards.'
                     '</div>')

            TIER_COLOUR = {"MIXED": "var(--ink2)", "SAFE": "var(--b4)",
                           "MEDIUM": "var(--accent)", "LOTTO": "var(--b2ink)"}
            TIER_WHY = {
                "MIXED": "one of each, so a long shot rides along",
                "SAFE": "every leg a near-certainty",
                "MEDIUM": "live, and well clear of the field",
                "LOTTO": "long odds the model rates far above a typical player",
            }

            _cards, _copy = "", []
            for _w, _wg in _sl.groupby("window"):
                _n_games = int(_wg["n_games"].iloc[0])
                _solo = bool(_wg["solo"].iloc[0])
                _t0 = pd.to_datetime(_wg["window_start"].iloc[0], utc=True,
                                     errors="coerce")
                _t1 = pd.to_datetime(_wg["window_end"].iloc[0], utc=True,
                                     errors="coerce")
                if pd.notna(_t0):
                    _t0 = _t0.tz_convert("America/New_York")
                if pd.notna(_t1):
                    _t1 = _t1.tz_convert("America/New_York")
                _label = (fmt_clock(_t0) if _t0 == _t1
                          else f"{fmt_clock(_t0)} – {fmt_clock(_t1)}")

                _copy.append(f'{_label}  ({_n_games} game'
                             f'{"s" if _n_games != 1 else ""}'
                             f'{" — every leg correlated" if _solo else ""})')
                _tier_html = ""
                for _tname in _S.TIER_ORDER:
                    _g = _wg[_wg["tier"] == _tname]
                    if _g.empty:
                        continue
                    _sp = float(_g["slip_p"].iloc[0])
                    _colour = TIER_COLOUR.get(_tname, "var(--ink2)")
                    _copy.append(f'  {_tname} slip  ({len(_g)} legs, all hit '
                                 f'{_sp:.1%})')
                    _rows = ""
                    for _, _b in _g.iterrows():
                        _slot = _b.get("lineup_slot")
                        _bat = (f' · bats {ORDINAL.get(int(_slot), int(_slot))}'
                                if pd.notna(_slot) else "")
                        if _b["kind"] == "arm":
                            _e = float(_b["edge"])
                            _ecol = ("var(--b4)" if _e >= 0.05 else
                                     "var(--b1ink)" if _e <= -0.05
                                     else "var(--ink3)")
                            _note = (f'market {float(_b["base"]):.0%} · '
                                     f'<b style="color:{_ecol}">{_e:+.1%}</b>'
                                     f' — a measured edge')
                            _tail = (f'   market {float(_b["base"]):.0%}, '
                                     f'{_e:+.1%}')
                        else:
                            _note = (f'{float(_b["lift"]):.1f}× the '
                                     f'{float(_b["base"]):.0%} slate median')
                            _tail = f'   {float(_b["lift"]):.1f}x field'
                        _clash = bool(_b.get("clash")) and not _solo
                        _same = (' · <span style="color:var(--warn)">same '
                                 'game</span>' if _clash else "")
                        _rows += (
                            f'<div style="display:flex;gap:9px;'
                            f'align-items:baseline;padding:3px 0 3px 12px">'
                            f'<span style="font-weight:600;color:var(--ink);'
                            f'font-size:12.5px">{_b["name"]}</span>'
                            f'<span style="color:var(--ink3);font-size:11.5px">'
                            f'{_b["team"]} v {_b["opponent"]}{_bat} · '
                            f'{_b["prop"]}</span>'
                            f'<span style="margin-left:auto;font-weight:640;'
                            f'color:var(--ink);font-size:12.5px">'
                            f'{pct(float(_b["p"]))}</span></div>'
                            f'<div style="font-size:11px;color:var(--ink3);'
                            f'padding-left:12px">{_note}{_same}</div>')
                        _copy.append(
                            f'    {_b["name"]} ({_b["team"]}) — '
                            f'{_b["prop"]} — {pct(float(_b["p"]))}{_tail}'
                            + ("   [same game]" if _clash else ""))
                    _tier_html += (
                        f'<div style="padding:9px 0;border-top:1px solid '
                        f'var(--line)">'
                        f'<div style="display:flex;gap:10px;'
                        f'align-items:baseline">'
                        f'<span style="color:{_colour};font-weight:640;'
                        f'letter-spacing:.08em;font-size:11px">{_tname}</span>'
                        f'<span style="color:var(--ink3);font-size:11px">'
                        f'{len(_g)} legs · {TIER_WHY.get(_tname, "")}</span>'
                        f'<span style="margin-left:auto;color:var(--ink2);'
                        f'font-size:11.5px">all hit '
                        f'<b style="color:var(--ink)">{_sp:.1%}</b></span>'
                        f'</div>{_rows}</div>')
                    _copy.append("")
                _cards += (
                    f'<div class="sp-find" style="padding:13px 15px">'
                    f'<div class="kind" style="color:var(--accent)">{_label}'
                    f'</div><div class="txt" style="margin:2px 0 2px">'
                    f'{_n_games} game{"s" if _n_games != 1 else ""} in this '
                    f'window'
                    + ('<span style="color:var(--warn)"> — one game, so every '
                       'leg here is correlated</span>' if _solo else '')
                    + f'</div>{_tier_html}</div>')

            html(f'<div class="sp-read">{_cards}</div>')

            if bool(_sl["clash"].any()) or bool(_sl["solo"].any()):
                html('<div style="margin:4px 0 14px;font-size:12.5px;'
                     'color:var(--b2ink);line-height:1.55">'
                     '<b>Some slips reuse a game.</b> That happens when a '
                     'window has fewer games than legs, and it matters twice '
                     'over. Two legs from one lineup share a pitcher, a park '
                     'and a night, so the true chance of both landing is '
                     'HIGHER than the product — which means the <i>all '
                     'hit</i> figure beside such a slip is too LOW, and a '
                     'book pricing the legs as independent is paying you '
                     'less than the correlation is worth. Flagged rather '
                     'than dropped, because a one-game window has nothing '
                     'to swap in.</div>')

            html('<div style="font-size:13px;font-weight:620;color:var(--ink);'
                 'margin-top:6px">Copy</div>')
            st.code("\n".join(_copy).strip(), language=None)

            _exp = (_sl.drop_duplicates(["window", "tier"])
                       .groupby("tier")["slip_p"].sum())
            html(f'<div style="margin-top:8px;font-size:12px;'
                 f'color:var(--ink3);line-height:1.55">'
                 f'<b style="color:var(--ink2)">What to expect tonight.</b> '
                 f'Adding up the slip probabilities: '
                 + " · ".join(f'{t} <b style="color:var(--ink2)">'
                              f'{_exp.get(t, 0):.2f}</b>'
                              for t in _S.TIER_ORDER if t in _exp.index)
                 + f' slips expected to land, out of '
                 f'{_sl.groupby(["window", "tier"]).ngroups}. A LOTTO slip is '
                 f'expected about once every fifty nights, so a long run of '
                 f'nothing there is the correct-looking result rather than a '
                 f'failure — which is why the Results tab reports expected '
                 f'against actual and not a win-loss record.<br>'
                 f'<b style="color:var(--ink2)">Why the slips are '
                 f'separated.</b> A 19% leg on the same slip as a 75% leg '
                 f'does not make it riskier in proportion — it decides it. '
                 f'The long shot does all the work and the near-certainty '
                 f'mostly shortens the price. Parlays are built within a '
                 f'risk level; the mixing here is across prop TYPES.<br>'
                 f'<b style="color:var(--ink2)">SAFE</b> is ranked by raw '
                 f'probability; <b style="color:var(--ink2)">MEDIUM</b> and '
                 f'<b style="color:var(--ink2)">LOTTO</b> by lift over the '
                 f'slate median for that prop, because down there raw '
                 f'probability just re-sorts by which prop has the higher '
                 f'base rate — a fact about the prop, not the player. A 12% '
                 f'home run is the field; a 39% home run is a different '
                 f'claim.<br>'
                 f'<b style="color:var(--ink2)">Each leg is a different '
                 f'player</b>, because many of these props are NESTED — two '
                 f'hits implies one hit, a home run implies H+R+RBI 0.5 — so '
                 f'a slip taking the same bat three times would be one bet '
                 f'wearing three labels, worth its smallest leg rather than '
                 f'the product.<br>'
                 f'<b style="color:var(--ink2)">Every hitter number here is '
                 f'model-only.</b> There is no hitter-prop odds capture — '
                 f'player props are charged per event and the free tier is '
                 f'500 credits a month. Lift is the honest stand-in for '
                 f'market disagreement and is not the same thing. A pitcher '
                 f'leg carries the real number, and at most one goes in a '
                 f'slip so a single measured edge cannot look like '
                 f'three.</div>')


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
        for col in ("n", "base_rate", "mean_pred", "brier", "auc"):
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
            # Ranking, pooled the same way. This is the OTHER half of being
            # right and it was invisible on this page until now: a model
            # that quotes the league rate to every hitter matches the slate
            # total exactly and has an AUC of 0.50. Calibration alone
            # cannot tell those two apart.
            auc = ((g["auc"] * g["n"]).sum() / w
                   if "auc" in g and g["auc"].notna().any() else float("nan"))
            return pd.Series({"n": w, "said": said, "did": did,
                              "auc": auc, "skill": skill, "sigma": sigma})

        # Column list is explicit: `groupby(...).apply()` over the whole
        # frame warns (and in newer pandas will error) about operating on
        # the grouping column itself.
        by_prop = (log.groupby("label", sort=False)
                      [["n", "mean_pred", "base_rate", "brier", "auc"]]
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
            # Left deliberately uncoloured above 0.50. Shading it would
            # invite reading a gap between 0.57 and 0.62 that four slates
            # cannot support. The one threshold that means anything is the
            # coin flip, so that is the only one marked.
            f'<td style="color:{"var(--b1ink)" if r.auc < 0.5 else "var(--ink2)"};'
            f'font-variant-numeric:tabular-nums">'
            f'{"—" if pd.isna(r.auc) else f"{r.auc:.3f}"}</td>'
            f'<td style="color:{skill_color(r.skill)};font-weight:560">'
            f'{r.skill * 100:+.2f}%</td></tr>'
            for r in by_prop.itertuples())
        html('<table class="plain"><thead><tr><th>Prop</th><th>Graded</th>'
             '<th>Model said</th><th>Actually happened</th>'
             '<th>Miss</th><th>Ranking</th><th>Edge</th></tr></thead><tbody>'
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

        # ---- how to read the table above ------------------------------
        #
        # The question this answers -- "the model said 11%, and the guy
        # either homered or he didn't, so how is that graded?" -- is the
        # single most reasonable confusion this page produces, and it comes
        # back every time you look at it after a gap. So the answer lives
        # here rather than in a document somewhere else, and it is built
        # from the LIVE numbers so the worked example never goes stale.
        #
        # Collapsed by default: the numbers are the point of the tab, and
        # an open explainer would push them under the fold.
        with st.expander("How to read this"):
            # Prefer home runs for the example -- a rare event over a big
            # pile is the clearest case. Fall back to whatever has the most
            # rows if the HR prop is not in the log.
            ex = by_prop[by_prop["label"] == "at least 1 home run"]
            if ex.empty:
                ex = by_prop.nlargest(1, "n")
            ex = ex.iloc[0]
            n_ex = int(ex["n"])
            expected, actual = ex["said"] * n_ex, ex["did"] * n_ex

            html(
                f'<div style="font-size:13px;line-height:1.65;color:var(--ink2);'
                f'max-width:760px">'

                f'<p style="margin:0 0 12px"><b style="color:var(--ink)">'
                f'A single prediction cannot be checked.</b> The model said '
                f'one hitter 24% and another 4%. Each of them either homered '
                f'or did not. Neither number was wrong. So the grading never '
                f'looks at one prediction — it looks at a pile of them.</p>'

                f'<p style="margin:0 0 6px"><b style="color:var(--ink)">'
                f'Add the percentages up.</b> That is how many the model '
                f'expected to see. For <i>{ex["label"]}</i>:</p>'
                f'<div style="background:var(--card2);border:1px solid var(--line);'
                f'border-radius:9px;padding:12px 14px;margin:0 0 12px;'
                f'font-variant-numeric:tabular-nums">'
                f'{n_ex:,} hitters averaging {pct(ex["said"])} '
                f'&nbsp;→&nbsp; <b style="color:var(--ink)">'
                f'{expected:.0f} expected</b><br>'
                f'What actually happened &nbsp;→&nbsp; '
                f'<b style="color:var(--ink)">{actual:.0f} of them</b></div>'
                f'<p style="margin:0 0 14px">That is the whole '
                f'<i>Model said</i> / <i>Actually happened</i> pair — those '
                f'two counts, divided by {n_ex:,}. The model committed to '
                f'{expected:.0f} across the slate and {actual:.0f} showed '
                f'up.</p>'

                f'<p style="margin:0 0 12px"><b style="color:var(--ink)">'
                f'But matching the total is only half of being right.</b> '
                f'Imagine a lazy model that quoted {pct(ex["said"])} to '
                f'<i>every</i> hitter — the best power hitter in baseball '
                f'and a backup catcher, identical. It would also expect '
                f'{expected:.0f}. It would look perfect on that column. And '
                f'it would be worthless.</p>'

                f'<p style="margin:0 0 12px">So the second question is '
                f'whether the hitters it rated high actually did it more '
                f'often than the ones it rated low. That is '
                f'<b style="color:var(--ink)">Ranking</b>: pick one hitter '
                f'who did it and one who did not, at random — it is how '
                f'often the model had given the right one the bigger '
                f'number. 0.50 is a coin flip, and the lazy model above '
                f'scores exactly 0.50 however well it matches the total.</p>'

                f'<p style="margin:0 0 12px"><b style="color:var(--ink)">'
                f'Edge</b> is both halves in one number: how much of the '
                f'guesswork the model removes against quoting the league '
                f'rate to everybody. <b style="color:var(--ink)">Miss</b> '
                f'asks whether the gap could just be luck — it is the gap '
                f'divided by the random swing you would expect on this many '
                f'hitters, so under 2σ means the sample is small, not that '
                f'the model is off.</p>'

                f'<p style="margin:0;color:var(--ink3)">Both columns have to '
                f'be right. A model can match every total and rank nothing, '
                f'or rank perfectly while being systematically too high. '
                f'Neither one alone is a working model.</p>'
                f'</div>')

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

    # ---- starting pitchers -------------------------------------------
    #
    # The pitcher props have never been on this page. They are the
    # strongest part of the model -- pooled Brier skill around +0.09 on
    # the 5.5 strikeout line against +0.01 to +0.03 for any hitter prop --
    # and the only place to read them was the terminal after a score run.
    #
    # Separate table rather than more rows on the hitter one: the unit is
    # a STARTER, about fifteen a night against 250 hitters, so an `n` in
    # this table means something very different from an `n` above it.
    prow = os.path.join(CACHE_DIR, "pitcher_scoring_log.csv")
    plog = None
    if os.path.exists(prow):
        try:
            plog = pd.read_csv(prow)
        except Exception:
            plog = None

    if plog is not None and not plog.empty and "brier_skill" in plog.columns:
        html('<div style="font-size:15px;font-weight:640;color:var(--ink);'
             'margin-top:30px">Starting pitchers</div>'
             '<div style="color:var(--ink3);font-size:12.5px;'
             'margin-bottom:14px">One row per starter graded, not per '
             'hitter — so these counts are much smaller than the ones '
             'above and move around more.</div>')

        # Rows written before the outs prop existed are all strikeouts.
        if "prop" not in plog.columns:
            plog["prop"] = "strikeouts"
        plog["prop"] = plog["prop"].fillna("strikeouts")

        # The per-batter rate the strikeout prop compounds was replaced on
        # this date, and the old path scored -0.14 live against the new
        # one's +0.09 backtest. Two different models must not be pooled,
        # so the older rows are dropped from this table rather than
        # averaged into it -- the terminal shows both splits.
        K_PATH_CHANGED = "2026-09-05"
        dates = pd.to_datetime(plog["game_date"], errors="coerce")
        older = int((dates < pd.Timestamp(K_PATH_CHANGED)).sum())
        plog = plog[dates >= pd.Timestamp(K_PATH_CHANGED)]

    if plog is not None and not plog.empty:
        def _ppool(g):
            w = g["n"].to_numpy(dtype=float)
            did = float((g["base_rate"] * g["n"]).sum() / w.sum())
            brier = float((g["brier"] * g["n"]).sum() / w.sum())
            ref = did * (1.0 - did)
            se = (did * (1.0 - did) / w.sum()) ** 0.5 if w.sum() else float("nan")
            said = float((g["mean_pred"] * g["n"]).sum() / w.sum())
            return pd.Series({
                "n": int(w.sum()), "slates": g["game_date"].nunique(),
                "said": said, "did": did,
                "skill": 1.0 - brier / ref if ref > 0 else float("nan"),
                "sigma": (said - did) / se if se else float("nan")})

        by_line = (plog.groupby(["prop", "line"], sort=False)
                   [["n", "base_rate", "brier", "mean_pred", "game_date"]]
                   .apply(_ppool).reset_index()
                   .sort_values(["prop", "line"]))

        starters = int(plog.groupby("game_date")["n"].max().sum())
        n_pslates = plog["game_date"].nunique()
        best = by_line["skill"].max()
        html(f'<div class="sp-kpi">'
             f'<div><span class="v">{n_pslates}</span>'
             f'<span class="k">Slates scored</span></div>'
             f'<div><span class="v">{starters:,}</span>'
             f'<span class="k">Starters graded</span></div>'
             f'<div><span class="v">{best * 100:+.1f}%</span>'
             f'<span class="k">Best line</span></div></div>')

        LABEL = {"strikeouts": "Strikeouts over", "outs": "Outs over"}
        rows = ""
        for r in by_line.itertuples():
            extra = ""
            if r.prop == "outs":
                extra = f' <span style="color:var(--ink3)">({r.line / 3:.1f} inn)</span>'
            rows += (
                f'<tr><td style="font-weight:560">'
                f'{LABEL.get(r.prop, r.prop)} {r.line:g}{extra}</td>'
                f'<td style="color:var(--ink2)">{int(r.n):,}</td>'
                f'<td>{pct(r.said)}</td><td>{pct(r.did)}</td>'
                f'<td style="color:{"var(--ink2)" if abs(r.sigma) < 2 else "var(--warn)"}">'
                f'{r.sigma:+.1f}σ</td>'
                f'<td style="color:{skill_color(r.skill)};font-weight:560">'
                f'{r.skill * 100:+.2f}%</td></tr>')
        html('<table class="plain"><thead><tr><th>Prop</th>'
             '<th>Graded</th><th>Model said</th><th>Actually happened</th>'
             '<th>Miss</th><th>Edge</th></tr></thead><tbody>'
             + rows + '</tbody></table>')

        # The level check. Both props are built on the same projection of
        # how long a starter lasts, so if that runs long BOTH run long --
        # and a probability table cannot show it, because a model can be
        # biased on the count and still land near 50% on a line.
        pairs = [("Batters faced", "pred_bf", "actual_bf"),
                 ("Strikeouts", "pred_k", "actual_k"),
                 ("Outs recorded", "pred_outs", "actual_outs")]
        cells = ""
        for label, pc, ac in pairs:
            if pc not in plog.columns or ac not in plog.columns:
                continue
            sub = plog.dropna(subset=[pc, ac])
            # Nine innings is 27 outs. Anything past 30 is a parsing
            # failure, not a pitcher -- rows written before the innings
            # double-conversion was fixed carry 3.3e14 here, and one of
            # them would make this whole row meaningless.
            if ac == "actual_outs":
                sub = sub[sub[ac] <= 30]
            if sub.empty:
                continue
            w = sub["n"].to_numpy(dtype=float)
            pred = float((sub[pc] * sub["n"]).sum() / w.sum())
            act = float((sub[ac] * sub["n"]).sum() / w.sum())
            gap = pred - act

            # How sure is that gap? Without this the row reads as a
            # finding whatever it says, and the strikeout gap in
            # September 2026 was -0.34 at t = -2.36 -- real enough to
            # watch, nowhere near enough to correct for. A constant offset
            # fitted to thirteen September dates would also be fitting
            # expanded rosters and innings management, and would be wrong
            # in April.
            #
            # The standard error is taken ACROSS SLATES, not across
            # starts. Starts on one night share a weather system, an
            # umpire crew and a league-wide pattern of bullpen use, so
            # treating them as independent understates the error by
            # roughly the square root of the starts per night.
            per = (sub[pc] - sub[ac]).to_numpy(dtype=float)
            se = float("nan")
            if len(per) >= 3:
                var = float(((w * (per - gap) ** 2).sum() / w.sum())
                            * len(per) / max(len(per) - 1, 1))
                se = (var / len(per)) ** 0.5
            solid = pd.notna(se) and se > 0 and abs(gap) > 2 * se
            colour = "var(--warn)" if solid and abs(gap) >= 0.75 else (
                "var(--ink2)" if not solid else "var(--ink)")
            tail = (f' <span style="color:var(--ink3)">±{se:.2f}</span>'
                    if pd.notna(se) else "")
            verdict = ("" if pd.isna(se) else
                       ' <span style="color:var(--ink3);font-size:11px">'
                       + ("real" if solid else "not yet separable from noise")
                       + '</span>')
            cells += (
                f'<div style="display:flex;justify-content:space-between;'
                f'gap:14px;padding:3px 0;font-size:12.5px">'
                f'<span style="color:var(--ink3)">{label}</span>'
                f'<span style="font-variant-numeric:tabular-nums">'
                f'projected <b style="color:var(--ink)">{pred:.2f}</b> · '
                f'actual <b style="color:var(--ink)">{act:.2f}</b> · '
                f'<b style="color:{colour}">{gap:+.2f}</b>{tail}{verdict}'
                f'</span></div>')
        if cells:
            html(f'<div style="margin-top:16px;border:1px solid var(--line);'
                 f'border-radius:10px;padding:11px 14px;max-width:520px">'
                 f'<div style="font-size:11px;color:var(--ink3);'
                 f'letter-spacing:.06em;text-transform:uppercase;'
                 f'margin-bottom:5px">Level check — per start</div>'
                 f'{cells}</div>')

        note = ""
        if older:
            note = (f' The {older} row(s) from before {K_PATH_CHANGED} are '
                    f'left out: the strikeout prop compounded a different, '
                    f'untested rate until then, and pooling two models '
                    f'describes neither.')
        html(f'<div style="margin-top:12px;font-size:12px;color:var(--ink3);'
             f'line-height:1.55">Fifteen starters a night is a tenth of the '
             f'hitter sample, so a single slate here is almost pure noise '
             f'and even {n_pslates} is early.{note}<br>'
             f'<b style="color:var(--ink2)">Level check</b> is the half a '
             f'probability table cannot show: a model can sit near 50% on '
             f'every line and still be projecting starters a full inning '
             f'too deep. Both props are built on the same estimate of how '
             f'long a starter lasts, so when that drifts they drift '
             f'together.<br>'
             f'The <b style="color:var(--ink2)">±</b> is measured across '
             f'slates rather than across starts, because fifteen starters '
             f'on one night share a league-wide pattern of bullpen use and '
             f'are not fifteen independent draws. A gap marked '
             f'<i>not yet separable from noise</i> is not a number to '
             f'correct for — it is a number to keep watching, and this row '
             f'is the thing that will eventually say so.</div>')

    # ---- what a model probability is actually worth -------------------
    #
    # Nolan, on a starter the model liked at 73% over a 3.5 line while the
    # market sat at 52%: "why is the market saying so bad for him?"
    #
    # Three explanations were tested and died. The shrinkage is not too
    # strong -- K_PRIOR_BF = 250 is the measured optimum, and soft arms
    # BEAT their own rate (4.29 actual against 3.83 unshrunk), so
    # trusting his own low number would be worse. The distribution is not
    # too narrow -- standardised residual variance 0.957 +/- 0.080. And
    # "the model is projecting him above his own recent form" predicts
    # nothing at all (r = -0.045, p = 0.66 over 101 starts).
    #
    # What IS real is plain over-confidence, and it is worst exactly where
    # the biggest edges get displayed. Over 315 starts and six lines:
    #
    #     model said 73.8% on a 2.5 or 3.5 line  ->  happened 60.3%
    #
    # So this table is computed from the running log rather than
    # hard-coded, because it is the number that decides whether a printed
    # edge is real, and it should move as the record grows.
    _cal = None
    _cpath = os.path.join(CACHE_DIR, "pitcher_row_log.csv")
    if os.path.exists(_cpath):
        try:
            _cal = pd.read_csv(_cpath)
        except Exception:
            _cal = None
    if (_cal is not None and "k_dist" in _cal.columns
            and "strikeouts" in _cal.columns):
        _c = _cal.dropna(subset=["k_dist", "strikeouts"])
        _pts = []
        for _, _r in _c.iterrows():
            _p = parse_pmf(_r["k_dist"])
            if _p is None:
                continue
            for _ln in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
                _pts.append((_ln, prob_over(_p, _ln),
                             int(_r["strikeouts"] > _ln)))
        _cd = pd.DataFrame(_pts, columns=["line", "said", "hit"])
        if len(_cd) >= 200:
            html(f'<div style="font-size:15px;font-weight:640;'
                 f'color:var(--ink);margin-top:30px">'
                 f'What a strikeout probability is worth</div>'
                 f'<div style="color:var(--ink3);font-size:12.5px;'
                 f'margin-bottom:12px">When the model says X%, how often '
                 f'has it happened? {int(_cd["said"].notna().sum()):,} '
                 f'graded probabilities from '
                 f'{_c["game_date"].nunique()} slates.</div>')
            _cr = ""
            for _lo, _hi in ((0, .2), (.2, .35), (.35, .5), (.5, .65),
                             (.65, .8), (.8, 1.01)):
                _g = _cd[(_cd.said >= _lo) & (_cd.said < _hi)]
                if len(_g) < 25:
                    continue
                _said, _act = _g.said.mean(), _g.hit.mean()
                _se = float(((_g.said * (1 - _g.said)).sum()) ** 0.5) / len(_g)
                _off = abs(_act - _said) > 2 * _se
                _cr += (f'<tr><td style="font-weight:560">'
                        f'{_lo:.0%}–{_hi if _hi <= 1 else 1:.0%}</td>'
                        f'<td style="text-align:right">{len(_g)}</td>'
                        f'<td style="text-align:right">{_said:.1%}</td>'
                        f'<td style="text-align:right">{_act:.1%}</td>'
                        f'<td style="text-align:right;color:'
                        f'{"var(--warn)" if _off else "var(--ink3)"}">'
                        f'{_act - _said:+.1%}</td>'
                        f'<td style="text-align:right;color:var(--ink3)">'
                        f'±{_se:.1%}</td></tr>')
            html(f'<table class="plain"><thead><tr><th>Model said</th>'
                 f'<th style="text-align:right">n</th>'
                 f'<th style="text-align:right">Mean said</th>'
                 f'<th style="text-align:right">Happened</th>'
                 f'<th style="text-align:right">Gap</th>'
                 f'<th style="text-align:right">±</th></tr></thead>'
                 f'<tbody>{_cr}</tbody></table>')
            html('<div style="margin-top:10px;font-size:12px;'
                 'color:var(--ink3);line-height:1.55">'
                 'Every band reading negative means the model is '
                 '<b style="color:var(--ink2)">over-confident</b>, not '
                 'that it is wrong about who the good pitchers are — it '
                 'ranks them well (r = 0.82 against the market). It is '
                 'the SIZE of each probability that runs hot, and the '
                 'worst band tends to be 65–80%, which is exactly where '
                 'the Pitchers tab prints its biggest edges. An edge '
                 'computed from an overstated probability is overstated '
                 'by the same amount.<br>'
                 'This is why the number is measured here rather than '
                 'corrected in the model: the running record moves, and '
                 'a constant baked in today would be fitted to one '
                 'September.</div>')

    # ---- the same question for the hitters ---------------------------
    #
    # A prop can read said 11.5% / happened 11.5% across 3,700 rows and
    # still be badly hot in the band a LOTTO leg is actually picked from.
    # That is precisely how the strikeout props read fine in aggregate
    # while running 13.5 points hot at 65-80%, and until 2026-09-15 the
    # hitter side had no file that could show it: scoring_log.csv keeps
    # per-slate aggregates, and by the time a slate is summarised every
    # individual probability has been averaged away.
    #
    # hitter_row_log.csv keeps the rows. This reads them.
    _hl = None
    _hpath = os.path.join(CACHE_DIR, "hitter_row_log.csv")
    if os.path.exists(_hpath):
        try:
            _hl = pd.read_csv(_hpath)
        except Exception:
            _hl = None
    if _hl is not None and len(_hl) >= 300:
        # (prediction column, truth, label). A count line is graded the
        # same way score_slate grades it, so the two cannot drift.
        _HB = [("prob_hr", ("got_hr", None), "HR"),
               ("prob_hit", ("got_hit", None), "1+ hit"),
               ("prob_walk", ("got_walk", None), "1+ walk"),
               ("prob_hits_over_1.5", ("hits", 1.5), "Hits 1.5"),
               ("prob_tb_over_1.5", ("total_bases", 1.5), "TB 1.5"),
               ("prob_hrr_over_1.5", ("hrr", 1.5), "H+R+RBI 1.5")]
        _hr_rows, _n_flag, _n_band = "", 0, 0
        for _col, (_tc, _line), _lab in _HB:
            if _col not in _hl.columns or _tc not in _hl.columns:
                continue
            _d = _hl.dropna(subset=[_col, _tc]).copy()
            _d["_y"] = ((_d[_tc] > _line).astype(float) if _line is not None
                        else pd.to_numeric(_d[_tc], errors="coerce"))
            _d = _d.dropna(subset=["_y"])
            _first = True
            for _lo, _hi in ((0, .10), (.10, .20), (.20, .30), (.30, 1.01)):
                _g = _d[(_d[_col] >= _lo) & (_d[_col] < _hi)]
                if len(_g) < 25:
                    continue
                _said, _act = float(_g[_col].mean()), float(_g["_y"].mean())
                # Poisson-binomial: each row is its own coin, so the
                # variance is the sum of p(1-p) rather than n p_bar(1-p_bar).
                _se = (float(((_g[_col] * (1 - _g[_col])).sum())) ** 0.5
                       / len(_g))
                _off = abs(_act - _said) > 2 * _se
                _n_band += 1
                _n_flag += int(_off)
                _hr_rows += (
                    f'<tr><td style="font-weight:560">'
                    f'{_lab if _first else ""}</td>'
                    f'<td style="color:var(--ink3)">'
                    f'{_lo:.0%}–{min(_hi, 1):.0%}</td>'
                    f'<td style="text-align:right">{len(_g):,}</td>'
                    f'<td style="text-align:right">{_said:.1%}</td>'
                    f'<td style="text-align:right">{_act:.1%}</td>'
                    f'<td style="text-align:right;color:'
                    f'{"var(--warn)" if _off else "var(--ink3)"}">'
                    f'{_act - _said:+.1%}</td>'
                    f'<td style="text-align:right;color:var(--ink3)">'
                    f'±{_se:.1%}</td></tr>')
                _first = False
        if _hr_rows:
            html(f'<div style="font-size:15px;font-weight:640;'
                 f'color:var(--ink);margin-top:30px">'
                 f'What a hitter probability is worth</div>'
                 f'<div style="color:var(--ink3);font-size:12.5px;'
                 f'margin-bottom:12px">{len(_hl):,} graded hitter-games '
                 f'from {_hl["game_date"].nunique()} slates. The band that '
                 f'matters for a LOTTO leg is the bottom one on each prop, '
                 f'because that is where those legs are picked from.</div>')
            html(f'<table class="plain"><thead><tr><th>Prop</th>'
                 f'<th>Model said</th>'
                 f'<th style="text-align:right">n</th>'
                 f'<th style="text-align:right">Mean said</th>'
                 f'<th style="text-align:right">Happened</th>'
                 f'<th style="text-align:right">Gap</th>'
                 f'<th style="text-align:right">±</th></tr></thead>'
                 f'<tbody>{_hr_rows}</tbody></table>')
            # The multiple-comparisons line is not decoration. With this
            # many bands on screen, roughly one in twenty clears two sigma
            # on noise alone, and an amber cell that is simply the expected
            # one is the easiest way to talk yourself into a fix that is
            # not needed.
            _exp = _n_band * 0.05
            html(f'<div style="margin-top:10px;font-size:12px;'
                 f'color:var(--ink3);line-height:1.55">'
                 f'{_n_flag} of {_n_band} bands clear two standard errors; '
                 f'about {_exp:.1f} would on chance alone, so treat an '
                 f'amber cell as something to watch rather than something '
                 f'to fix. What would matter is several bands of the same '
                 f'prop leaning the same way, or a gap the size of the '
                 f'strikeout one above.<br>'
                 f'Compare against the pitcher table: the worst band here '
                 f'has been a couple of points, where the strikeout props '
                 f'ran <b style="color:var(--ink2)">13.5 points hot</b> at '
                 f'65–80%. The hitter side is not carrying that '
                 f'problem.</div>')

    # ---- the velocity marker, grading itself -------------------------
    #
    # Same contract the hitter hot/cold marker has: shown, logged, fed to
    # nothing, and left to earn its way in. The difference is that this
    # one arrived with a measured reason to exist -- 4,599 cached starts,
    # r = 0.052, p = 0.001 with the seasonal arc removed -- rather than a
    # hope, and the hitter version's own verdict after 13 slates was
    # nothing at all (hot minus cold -0.71 points, z = -0.31).
    #
    # So this block exists to answer the same question on live slates,
    # where it counts: does a starter whose fastball is down miss fewer
    # bats than his own projection says?
    # pitcher_row_log.csv is the only file where the marker sits next to
    # the outcome: score_slate's _log_pitcher_rows copies both out of
    # pitchers_{date}.csv after the games.
    _vlog = None
    _vpath = os.path.join(CACHE_DIR, "pitcher_row_log.csv")
    if os.path.exists(_vpath):
        try:
            _vlog = pd.read_csv(_vpath)
        except Exception:
            _vlog = None
    if (_vlog is not None and "velo_state" in _vlog.columns
            and "expected_k" in _vlog.columns
            and "strikeouts" in _vlog.columns):
        _v = _vlog.dropna(subset=["velo_state", "expected_k", "strikeouts"])
        _v = _v[_v["velo_state"].isin(["Hot", "Normal", "Cold"])]
        if len(_v) >= 12:
            html('<div style="font-size:15px;font-weight:640;color:var(--ink);'
                 'margin-top:30px">Velocity marker — is it real?</div>'
                 '<div style="color:var(--ink3);font-size:12.5px;'
                 'margin-bottom:12px">A starter whose fastball is off his '
                 'own norm: does he miss fewer bats than his projection '
                 'says? Shown only — the model does not use this.</div>')
            _vr = ""
            for _st in ("Hot", "Normal", "Cold"):
                _g = _v[_v["velo_state"] == _st]
                if _g.empty:
                    continue
                _gap = float((_g["strikeouts"] - _g["expected_k"]).mean())
                _se = float(_g["strikeouts"].sub(_g["expected_k"]).std()
                            / max(len(_g) ** 0.5, 1))
                _vr += (f'<tr><td style="font-weight:560">{_st}</td>'
                        f'<td style="text-align:right">{len(_g)}</td>'
                        f'<td style="text-align:right">'
                        f'{_g["expected_k"].mean():.2f}</td>'
                        f'<td style="text-align:right">'
                        f'{_g["strikeouts"].mean():.2f}</td>'
                        f'<td style="text-align:right">{_gap:+.2f}</td>'
                        f'<td style="text-align:right;color:var(--ink3)">'
                        f'±{_se:.2f}</td></tr>')
            html(f'<table class="plain"><thead><tr><th>Fastball</th>'
                 f'<th style="text-align:right">Starts</th>'
                 f'<th style="text-align:right">Projected K</th>'
                 f'<th style="text-align:right">Actual K</th>'
                 f'<th style="text-align:right">Gap</th>'
                 f'<th style="text-align:right">±</th></tr></thead>'
                 f'<tbody>{_vr}</tbody></table>')
            _hot = _v[_v.velo_state == "Hot"]
            _cold = _v[_v.velo_state == "Cold"]
            if len(_hot) >= 5 and len(_cold) >= 5:
                _h = float((_hot.strikeouts - _hot.expected_k).mean())
                _c = float((_cold.strikeouts - _cold.expected_k).mean())
                _sd = _v["strikeouts"].sub(_v["expected_k"]).std()
                _sep = float(_sd * (1 / len(_hot) + 1 / len(_cold)) ** 0.5)
                _z = (_h - _c) / _sep if _sep else 0
                html(f'<div style="margin-top:10px;font-size:12.5px;'
                     f'color:var(--ink3)">Hot minus cold: '
                     f'<b style="color:var(--ink2)">{_h - _c:+.2f}</b> '
                     f'strikeouts ±{_sep:.2f}, z = {_z:+.1f} — '
                     f'<b style="color:'
                     f'{"var(--warn)" if abs(_z) > 2 else "var(--ink3)"}">'
                     f'{"separable from noise" if abs(_z) > 2 else "not yet separable from noise"}'
                     f'</b>. History says to expect about +0.5; on '
                     f'{len(_v)} starts the error bar is still wider than '
                     f'that, so this row needs a season before it means '
                     f'anything.</div>')

    # ---- whole slips ------------------------------------------------
    #
    # Nolan asked which slips hit. This is that, and it is deliberately
    # EXPECTED against ACTUAL rather than a win-loss record.
    #
    # Twenty slips a night and about 2.3 expected to land, 1.7 of them
    # SAFE. A LOTTO slip is expected roughly once every fifty-six nights,
    # so its record reads 0-for-everything for months — and reading that
    # as a failure is exactly the mistake the layout has to prevent. To
    # detect a 20% miscalibration takes about 38 nights for SAFE, 286 for
    # MEDIUM, 526 for MIXED and 5,500 for LOTTO.
    #
    # Expected accumulates information every night even when nothing
    # lands, which a record does not.
    _slog_path = os.path.join(CACHE_DIR, "slip_log.csv")
    _slog = None
    if os.path.exists(_slog_path):
        try:
            _slog = pd.read_csv(_slog_path).dropna(subset=["hit"])
        except Exception:
            _slog = None

    if _slog is not None and not _slog.empty:
        _nights = _slog["game_date"].nunique()
        html(f'<div style="font-size:15px;font-weight:640;color:var(--ink);'
             f'margin-top:30px">Whole slips</div>'
             f'<div style="color:var(--ink3);font-size:12.5px;'
             f'margin-bottom:12px">Did every leg land? '
             f'{len(_slog)} slips over {_nights} night'
             f'{"s" if _nights != 1 else ""}, graded against the version '
             f'committed before first pitch.</div>')

        _rows = ""
        for _t in ("MIXED", "SAFE", "MEDIUM", "LOTTO"):
            _sub = _slog[_slog["tier"] == _t]
            if _sub.empty:
                continue
            _exp = float(_sub["slip_p"].sum())
            _act = float(_sub["hit"].sum())
            # Poisson-binomial: the variance of a sum of independent
            # indicators is the sum of p(1-p), NOT n*p*(1-p). Using the
            # latter would overstate the error bar several times over on
            # a tier whose slips have wildly different probabilities.
            _se = float(((_sub["slip_p"] * (1 - _sub["slip_p"])).sum()) ** 0.5)
            if _exp < 1:
                _verdict, _vc = "too few expected to read yet", "var(--ink3)"
            elif _se > 0 and abs(_act - _exp) > 2 * _se:
                _verdict = "running hot" if _act > _exp else "running cold"
                _vc = "var(--warn)"
            else:
                _verdict, _vc = "not yet separable from noise", "var(--ink3)"
            _rows += (
                f'<tr><td style="font-weight:560">{_t}</td>'
                f'<td style="text-align:right">{len(_sub)}</td>'
                f'<td style="text-align:right">{_exp:.1f}</td>'
                f'<td style="text-align:right">{_act:.0f}</td>'
                f'<td style="text-align:right">{_act - _exp:+.1f}</td>'
                f'<td style="text-align:right;color:var(--ink3)">'
                f'±{_se:.1f}</td>'
                f'<td style="color:{_vc};font-size:12px">{_verdict}</td></tr>')
        html(f'<table class="plain"><thead><tr><th>Tier</th>'
             f'<th style="text-align:right">Slips</th>'
             f'<th style="text-align:right">Expected</th>'
             f'<th style="text-align:right">Hit</th>'
             f'<th style="text-align:right">Gap</th>'
             f'<th style="text-align:right">±</th>'
             f'<th></th></tr></thead><tbody>{_rows}</tbody></table>')

        # The one thing this data can measure that nothing else can.
        #
        # A slip's probability is the product of its legs, which is right
        # only when they are independent. Two legs from one lineup share a
        # pitcher, a park and a night, so they land together more often
        # than the product implies. The tab has said so in words since it
        # was built and has never been able to say by how much. Split the
        # pool and the gap between the two halves IS the correlation.
        if "correlated" in _slog.columns and _slog["correlated"].nunique() > 1:
            _cr = ""
            for _flag, _lab in ((False, "Legs in different games"),
                                (True, "Two or more legs in one game")):
                _sub = _slog[_slog["correlated"].astype(bool) == _flag]
                if _sub.empty:
                    continue
                _exp = float(_sub["slip_p"].sum())
                _act = float(_sub["hit"].sum())
                _se = float(((_sub["slip_p"] * (1 - _sub["slip_p"])).sum()) ** 0.5)
                _ratio = (f"{_act / _exp:.2f}×" if _exp > 0 else "—")
                _cr += (
                    f'<tr><td style="font-weight:560">{_lab}</td>'
                    f'<td style="text-align:right">{len(_sub)}</td>'
                    f'<td style="text-align:right">{_exp:.1f}</td>'
                    f'<td style="text-align:right">{_act:.0f}</td>'
                    f'<td style="text-align:right">{_ratio}</td>'
                    f'<td style="text-align:right;color:var(--ink3)">'
                    f'±{_se:.1f}</td></tr>')
            html(f'<div style="font-size:13px;font-weight:620;'
                 f'color:var(--ink);margin-top:20px">Is the correlation '
                 f'warning worth a number yet?</div>'
                 f'<table class="plain" style="margin-top:6px">'
                 f'<thead><tr><th>Slip</th>'
                 f'<th style="text-align:right">Slips</th>'
                 f'<th style="text-align:right">Expected</th>'
                 f'<th style="text-align:right">Hit</th>'
                 f'<th style="text-align:right">Actual ÷ expected</th>'
                 f'<th style="text-align:right">±</th></tr></thead>'
                 f'<tbody>{_cr}</tbody></table>')

        html(f'<div style="margin-top:12px;font-size:12px;color:var(--ink3);'
             f'line-height:1.55">'
             f'<b style="color:var(--ink2)">Expected</b> is the sum of the '
             f'committed slip probabilities — what should have landed if '
             f'the numbers are honest. It is reported instead of a win-loss '
             f'record because a record cannot be read: about 2.3 of 20 '
             f'slips a night are expected to land, and a LOTTO slip roughly '
             f'once every fifty-six nights, so <b style="color:var(--ink2)">'
             f'0-for-200 on LOTTO is the correct-looking result, not a '
             f'failure</b>. Telling a calibrated tier from one that is 20% '
             f'off takes around 38 nights for SAFE, 286 for MEDIUM and '
             f'5,500 for LOTTO — so SAFE is the only row that will say '
             f'anything this season.<br>'
             f'The <b style="color:var(--ink2)">±</b> is the Poisson-'
             f'binomial standard deviation, the sum of p(1−p) across the '
             f'slips, not n·p̄(1−p̄) — these slips have wildly different '
             f'probabilities and the simpler formula would overstate the '
             f'error bar several times over.<br>'
             f'The second table is the only place the correlation warning '
             f'can become a measurement. Slips whose legs share a game '
             f'should beat their product, because the legs land together; '
             f'clean slips should sit on it. If that gap opens up, the '
             f'ratio is how much a same-game leg is really worth — and the '
             f'<i>all hit</i> figure on the Bet ready tab is too low by '
             f'about that factor.</div>')
