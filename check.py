"""Render every tab path and report."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")

# game_pk to drill into -- read from the real slate
import pandas as pd
slate = pd.read_csv(os.path.join(RUN, "cache", "slate_2026-09-12.csv"))
pk = int(slate["game_pk"].dropna().iloc[0])
pids = [str(int(x)) for x in slate["player_id"].dropna().unique()[:3]]

CASES = [
    ("plain",           {}),
    ("drill-down",      {"game": str(pk)}),
    ("drill + picks",   {"game": str(pk), "picks": ",".join(pids)}),
    ("bad game id",     {"game": "999999"}),
    ("picks only",      {"picks": ",".join(pids)}),
    ("bad pick id",     {"picks": "not-a-number,123"}),
]

print(f"{'case':16} {'ok':4} {'tabs':5} {'dfs':4} {'sel':4} {'back':5} picks")
print("-" * 62)
failed = []
for label, q in CASES:
    r = harness.render(RUN, q, label)
    html = r["html"]
    sel = "yes" if 'sp-g2 on' in html or 'class="sp-g on' in html else "no"
    # "all games" is now a button label, not markdown, so look for the
    # drill-down heading instead of the link text.
    back = "yes" if 'sp-display" style="font-size:19px' in html else "no"
    # The game rows no longer carry ?picks= by hand -- st.query_params
    # assignment preserves other keys. What matters now is that the panel
    # still renders the saved hitters, so check for that instead.
    # The count the sidebar panel reports, so a case that passes picks in
    # the URL and gets zero back is visible rather than just "present".
    _m = re.search(r">(\d+) of 8 · tick", html)
    picks = f"{_m.group(1)} saved" if _m else "no panel"
    print(f"{label:16} {'ok' if r['ok'] else 'FAIL':4} "
          f"{r['tabs']:<5} {r['dataframes']:<4} {sel:4} {back:5} {picks}")
    if not r["ok"]:
        failed.append(r)
    if r["errors"]:
        print(f"    st.error: {r['errors']}")

if failed:
    for r in failed:
        print(f"\n===== {r['label']} =====\n{r['trace']}")
    sys.exit(1)

# Tab labels, once.
r = harness.render(RUN, {}, "labels")
print("\ntabs:", r["tab_labels"])

# The summary line, verbatim.
m = re.search(r'class="sp-summary">(.*?)</', r["html"], re.S)
if m:
    print("\nsummary:", re.sub(r"<[^>]+>", "", m.group(1)).strip())

# Which finding cards fired.
cards = re.findall(r'class="sp-find-t">(.*?)<', r["html"])
print("cards:", cards or "(none)")


# ---- the live clock ---------------------------------------------------
#
# It renders through st.components.v1, which is a different code path from
# every other thing on the page: markdown strips <script>, so if someone
# "simplifies" this into an st.markdown call it will keep rendering and
# quietly stop ticking. Cheapest possible guard.
import harness as _h
_h.render(CACHE_DIR if "CACHE_DIR" in dir() else "cache", {}, "clock")
_clock = [a[0] for n, a, k in _h.CALLS if n == "components.html"]
print()
if _clock and "setInterval" in _clock[0] and "America/New_York" in _clock[0]:
    print("clock            ok    ticking, ballpark zone resolved")
else:
    print("clock            FAIL  not rendered as a live component")

# ---- does the DEPLOYED app get the files this page reads? -------------
#
# The dashboard degrades silently. A missing cache file does not raise --
# every block that reads one is wrapped in `if os.path.exists(...)`, by
# design, so a slate predicted before a feature existed still renders.
# The cost of that design is this failure mode: on 2026-09-15 the low-line
# warning and three whole Results sections were missing from the deployed
# site for days while every one of them rendered locally, because
# `cache/pitcher_row_log.csv` and `cache/hitter_row_log.csv` were never
# whitelisted in .gitignore and `git add -A` skips an ignored file without
# a word.
#
# .gitignore ignores `cache/*` and then negates the files the app needs,
# so every new reader needs a new negation in the same commit. This checks
# that they match.
import re as _re, os as _os, fnmatch as _fn

_src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "dashboard.py")).read()
_gi_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         ".gitignore")
_reads = set(_re.findall(r'CACHE_DIR,\s*f?"([^"]+)"', _src))
_reads |= set(_re.findall(r'cache_path\(f?"([^"]+)"', _src))
_reads |= set(_re.findall(r'glob\.glob\([^)]*"([^"]*\.csv)"', _src))
print()
if not _os.path.exists(_gi_path):
    print("gitignore         --    .gitignore not staged here; skipped")
else:
    _allow = set(_re.findall(r'^!cache/(\S+)', open(_gi_path).read(), _re.M))
    _missing = []
    for _p in sorted(_reads):
        _b = _re.sub(r"\{[^}]+\}", "*", _os.path.basename(_p))
        if not any(_fn.fnmatch(_b, _a) or _fn.fnmatch(_a, _b) for _a in _allow):
            _missing.append(_b)
    if _missing:
        print(f"gitignore        FAIL  the deployed app will never see: "
              f"{', '.join(_missing)}")
        print("                       add !cache/<name> to .gitignore, or the "
              "panel that reads it renders nothing on the website")
    else:
        print(f"gitignore        ok    all {len(_reads)} cache file(s) the "
              f"dashboard reads are tracked")
