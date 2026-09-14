"""
Render dashboard.py without Streamlit.

WHY THIS EXISTS
---------------
dashboard.py is a flat script: every `with tab_x:` block shares one
namespace and runs top to bottom on every rerun. That is exactly the
shape that produces NameError from a definition that sits BELOW the tab
that uses it -- BASE_LABELS bit once already. The only way to catch that
class of bug is to actually execute the whole file, which means standing
up enough of the streamlit API for it to run.

This is a fake streamlit, not a mock: every call records itself and
returns something plausible, so the script runs end to end and the
counts at the end say what rendered.
"""
import sys, types, os, contextlib, traceback
import pandas as pd

CALLS = []


class _Ctx:
    def __enter__(self):  return self
    def __exit__(self, *a): return False


class _Col(_Ctx):
    def __getattr__(self, name):  return getattr(ST, name)


class _Styler:
    """pandas Styler is real; this only catches .apply/.format chaining."""
    def __init__(self, frame): self.data = frame
    def apply(self, *a, **k):  return self
    def format(self, *a, **k): return self
    def hide(self, *a, **k):   return self


class _Params(dict):
    def get(self, k, d=None): return dict.get(self, k, d)


class _State(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v


class _ColumnConfig:
    def __getattr__(self, name):
        return lambda *a, **k: {"_cfg": name, "args": a, "kwargs": k}


class _Sidebar(_Ctx):
    def __getattr__(self, name): return getattr(ST, name)


class _Slot:
    """What st.empty() hands back: anything written to it renders in place."""

    def _rec(self, name, *a, **k):
        CALLS.append((name, a, k))

    def markdown(self, *a, **k):   self._rec("markdown", *a, **k)
    def caption(self, *a, **k):    self._rec("caption", *a, **k)
    def dataframe(self, *a, **k):  self._rec("dataframe", *a, **k)
    def info(self, *a, **k):       self._rec("info", *a, **k)
    def warning(self, *a, **k):    self._rec("warning", *a, **k)
    def empty(self, *a, **k):      self._rec("empty_clear", *a, **k)


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.query_params = _Params()
        self.session_state = _State()
        self.column_config = _ColumnConfig()
        self.sidebar = _Sidebar()

    # --- recorded no-ops ------------------------------------------------
    def _rec(self, name, *a, **k):
        CALLS.append((name, a, k))

    def set_page_config(self, *a, **k): self._rec("set_page_config", *a, **k)
    def markdown(self, *a, **k):        self._rec("markdown", *a, **k)
    def code(self, *a, **k):            self._rec("code", *a, **k)
    def error(self, *a, **k):           self._rec("error", *a, **k)
    def info(self, *a, **k):            self._rec("info", *a, **k)
    def warning(self, *a, **k):         self._rec("warning", *a, **k)
    def caption(self, *a, **k):         self._rec("caption", *a, **k)
    def dataframe(self, *a, **k):       self._rec("dataframe", *a, **k)
    def rerun(self, *a, **k):           self._rec("rerun", *a, **k)

    def data_editor(self, frame, *a, **k):
        self._rec("data_editor", frame, **k)
        # Streamlit hands back the edited frame; nothing edited here.
        return frame.data if isinstance(frame, _Styler) else frame

    def button(self, *a, **k):
        self._rec("button", *a, **k); return False

    def empty(self, *a, **k):
        """
        A placeholder whose content is written later.

        The dashboard uses it where something has to APPEAR near the top of
        a tab but can only be COMPUTED after the table below it has been
        built -- the flat script runs top to bottom, so reserving the slot
        is the only way round that. Filling it is a real render, so the
        returned object records like the module does and its calls land in
        the same CALLS list the checks read.
        """
        self._rec("empty", *a, **k)
        return _Slot()

    def text_input(self, label, value="", **k):
        self._rec("text_input", label, **k); return value

    def selectbox(self, label, options, index=0, **k):
        self._rec("selectbox", label, options, **k)
        options = list(options)
        return options[index] if options else None

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        self._rec("columns", spec, **k)
        return [_Col() for _ in range(n)]

    def tabs(self, labels):
        self._rec("tabs", list(labels))
        return [_Ctx() for _ in labels]

    def expander(self, label, **k):
        self._rec("expander", label, **k); return _Ctx()

    def stop(self):
        raise SystemExit("st.stop()")

    def cache_data(self, *a, **k):
        # Used both bare and called: @st.cache_data and @st.cache_data(ttl=..)
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn


ST = FakeStreamlit()


def render(cache_dir, query=None, label=""):
    """Execute dashboard.py once. Returns a summary dict."""
    CALLS.clear()
    ST.query_params = _Params(query or {})
    ST.session_state = _State()
    sys.modules["streamlit"] = ST

    # pandas Styler -> our stub, so .apply(axis=None) chains don't try to
    # actually compute CSS over the frame.
    real_style = pd.DataFrame.style
    pd.DataFrame.style = property(lambda self: _Styler(self))

    here = os.getcwd()
    os.chdir(cache_dir)
    err = None
    try:
        src = open(SCRIPT).read()
        g = {"__name__": "__main__", "__file__": SCRIPT}
        exec(compile(src, SCRIPT, "exec"), g)
    except SystemExit as e:
        err = None if "st.stop()" in str(e) else e
    except BaseException as e:
        err = e
    finally:
        os.chdir(here)
        pd.DataFrame.style = real_style

    tabs = [c for c in CALLS if c[0] == "tabs"]
    html = "\n".join(str(c[1][0]) for c in CALLS if c[0] == "markdown" and c[1])
    out = {
        "label": label,
        "ok": err is None,
        "err": err,
        "tabs": len(tabs[0][1][0]) if tabs else 0,
        "tab_labels": tabs[0][1][0] if tabs else [],
        "dataframes": sum(1 for c in CALLS if c[0] in ("dataframe", "data_editor")),
        "errors": [c[1][0] for c in CALLS if c[0] == "error"],
        "html": html,
    }
    if err is not None:
        out["trace"] = "".join(traceback.format_exception(
            type(err), err, err.__traceback__))
    return out


SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
