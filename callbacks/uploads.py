"""
callbacks/uploads.py
─────────────────────────────────────────────────────────────────────────────
Upload callbacks — one per SPARK instrument.

Each callback is the sole writer to its store (single-writer rule).
The Clear button resets all stores to None.

The holdout cohort for ridge regression is handled separately
in callbacks/ridge.py and writes to ridge-holdout-store.
"""

import base64
import tempfile
from pathlib import Path

import pandas as pd
from dash import Input, Output, State, ctx, html, no_update

from helpers.store import df_to_store, patient_count
from modules import loader as L


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def _parse_uploaded(contents_list, filenames_list):
    """Decode Dash upload contents to temp file paths. Caller must delete."""
    if contents_list is None:
        return []
    if not isinstance(contents_list, list):
        contents_list = [contents_list]
    if filenames_list is None:
        filenames_list = [f"upload_{i}.csv" for i in range(len(contents_list))]
    elif not isinstance(filenames_list, list):
        filenames_list = [filenames_list]
    while len(filenames_list) < len(contents_list):
        filenames_list.append(f"upload_{len(filenames_list)}.csv")

    paths = []
    for contents, name in zip(contents_list, filenames_list):
        try:
            _, content_string = contents.split(",")
            data = base64.b64decode(content_string)
            suffix = Path(name).suffix or ".csv"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            paths.append(tmp.name)
        except Exception as e:
            print(f"[upload] parse error for {name}: {e}")
    return paths


def _ok(n_rows: int, n_cols: int) -> html.Span:
    return html.Span(f"✓ {n_rows:,} rows · {n_cols} cols",
                     className="status-ok")


def _err(msg: str) -> html.Span:
    return html.Span(f"⚠ {str(msg)[:80]}", className="status-err")


def _make_handler(loader_fn, expects_list: bool):
    """Factory: returns an upload callback for one instrument."""
    def handler(contents, n_clear, filenames):
        if ctx.triggered_id == "btn-clear":
            return None, html.Span("", className="status-muted")
        if contents is None:
            return no_update, no_update

        paths = _parse_uploaded(contents, filenames)
        if not paths:
            return no_update, _err("No files parsed")

        try:
            df = loader_fn(paths) if expects_list else loader_fn(paths[0])
            if df is None or df.empty:
                return no_update, _err("Loader returned empty data")
            return df_to_store(df), _ok(len(df), len(df.columns))
        except Exception as e:
            import traceback; traceback.print_exc()
            return no_update, _err(e)
        finally:
            for p in paths:
                try: Path(p).unlink()
                except: pass

    return handler


# ─────────────────────────────────────────────────────────────────────────────
# Registration

def register(app):
    # ── Instrument uploads ───────────────────────────────────────────────────
    loaders = {
        "dcdq": (L.load_dcdq,        False),
        "rbs":  (L.load_rbs,         False),
        "scq":  (L.load_scq,         False),
        "ados": (L.load_ados,        True),
        "cbcl": (L.load_cbcl,        True),
        "cov":  (L.load_covariates,  True),
    }

    for key, (loader_fn, expects_list) in loaders.items():
        handler = _make_handler(loader_fn, expects_list)
        app.callback(
            Output(f"{key}-store",  "data"),
            Output(f"status-{key}", "children"),
            Input(f"upload-{key}",  "contents"),
            Input("btn-clear",      "n_clicks"),
            State(f"upload-{key}",  "filename"),
            prevent_initial_call=True,
        )(handler)

    # ── Load summary (sidebar) ───────────────────────────────────────────────
    @app.callback(
        Output("load-summary", "children"),
        Input("dcdq-store",    "data"),
        Input("rbs-store",     "data"),
        Input("scq-store",     "data"),
        Input("ados-store",    "data"),
        Input("cbcl-store",    "data"),
        Input("cov-store",     "data"),
    )
    def _update_summary(dcdq, rbs, scq, ados, cbcl, cov):
        sources = [
            ("DCDQ",       dcdq),
            ("RBS-R",      rbs),
            ("SCQ",        scq),
            ("ADOS",       ados),
            ("CBCL",       cbcl),
            ("Covariates", cov),
        ]
        rows = []
        for label, store in sources:
            n = patient_count(store)
            if n > 0:
                rows.append(html.Div([
                    html.Span("✓ ", style={"color": "var(--success)"}),
                    html.Span(f"{label}: ", style={"color": "var(--text)"}),
                    html.Span(f"{n:,}", style={"color": "var(--text-muted)"}),
                ], style={"fontSize": "11px", "marginBottom": "2px"}))

        if not rows:
            return html.Span("No data loaded yet",
                             style={"color": "var(--text-muted)",
                                    "fontStyle": "italic"})
        return rows
