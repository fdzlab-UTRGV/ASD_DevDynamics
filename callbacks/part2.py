"""
callbacks/part2.py
─────────────────────────────────────────────────────────────────────────────
Part 2 tab callbacks — mass-univariate √ΔR² analysis.
"""

import numpy as np
import pandas as pd
import uuid
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from helpers.store import get_merged_data, df_from_store
from helpers.theme import get_plotly_layout, get_axis_style
from modules.mass_univariate import run_mass_univariate
from modules.domains import (sort_by_domain, get_domain, domain_band_shapes,
                              domain_tick_colors, DOMAIN_COLORS, DOMAIN_ORDER,
                              make_legend_traces)
import modules.schema as S

# ── Predictor / outcome definitions for Part 2 ────────────────────────────────

# All input domain columns (DCDQ, RBS-R, SCQ + sensory instruments)
PART2_PREDICTORS = [
    # Motor / behavioral
    ("dcdq", "Gross Motor"),
    ("dcdq", "Fine Motor"),
    ("dcdq", "Coordination"),
    ("rbs",  "Sensory"),
    ("rbs",  "Obsessive"),
    ("rbs",  "Sameness"),
    ("rbs",  "Ritualistic"),
    ("rbs",  "Stereotyped"),
    ("rbs",  "SIB"),
    # Social communication (as input in Part 2)
    ("scq",  "Social"),
    ("scq",  "Communication"),
    ("scq",  "Sensory"),
]

# Sensory predictors from sensory-store (different source)
SENSORY_PREDICTORS = [
    ("sp",  "sp_low_reg",     "SP Low Reg"),
    ("sp",  "sp_seeking",     "SP Seeking"),
    ("sp",  "sp_sensitivity", "SP Sensitivity"),
    ("sp",  "sp_avoiding",    "SP Avoiding"),
    ("seq", "seq_hyper",      "SEQ Hyper"),
    ("seq", "seq_hypo",       "SEQ Hypo"),
    ("seq", "seq_enhanced",   "SEQ Enhanced"),
    ("seq", "seq_seeking",    "SEQ Seeking"),
    ("isq", "isq_noticing",    "ISQ Noticing"),
    ("isq", "isq_interpreting","ISQ Interpreting"),
    ("isq", "isq_acting",      "ISQ Acting"),
]

# All outcome domain columns
PART2_OUTCOMES = [
    ("cbcl", "Anxious/Dep."),
    ("cbcl", "Internalizing"),
    ("cbcl", "Externalizing"),
    ("cbcl", "Social Prob."),
    ("cbcl", "Attention"),
    ("cbcl", "Thought Prob."),
    ("cbcl", "Aggression"),
    ("cbcl", "Total"),
]

# ADOS/CSS outcomes (from css-store)
CSS_OUTCOMES = [
    ("css", "css_total",  "ADOS CSS-Total"),
    ("css", "css_sa",     "ADOS CSS-SA"),
    ("css", "css_rrb",    "ADOS CSS-RRB"),
]

SOURCE_STATES = [
    State("dcdq-store",    "data"),
    State("rbs-store",     "data"),
    State("scq-store",     "data"),
    State("ados-store",    "data"),
    State("cbcl-store",    "data"),
    State("cov-store",     "data"),
    State("sensory-store", "data"),
    State("css-store",     "data"),
]


def _get_merged(*vals):
    keys = ["dcdq", "rbs", "scq", "ados", "cbcl", "cov", "sensory", "css"]
    return get_merged_data(**dict(zip(keys, vals)))


def _sig_marker(q, thresh):
    if np.isnan(q):
        return ""
    if q < thresh * 0.001:
        return "***"
    if q < thresh * 0.01:
        return "**"
    if q < thresh:
        return "*"
    return ""


def register(app):

    # ── Populate predictor / outcome checklists ───────────────────────────────
    @app.callback(
        Output("p2-predictors",     "options"),
        Output("p2-predictors",     "value"),
        Output("p2-outcomes",       "options"),
        Output("p2-outcomes",       "value"),
        Output("p2-suppressor-var", "options"),
        Output("p2-suppressor-var", "value"),
        Input("main-tabs",          "active_tab"),
        Input("cbcl-store",         "data"),   # re-populate when outcomes arrive
        Input("css-store",          "data"),   # re-populate when ADOS severity computed
        Input("sensory-store",      "data"),   # re-populate when sensory predictors arrive
        *SOURCE_STATES,
    )
    def populate_p2(tab, _cbcl_trigger, _css_trigger, _sensory_trigger,
                    dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        if tab != "tab-p2":
            return (no_update, no_update, no_update, no_update,
                    no_update, no_update)

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        all_cols = set(mg.columns) if mg is not None else set()

        # Build predictor options
        pred_opts, pred_vals = [], []
        for scale, domain in PART2_PREDICTORS:
            col = f"{scale}_{domain}"
            if col in all_cols:
                pred_opts.append({"label": f"{scale.upper()} {domain}", "value": col})
                pred_vals.append(col)

        sdf = df_from_store(sensory) if sensory else None
        if sdf is not None:
            for _, col, label in SENSORY_PREDICTORS:
                if col in sdf.columns and sdf[col].notna().any():
                    pred_opts.append({"label": label, "value": f"_sens_{col}"})

        # Build outcome options
        out_opts, out_vals = [], []
        for scale, domain in PART2_OUTCOMES:
            col = f"{scale}_{domain}"
            if col in all_cols:
                out_opts.append({"label": f"{scale.upper()} {domain}", "value": col})
                out_vals.append(col)
        for _, col, label in CSS_OUTCOMES:
            if col in all_cols:
                out_opts.append({"label": label, "value": col})
                out_vals.append(col)

        # Suppressor options — CBCL columns + any available numeric column
        SUPPRESSOR_CANDIDATES = [
            ("cbcl_Anxious/Dep.",   "CBCL Anxious/Dep"),
            ("cbcl_Internalizing",  "CBCL Internalising"),
            ("cbcl_Externalizing",  "CBCL Externalising"),
            ("cbcl_Total",          "CBCL Total"),
            ("cbcl_Attention",      "CBCL Attention"),
            ("css_total",           "ADOS CSS-Total"),
            ("css_sa",              "ADOS CSS-SA"),
            ("css_rrb",             "ADOS CSS-RRB"),
            ("scq_Social",          "SCQ Social"),
        ]
        supp_opts = [{"label": lbl, "value": col}
                     for col, lbl in SUPPRESSOR_CANDIDATES
                     if col in all_cols]
        supp_val = supp_opts[0]["value"] if supp_opts else None

        return (pred_opts, pred_vals, out_opts, out_vals,
                supp_opts, supp_val)

    # ── Run mass-univariate analysis ──────────────────────────────────────────
    @app.callback(
        Output("p2-content",        "children"),
        Output("p2-results-store",  "data"),
        Output("btn-p2-export",     "disabled"),
        Output("btn-p2-pca",        "disabled"),
        Output("btn-p2-split",      "disabled"),
        Output("btn-p2-suppress",   "disabled"),
        Input("btn-p2-run",        "n_clicks"),
        State("p2-predictors",     "value"),
        State("p2-outcomes",       "value"),
        State("p2-covariates",     "value"),
        State("p2-fdr-thresh",     "value"),
        State("p2-display-mode",   "value"),
        State("p2-group-domain",   "value"),
        State("theme-store",       "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def run_p2(_, pred_vals, out_vals, covariates, fdr_thresh, display_mode,
               group_domain, theme,
               dcdq, rbs, scq, ados, cbcl, cov, sensory, css):

        if not pred_vals or not out_vals:
            return (dbc.Alert("Select at least one predictor and one outcome.",
                              color="warning"),
                    None, True, True, True, True)

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return (dbc.Alert("No data loaded.", color="warning"),
                    None, True, True, True, True)

        cov_present = [c for c in (covariates or []) if c in mg.columns]

        # ── Handle sensory predictors from sensory-store ──────────────────
        sdf = df_from_store(sensory) if sensory else None
        sens_preds = [v for v in pred_vals if v.startswith("_sens_")]
        std_preds  = [v for v in pred_vals if not v.startswith("_sens_")]

        # Sensory columns are already in mg via _get_merged — just strip prefix
        for v in sens_preds:
            col = v.replace("_sens_", "")
            if col in mg.columns:
                std_preds.append(col)

        all_preds = [p for p in std_preds if p in mg.columns]
        all_outs  = [o for o in out_vals if o in mg.columns]

        if not all_preds:
            return (dbc.Alert("None of the selected predictors are in the data.",
                              color="warning"),
                    None, True, True, True, True)
        if not all_outs:
            return (dbc.Alert(
                "None of the selected outcomes are in the data. "
                "Load CBCL data and/or compute ADOS Severity first.",
                color="warning"),
                None, True, True, True, True)

        result = run_mass_univariate(mg, all_preds, all_outs, cov_present)
        if "error" in result:
            return (dbc.Alert(result["error"], color="danger"),
                    None, True, True, True, True)

        t = theme or "dark"
        dark = t == "dark"
        thresh = float(fdr_thresh or 0.05)

        # ── Choose display matrix ─────────────────────────────────────────
        do_group = bool(group_domain and "group" in (group_domain or []))

        if display_mode == "beta":
            mat = result["beta"]
            cbar_title = "β"
            mid_val = 0
        elif display_mode == "r2_full":
            mat = result["r2_full"]
            cbar_title = "R²"
            mid_val = mat.values[~np.isnan(mat.values)].mean() if not mat.empty else 0
        else:
            mat = result["sqrt_dr2"]
            cbar_title = "√ΔR²"
            mid_val = 0

        pval_fdr = result["pval_fdr"]
        n_obs    = result["n_obs"]

        # ── Domain grouping: sort predictors and outcomes by domain ───────
        pred_order = sort_by_domain(mat.index.tolist()) if do_group else mat.index.tolist()
        out_order  = sort_by_domain(mat.columns.tolist()) if do_group else mat.columns.tolist()
        mat      = mat.loc[pred_order, out_order]
        pval_fdr = pval_fdr.loc[pred_order, out_order]
        n_obs    = n_obs.loc[pred_order, out_order]

        # ── Build heatmap with FDR annotations ────────────────────────────
        z_vals = mat.values.tolist()

        def _clean_label(c):
            return (c.replace("cbcl_", "CBCL ").replace("css_", "ADOS ")
                     .replace("dcdq_", "DCDQ ").replace("rbs_", "RBS-R ")
                     .replace("scq_", "SCQ ").replace("ados_", "ADOS ")
                     .replace("sp_", "SP ").replace("seq_", "SEQ ")
                     .replace("isq_", "ISQ ").replace("_", " "))

        x_labs = [_clean_label(c) for c in mat.columns]
        y_labs = [_clean_label(r) for r in mat.index]

        # Annotation text: value + significance stars
        ann_text = []
        for pi, pred in enumerate(mat.index):
            row_ann = []
            for oi, out in enumerate(mat.columns):
                v = mat.loc[pred, out]
                q = pval_fdr.loc[pred, out]
                n = n_obs.loc[pred, out]
                if np.isnan(v):
                    row_ann.append("")
                else:
                    stars = _sig_marker(q, thresh)
                    row_ann.append(f"{v:.3f}{stars}<br><span style='font-size:8px'>n={int(n)}</span>")
            ann_text.append(row_ann)

        mid = "#0d0f14" if dark else "#f1f5f9"
        fig = go.Figure(go.Heatmap(
            z=z_vals,
            x=x_labs,
            y=y_labs,
            colorscale=[[0, "#34d399"], [0.5, mid], [1, "#f87171"]],
            zmid=mid_val,
            text=ann_text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar={"title": cbar_title, "thickness": 12,
                      "tickfont": {"size": 9}},
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>"
                          f"{cbar_title} = %{{z:.4f}}<extra></extra>",
        ))

        # ── Domain colour bands on both axes ─────────────────────────────
        ax = get_axis_style(t)
        layout_extra = {}
        if do_group:
            row_shapes = domain_band_shapes(mat.index.tolist(), axis="y", dark=dark)
            col_shapes = domain_band_shapes(mat.columns.tolist(), axis="x", dark=dark)
            layout_extra["shapes"] = row_shapes + col_shapes

            # Colour-coded tick labels using SVG font colour hack via annotations
            row_cols = domain_tick_colors(mat.index.tolist())
            col_cols = domain_tick_colors(mat.columns.tolist())

            tick_anns = []
            for i, (lbl, col) in enumerate(zip(y_labs, row_cols)):
                tick_anns.append({
                    "x": -0.01, "y": i, "xref": "paper", "yref": "y",
                    "text": f"<span style='color:{col}'>{lbl}</span>",
                    "showarrow": False, "xanchor": "right",
                    "font": {"size": 9},
                })
            for j, (lbl, col) in enumerate(zip(x_labs, col_cols)):
                tick_anns.append({
                    "x": j, "y": -0.01, "xref": "x", "yref": "paper",
                    "text": f"<span style='color:{col}'>{lbl}</span>",
                    "showarrow": False, "yanchor": "top",
                    "textangle": -35, "font": {"size": 9},
                })
            layout_extra["annotations"] = tick_anns
            layout_extra["xaxis"] = {**ax, "showticklabels": False,
                                     "tickfont": {"size": 9}}
            layout_extra["yaxis"] = {**ax, "showticklabels": False,
                                     "tickfont": {"size": 9},
                                     "autorange": "reversed"}

            # Add domain legend traces
            for tr in make_legend_traces():
                fig.add_trace(go.Scatter(**tr))
        else:
            layout_extra["xaxis"] = {**ax, "tickangle": -35,
                                      "tickfont": {"size": 9}, "side": "bottom"}
            layout_extra["yaxis"] = {**ax, "tickfont": {"size": 9},
                                      "autorange": "reversed"}

        n_pred = len(mat.index)
        n_out  = len(mat.columns)
        fig.update_layout(**{
            **get_plotly_layout(t),
            "height": max(300, n_pred * 38 + 120),
            "margin": {"l": 180, "r": 80, "t": 40, "b": 120},
            **layout_extra,
        })

        # ── Summary statistics ────────────────────────────────────────────
        flat_sr2 = result["sqrt_dr2"].values.flatten()
        flat_sr2 = flat_sr2[~np.isnan(flat_sr2)]
        flat_fdr = result["pval_fdr"].values.flatten()
        flat_fdr = flat_fdr[~np.isnan(flat_fdr)]
        n_sig = int((flat_fdr < thresh).sum())

        cov_str = ", ".join(cov_present) if cov_present else "none"
        summary = html.Div([
            html.Span(f"✓ {result['n_tests']} tests  ·  ",
                      style={"color": "var(--success)", "fontWeight": "700"}),
            html.Span(f"{n_sig} significant at FDR q < {thresh}  ·  ",
                      style={"color": "var(--accent)"}),
            html.Span(f"Covariates: {cov_str}  ·  ",
                      style={"color": "var(--text-muted)"}),
            html.Span(f"Effect sizes: max |√ΔR²| = {np.max(np.abs(flat_sr2)):.3f}  "
                      f"median = {np.median(np.abs(flat_sr2)):.3f}",
                      style={"color": "var(--text-muted)"}),
        ], style={"fontSize": "11px", "marginBottom": "10px"})

        legend_note = html.Div(
            f"* q < {thresh}  ** q < {thresh/5:.3f}  *** q < {thresh/500:.4f}  "
            "│  FDR-corrected across all predictor × outcome pairs simultaneously.",
            style={"fontSize": "10px", "color": "var(--text-muted)",
                   "marginTop": "8px"},
        )

        content = html.Div([
            summary,
            dcc.Graph(figure=fig,
                      config={"displayModeBar": True,
                              "toImageButtonOptions": {
                                  "format": "png", "scale": 2,
                                  "filename": "sqrt_dr2_fingerprint"}}),
            legend_note,
        ])

        # Serialise result for store
        store_payload = {
            "run_id":     uuid.uuid4().hex,
            "sqrt_dr2":   result["sqrt_dr2"].to_dict(),
            "pval_fdr":   result["pval_fdr"].to_dict(),
            "pval_raw":   result["pval_raw"].to_dict(),
            "n_obs":      result["n_obs"].to_dict(),
            "beta":       result["beta"].to_dict(),
            "r2_full":    result["r2_full"].to_dict(),
            "predictors": result["predictors"],
            "outcomes":   result["outcomes"],
            "cov_cols":   result["cov_cols"],
            "n_tests":    result["n_tests"],
        }

        return content, store_payload, False, False, False, False

    # ── Export CSV ────────────────────────────────────────────────────────────
    @app.callback(
        Output("p2-download", "data"),
        Input("btn-p2-export", "n_clicks"),
        State("p2-results-store", "data"),
        prevent_initial_call=True,
    )
    def export_p2(_, payload):
        if not payload:
            return no_update
        mat = pd.DataFrame(payload["sqrt_dr2"])
        mat.index.name = "predictor"
        mat.columns.name = "outcome"
        # Long format for easier use
        long = mat.reset_index().melt(id_vars="predictor",
                                      var_name="outcome",
                                      value_name="sqrt_dr2")
        fdr = pd.DataFrame(payload["pval_fdr"])
        fdr_long = fdr.reset_index().melt(id_vars="index",
                                           var_name="outcome",
                                           value_name="pval_fdr")
        long["pval_fdr"] = fdr_long["pval_fdr"].values
        n_df = pd.DataFrame(payload["n_obs"])
        n_long = n_df.reset_index().melt(id_vars="index", var_name="outcome",
                                          value_name="n")
        long["n"] = n_long["n"].values
        long["run_id"] = payload.get("run_id", "")
        long["covariates"] = "|".join(payload.get("cov_cols", []))
        long["n_tests"] = payload.get("n_tests")
        return dcc.send_data_frame(long.to_csv, "sqrt_dr2_results.csv",
                                   index=False)

    # ── PCA of √ΔR² matrix ───────────────────────────────────────────────────
    @app.callback(
        Output("p2-pca-content",      "children"),
        Output("p2-pca-store",        "data"),
        Output("btn-p2-pca-export",   "disabled"),
        Input("btn-p2-pca",           "n_clicks"),
        State("p2-results-store","data"),
        State("theme-store",     "data"),
        prevent_initial_call=True,
    )
    def run_p2_pca(_, payload, theme):
        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), None, True

        mat = pd.DataFrame(payload["sqrt_dr2"]).astype(float)
        mat = mat.dropna(how="all").dropna(axis=1, how="all")

        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return dbc.Alert("Not enough data for PCA.", color="warning"), None, True

        # Fill remaining NaN with 0 (missing pair = no association)
        Z = mat.fillna(0).values

        # Centre and scale each column
        Z = Z - Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std == 0] = 1
        Z = Z / std

        # SVD
        try:
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            return dbc.Alert("SVD failed.", color="danger"), None, True

        var_exp = (s**2) / (s**2).sum()
        pc1_load = Vt[0]   # outcome loadings on PC1
        pc1_score = U[:, 0] * s[0]   # predictor scores on PC1

        t = theme or "dark"
        ax = get_axis_style(t)

        # PC1 outcome loadings bar chart
        fig_load = go.Figure(go.Bar(
            x=mat.columns.tolist(),
            y=pc1_load.tolist(),
            marker_color=["#f87171" if v > 0 else "#34d399" for v in pc1_load],
        ))
        fig_load.update_layout(**{
            **get_plotly_layout(t),
            "title": {"text": f"PC1 outcome loadings ({var_exp[0]*100:.1f}% variance)",
                      "font": {"size": 12}},
            "height": 260,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 80},
            "xaxis": {**ax, "tickangle": -35, "tickfont": {"size": 9}},
            "yaxis": {**ax, "title": "Loading"},
        })

        # PC1 predictor scores bar chart
        fig_score = go.Figure(go.Bar(
            x=mat.index.tolist(),
            y=pc1_score.tolist(),
            marker_color=["#f87171" if v > 0 else "#34d399" for v in pc1_score],
        ))
        fig_score.update_layout(**{
            **get_plotly_layout(t),
            "title": {"text": "PC1 predictor scores", "font": {"size": 12}},
            "height": 260,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 100},
            "xaxis": {**ax, "tickangle": -45, "tickfont": {"size": 9}},
            "yaxis": {**ax, "title": "Score"},
        })

        # Scree
        fig_scree = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(min(8, len(s)))],
            y=[float(v*100) for v in var_exp[:8]],
            marker_color="#38bdf8",
        ))
        fig_scree.update_layout(**{
            **get_plotly_layout(t),
            "title": {"text": "Variance explained per PC", "font": {"size": 12}},
            "height": 220,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "xaxis": {**ax},
            "yaxis": {**ax, "title": "% variance"},
        })


        # ── Build PCA store payload ──────────────────────────────────────────
        n_pcs = min(5, len(s))
        pca_store = {
            "var_explained":     var_exp[:n_pcs].tolist(),
            "outcomes":          mat.columns.tolist(),
            "predictors":        mat.index.tolist(),
            "outcome_loadings":  {f"PC{i+1}": Vt[i].tolist() for i in range(n_pcs)},
            "predictor_scores":  {f"PC{i+1}": (U[:, i] * s[i]).tolist() for i in range(n_pcs)},
        }

        return html.Div([
            html.Div(
                f"PCA of {mat.shape[0]} predictors × {mat.shape[1]} outcomes √ΔR² matrix. "
                f"PC1 explains {var_exp[0]*100:.1f}% of total √ΔR² variance.",
                style={"fontSize": "11px", "color": "var(--text-muted)",
                       "marginBottom": "12px"},
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scree,
                                  config={"displayModeBar": False}), width=4),
                dbc.Col(dcc.Graph(figure=fig_load,
                                  config={"displayModeBar": False}), width=8),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_score,
                                  config={"displayModeBar": False}), width=12),
            ]),
        ]), pca_store, False

    # ── Split-half ridge regression ───────────────────────────────────────────
    @app.callback(
        Output("p2-split-content",      "children"),
        Output("p2-split-store",        "data"),
        Output("btn-p2-split-export",   "disabled"),
        Input("btn-p2-split",           "n_clicks"),
        State("p2-results-store",   "data"),
        State("p2-split-seed",      "value"),
        State("p2-covariates",      "value"),
        State("theme-store",        "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def run_p2_split(_, payload, seed, covariates, theme,
                     dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        from modules.split_half import run_split_half
        import plotly.graph_objects as go
        from scipy import stats as scipy_stats

        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), None, True

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return dbc.Alert("No data loaded.", color="warning"), None, True

        # Join sensory columns if needed
        preds = payload.get("predictors", [])
        sdf   = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds if c in sdf.columns]
            if sens_cols:
                sens_cols = [c for c in sens_cols if c not in mg.columns]
                if sens_cols:
                    mg = mg.join(sdf[sens_cols], how="left")

        all_preds = [p for p in preds if p in mg.columns]
        all_outs  = [o for o in payload.get("outcomes", []) if o in mg.columns]
        cov_present = [c for c in (covariates or []) if c in mg.columns]

        result = run_split_half(
            mg, all_preds, all_outs, cov_present,
            seed=int(seed or 42),
        )
        if "error" in result:
            return dbc.Alert(result["error"], color="danger"), None, True

        t    = theme or "dark"
        dark = t == "dark"
        ax   = get_axis_style(t)

        conc_r = result["concordance_r"]
        conc_p = result["concordance_p"]
        n_disc = result["n_disc"]
        n_rep  = result["n_rep"]

        # ── Concordance scatter ───────────────────────────────────────────
        d_vec = result["disc_dr2"].values.flatten().astype(float)
        r_vec = result["rep_dr2"].values.flatten().astype(float)
        mask  = ~(np.isnan(d_vec) | np.isnan(r_vec))

        # Labels for hover
        preds_list = result["disc_dr2"].index.tolist()
        outs_list  = result["disc_dr2"].columns.tolist()
        hover_labs = [f"{p} → {o}"
                      for p in preds_list for o in outs_list]
        hover_mask = [hover_labs[i] for i, m in enumerate(mask) if m]

        conc_color = ("#34d399" if conc_r > 0.7
                      else "#fbbf24" if conc_r > 0.4
                      else "#f87171")

        fig_conc = go.Figure()
        fig_conc.add_trace(go.Scatter(
            x=d_vec[mask].tolist(),
            y=r_vec[mask].tolist(),
            mode="markers",
            marker={"size": 8, "color": "#38bdf8", "opacity": 0.75},
            text=hover_mask,
            hovertemplate="%{text}<br>Discovery: %{x:.3f}<br>Replication: %{y:.3f}<extra></extra>",
        ))
        # Identity line
        lo = float(min(d_vec[mask].min(), r_vec[mask].min())) - 0.02
        hi = float(max(d_vec[mask].max(), r_vec[mask].max())) + 0.02
        fig_conc.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line={"color": "#334155", "dash": "dash", "width": 1},
            showlegend=False, hoverinfo="skip",
        ))
        p_str = ("p < .001" if conc_p < 0.001
                 else f"p = {conc_p:.3f}" if not np.isnan(conc_p) else "—")
        fig_conc.update_layout(**{
            **get_plotly_layout(t),
            "height": 340,
            "title": {"text": f"√ΔR² concordance — r = {conc_r:.3f}  ({p_str})",
                      "font": {"size": 12, "color": conc_color}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
            "xaxis": {**ax, "title": "Discovery √ΔR²"},
            "yaxis": {**ax, "title": "Replication √ΔR²"},
        })

        # ── Ridge results table ───────────────────────────────────────────
        ridge_rows = []
        for rr in sorted(result["ridge"], key=lambda x: -x["r_test"]):
            r    = rr["r_test"]
            p    = rr["p_test"]
            pstr = "p < .001" if p < 0.001 else f"p = {p:.3f}"
            bar_w = max(0, min(100, int(abs(r) * 100)))
            bar_col = "#34d399" if r > 0 else "#f87171"
            ridge_rows.append(html.Tr([
                html.Td(rr["outcome"].replace("_", " "),
                        style={"fontSize": "11px", "paddingRight": "12px"}),
                html.Td([
                    html.Div(style={
                        "width": f"{bar_w}px", "height": "10px",
                        "backgroundColor": bar_col,
                        "borderRadius": "3px", "display": "inline-block",
                    }),
                    html.Span(f" {r:.3f}",
                              style={"fontSize": "11px", "color": bar_col,
                                     "fontWeight": "700"}),
                ], style={"paddingRight": "12px"}),
                html.Td(f"{rr['r2_test']:.3f}",
                        style={"fontSize": "10px", "color": "var(--text-muted)",
                               "paddingRight": "12px"}),
                html.Td(pstr,
                        style={"fontSize": "10px", "color": "var(--text-muted)",
                               "paddingRight": "12px"}),
                html.Td(f"n={rr['n_rep']:,}  α={rr['alpha']}",
                        style={"fontSize": "10px", "color": "var(--text-muted)"}),
            ]))

        ridge_header = html.Tr([
            html.Th(c, style={"fontSize": "10px", "color": "var(--text-muted)",
                              "fontWeight": "600", "paddingRight": "12px",
                              "paddingBottom": "6px"})
            for c in ["Outcome", "Out-of-sample r", "r²", "p-value",
                      "n (test) · α"]
        ])

        p_conc = ("p < .001" if conc_p < 0.001
                  else f"p = {conc_p:.3f}" if not np.isnan(conc_p) else "—")

        # ── Split store payload ──────────────────────────────────────────────
        split_store = {
            "concordance_r":  float(conc_r),
            "concordance_p":  float(conc_p) if not np.isnan(conc_p) else None,
            "n_disc":         int(n_disc),
            "n_rep":          int(n_rep),
            "n_pairs":        int(result["n_pairs"]),
            "disc_dr2":       result["disc_dr2"].to_dict(),
            "rep_dr2":        result["rep_dr2"].to_dict(),
            "disc_pval_fdr":  result["disc_pval_fdr"].to_dict(),
            "rep_pval_fdr":   result["rep_pval_fdr"].to_dict(),
            "disc_n_obs":     result["disc_n_obs"].to_dict(),
            "rep_n_obs":      result["rep_n_obs"].to_dict(),
            "ridge":          result["ridge"],
        }
        return html.Div([
            # Headline
            html.Div([
                html.Span("Concordance r = ",
                          style={"fontSize": "12px", "color": "var(--text-muted)"}),
                html.Span(f"{conc_r:.3f}",
                          style={"fontSize": "18px", "fontWeight": "700",
                                 "color": conc_color}),
                html.Span(f"  ({p_conc})  ·  {result['n_pairs']} predictor×outcome pairs  ·  "
                          f"Discovery n={n_disc:,}  ·  Replication n={n_rep:,}",
                          style={"fontSize": "11px", "color": "var(--text-muted)"}),
            ], style={"marginBottom": "16px"}),

            dbc.Row([
                # Concordance scatter
                dbc.Col(dcc.Graph(figure=fig_conc,
                                  config={"displayModeBar": False,
                                          "toImageButtonOptions": {
                                              "filename": "concordance_scatter",
                                              "format": "png", "scale": 2}}),
                        width=5),
                # Ridge table
                dbc.Col([
                    html.Div("Ridge regression — out-of-sample prediction",
                             style={"fontSize": "12px", "fontWeight": "700",
                                    "marginBottom": "8px"}),
                    html.Div(
                        "Training: discovery half  ·  Testing: replication half  ·  "
                        "α selected by 5-fold CV within discovery.",
                        style={"fontSize": "10px", "color": "var(--text-muted)",
                               "marginBottom": "8px"},
                    ),
                    html.Table(
                        [ridge_header] + ridge_rows,
                        style={"borderCollapse": "collapse"},
                    ) if ridge_rows else dbc.Alert(
                        "Ridge regression requires complete cases across all "
                        "predictors and outcomes simultaneously.",
                        color="warning", style={"fontSize": "11px"}),
                ], width=7),
            ]),

            html.Div(
                "Concordance r measures whether the pattern of associations in the "
                "discovery half replicates in the independent replication half. "
                "Out-of-sample r measures multivariate prediction accuracy. "
                f"Split stratified by age quartile × sex (seed={seed}).",
                style={"fontSize": "10px", "color": "var(--text-muted)",
                       "marginTop": "12px"},
            ),
        ]), split_store, False

    # ── Suppressor analysis with √ΔR² ─────────────────────────────────────────
    @app.callback(
        Output("p2-suppress-content",     "children"),
        Output("p2-suppress-store",       "data"),
        Output("btn-p2-suppress-export",  "disabled"),
        Input("btn-p2-suppress",          "n_clicks"),
        State("p2-results-store",     "data"),
        State("p2-suppressor-var",    "value"),
        State("p2-covariates",        "value"),
        State("p2-fdr-thresh",        "value"),
        State("theme-store",          "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def run_p2_suppressor(_, payload, suppressor, covariates, fdr_thresh,
                          theme, dcdq, rbs, scq, ados, cbcl, cov,
                          sensory, css):
        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), None, True
        if not suppressor:
            return dbc.Alert("Select a suppressor variable.", color="warning"), None, True

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return dbc.Alert("No data loaded.", color="warning"), None, True
        if suppressor not in mg.columns:
            return dbc.Alert(
                f"Suppressor '{suppressor}' not in data. "
                "Load the relevant instrument first.",
                color="warning"), None, True

        # Join sensory columns if used as predictors
        preds = payload.get("predictors", [])
        sdf   = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds if c in sdf.columns]
            if sens_cols:
                sens_cols = [c for c in sens_cols if c not in mg.columns]
                if sens_cols:
                    mg = mg.join(sdf[sens_cols], how="left")

        all_preds   = [p for p in preds if p in mg.columns]
        all_outs    = [o for o in payload.get("outcomes", []) if o in mg.columns]

        # Reuse the exact covariate specification from the Step-1 run.
        # This prevents the suppressor analysis from silently drifting if the
        # live covariate control is changed after Step 1 has already run.
        stored_covariates = payload.get("cov_cols", [])
        cov_present = [c for c in stored_covariates if c in mg.columns]
        thresh      = float(fdr_thresh or 0.05)

        if suppressor in all_preds:
            all_preds = [p for p in all_preds if p != suppressor]
        if suppressor in all_outs:
            all_outs = [o for o in all_outs if o != suppressor]

        if not all_preds or not all_outs:
            return dbc.Alert("No predictors or outcomes after removing suppressor.",
                             color="warning"), None, True

        # ── Run WITH and WITHOUT suppressor ──────────────────────────────
        base_res = run_mass_univariate(mg, all_preds, all_outs,
                                        cov_present)
        aug_res  = run_mass_univariate(mg, all_preds, all_outs,
                                        cov_present + [suppressor])

        if "error" in base_res or "error" in aug_res:
            err = base_res.get("error") or aug_res.get("error")
            return dbc.Alert(err, color="danger"), None, True

        base_dr2 = base_res["sqrt_dr2"]
        aug_dr2  = aug_res["sqrt_dr2"]
        delta    = aug_dr2 - base_dr2   # positive = revealed; negative = concealed

        t    = theme or "dark"
        dark = t == "dark"
        ax   = get_axis_style(t)
        mid  = "#0d0f14" if dark else "#f1f5f9"

        # ── Δ√ΔR² heatmap ────────────────────────────────────────────────
        x_labs = [c.replace("cbcl_", "CBCL ").replace("css_", "ADOS ")
                  .replace("_", " ") for c in delta.columns]
        y_labs = [r.replace("_", " ").replace("dcdq ", "DCDQ ")
                  .replace("rbs ", "RBS-R ").replace("scq ", "SCQ ")
                  .replace("sp ", "SP ").replace("seq ", "SEQ ")
                  .replace("isq ", "ISQ ") for r in delta.index]

        ann = [[f"{v:+.3f}" if not np.isnan(v) else ""
                for v in row]
               for row in delta.values.tolist()]

        abs_max = float(np.nanmax(np.abs(delta.values)))
        fig_delta = go.Figure(go.Heatmap(
            z=delta.values.tolist(),
            x=x_labs, y=y_labs,
            colorscale=[[0, "#f87171"], [0.5, mid], [1, "#34d399"]],
            zmid=0, zmin=-abs_max, zmax=abs_max,
            text=ann, texttemplate="%{text}", textfont={"size": 9},
            colorbar={"title": "Δ√ΔR²", "thickness": 12,
                      "tickfont": {"size": 9}},
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>"
                          "Δ√ΔR² = %{z:.4f}<extra></extra>",
        ))
        fig_delta.update_layout(**{
            **get_plotly_layout(t),
            "height": max(280, len(delta.index) * 38 + 100),
            "title": {"text": f"Δ√ΔR² when '{suppressor}' added as covariate",
                      "font": {"size": 11}},
            "margin": {"l": 140, "r": 80, "t": 44, "b": 90},
            "xaxis": {**ax, "tickangle": -35, "tickfont": {"size": 9}},
            "yaxis": {**ax, "tickfont": {"size": 9}, "autorange": "reversed"},
        })

        # ── Before vs after scatter ───────────────────────────────────────
        b_vec = base_dr2.values.flatten().astype(float)
        a_vec = aug_dr2.values.flatten().astype(float)
        mask  = ~(np.isnan(b_vec) | np.isnan(a_vec))
        b_m, a_m = b_vec[mask], a_vec[mask]

        preds_list = base_dr2.index.tolist()
        outs_list  = base_dr2.columns.tolist()
        labs = [f"{p} → {o}" for p in preds_list for o in outs_list]
        labs_m = [labs[i] for i, m in enumerate(mask) if m]

        colors_m = ["#34d399" if a > b else "#f87171"
                    for b, a in zip(b_m, a_m)]

        lo = float(min(b_m.min(), a_m.min())) - 0.01
        hi = float(max(b_m.max(), a_m.max())) + 0.01

        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=b_m.tolist(), y=a_m.tolist(),
            mode="markers",
            marker={"size": 8, "color": colors_m, "opacity": 0.8},
            text=labs_m,
            hovertemplate="%{text}<br>Base: %{x:.3f} → Aug: %{y:.3f}<extra></extra>",
        ))
        fig_scat.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line={"color": "#334155", "dash": "dash", "width": 1},
            showlegend=False, hoverinfo="skip",
        ))
        fig_scat.update_layout(**{
            **get_plotly_layout(t),
            "height": 320,
            "title": {"text": "√ΔR² before vs after adding suppressor",
                      "font": {"size": 11}},
            "margin": {"l": 60, "r": 30, "t": 44, "b": 60},
            "xaxis": {**ax, "title": "√ΔR² base model"},
            "yaxis": {**ax, "title": "√ΔR² + suppressor"},
        })

        # ── Summary stats ─────────────────────────────────────────────────
        d_vec = delta.values.flatten().astype(float)
        d_vec = d_vec[~np.isnan(d_vec)]
        n_concealed = int((d_vec < -0.01).sum())
        n_revealed  = int((d_vec >  0.01).sum())
        n_neutral   = len(d_vec) - n_concealed - n_revealed
        mean_delta  = float(np.mean(d_vec))
        max_conceal = float(np.min(d_vec))
        max_reveal  = float(np.max(d_vec))

        supp_label = suppressor.replace("cbcl_", "CBCL ").replace("css_", "ADOS ") \
                               .replace("_", " ")

        # ── Suppressor store payload ───────────────────────────────────────────
        supp_store = {
            "suppressor":    suppressor,
            "delta":         delta.to_dict(),
            "base_dr2":      base_dr2.to_dict(),
            "aug_dr2":       aug_dr2.to_dict(),
            "base_pval_raw": base_res["pval_raw"].to_dict(),
            "base_pval_fdr": base_res["pval_fdr"].to_dict(),
            "aug_pval_raw":  aug_res["pval_raw"].to_dict(),
            "aug_pval_fdr":  aug_res["pval_fdr"].to_dict(),
            "base_n_obs":    base_res["n_obs"].to_dict(),
            "aug_n_obs":     aug_res["n_obs"].to_dict(),
            "cov_cols":      list(cov_present),
            "predictors":    list(all_preds),
            "outcomes":      list(all_outs),
            "base_n_tests":  int(base_res["n_tests"]),
            "aug_n_tests":   int(aug_res["n_tests"]),
            "fdr_threshold": float(thresh),
            "source_run_id": payload.get("run_id", ""),
            "n_concealed":   int(n_concealed),
            "n_revealed":    int(n_revealed),
            "mean_delta":    float(mean_delta),
            "max_concealed": float(max_conceal),
            "max_revealed":  float(max_reveal),
        }
        return html.Div([
            # Summary bar
            html.Div([
                html.Span(f"Suppressor: {supp_label}  ·  ",
                          style={"fontWeight": "700", "fontSize": "12px"}),
                html.Span(f"{n_concealed} pairs concealed (Δ < −0.01)  ·  ",
                          style={"color": "#f87171", "fontSize": "11px"}),
                html.Span(f"{n_revealed} revealed (Δ > +0.01)  ·  ",
                          style={"color": "#34d399", "fontSize": "11px"}),
                html.Span(f"mean Δ√ΔR² = {mean_delta:+.4f}  ·  "
                          f"max concealed = {max_conceal:+.4f}  ·  "
                          f"max revealed = {max_reveal:+.4f}",
                          style={"color": "var(--text-muted)", "fontSize": "10px"}),
            ], style={"marginBottom": "14px"}),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scat,
                                  config={"displayModeBar": False}), width=5),
                dbc.Col(dcc.Graph(figure=fig_delta,
                                  config={"displayModeBar": True,
                                          "toImageButtonOptions": {
                                              "filename": "suppressor_delta",
                                              "format": "png", "scale": 2}}),
                        width=7),
            ]),

            html.Div(
                f"Δ√ΔR² = √ΔR²(with {supp_label}) − √ΔR²(base). "
                "Red = suppressor masks the association (negative shift); "
                "green = suppressor reveals latent association (positive shift). "
                "Points below the diagonal in the scatter are concealed. "
                "Covariates held constant in both models.",
                style={"fontSize": "10px", "color": "var(--text-muted)",
                       "marginTop": "10px"},
            ),
        ]), supp_store, False

    # ── Export callbacks ──────────────────────────────────────────────────────

    @app.callback(
        Output("p2-split-download", "data"),
        Input("btn-p2-split-export", "n_clicks"),
        State("p2-split-store", "data"),
        prevent_initial_call=True,
    )
    def export_p2_split(_, store):
        if not store:
            return no_update
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Concordance table
            disc = pd.DataFrame(store["disc_dr2"]).astype(float)
            rep  = pd.DataFrame(store["rep_dr2"]).astype(float)
            disc_q = pd.DataFrame(store.get("disc_pval_fdr", {})).astype(float) if store.get("disc_pval_fdr") else None
            rep_q  = pd.DataFrame(store.get("rep_pval_fdr",  {})).astype(float) if store.get("rep_pval_fdr")  else None

            rows = []
            for pred in disc.index:
                for out in disc.columns:
                    row = {
                        "predictor":            pred,
                        "outcome":              out,
                        "discovery_sqrt_dr2":   round(float(disc.loc[pred, out]), 6) if pred in disc.index and out in disc.columns else None,
                        "replication_sqrt_dr2": round(float(rep.loc[pred, out]),  6) if pred in rep.index  and out in rep.columns  else None,
                    }
                    if disc_q is not None and pred in disc_q.index and out in disc_q.columns:
                        row["discovery_q"]     = round(float(disc_q.loc[pred, out]), 6)
                        row["replication_q"]   = round(float(rep_q.loc[pred, out]),  6)
                    rows.append(row)
            conc_df = pd.DataFrame(rows)
            buf1 = io.StringIO()
            conc_df.to_csv(buf1, index=False, float_format="%.6f", na_rep="NA")
            zf.writestr("split_half_concordance.csv", buf1.getvalue())

            # Ridge table
            ridge_raw = store.get("ridge", [])
            if ridge_raw:
                ridge_df = pd.DataFrame(ridge_raw)
                ridge_out = pd.DataFrame({
                    "outcome":         ridge_df["outcome"],
                    "out_of_sample_r": ridge_df["r_test"].round(6),
                    "r2":              ridge_df["r2_test"].round(6),
                    "p":               ridge_df["p_test"].round(6),
                    "n":               ridge_df["n_rep"],
                })
                buf2 = io.StringIO()
                ridge_out.to_csv(buf2, index=False, float_format="%.6f", na_rep="NA")
                zf.writestr("split_half_ridge.csv", buf2.getvalue())

        return dcc.send_bytes(buf.getvalue(), filename="split_half_results.zip")

    @app.callback(
        Output("p2-suppress-download", "data"),
        Input("btn-p2-suppress-export", "n_clicks"),
        State("p2-suppress-store", "data"),
        prevent_initial_call=True,
    )
    def export_p2_suppress(_, store):
        if not store:
            return no_update
        import io
        delta   = pd.DataFrame(store["delta"]).astype(float)
        base    = pd.DataFrame(store["base_dr2"]).astype(float)
        aug     = pd.DataFrame(store["aug_dr2"]).astype(float)

        def _frame(key):
            raw = store.get(key)
            return pd.DataFrame(raw).astype(float) if raw else None

        base_p = _frame("base_pval_raw")
        base_q = _frame("base_pval_fdr")
        aug_p  = _frame("aug_pval_raw")
        aug_q  = _frame("aug_pval_fdr")
        base_n = _frame("base_n_obs")
        aug_n  = _frame("aug_n_obs")

        cov_str = "|".join(store.get("cov_cols", []))
        suppressor = store.get("suppressor", "result")
        rows = []
        for pred in delta.index:
            for out in delta.columns:
                def _val(frame):
                    if frame is None or pred not in frame.index or out not in frame.columns:
                        return np.nan
                    return float(frame.loc[pred, out])

                rows.append({
                    "predictor":       pred,
                    "outcome":         out,
                    "delta_sqrt_dr2":  float(delta.loc[pred, out]),
                    "base_sqrt_dr2":   float(base.loc[pred, out]),
                    "aug_sqrt_dr2":    float(aug.loc[pred, out]),
                    "base_p_raw":      _val(base_p),
                    "base_q_value":    _val(base_q),
                    "aug_p_raw":       _val(aug_p),
                    "aug_q_value":     _val(aug_q),
                    "base_n":          _val(base_n),
                    "aug_n":           _val(aug_n),
                    "suppressor":      suppressor,
                    "covariates":      cov_str,
                    "base_n_tests":    store.get("base_n_tests"),
                    "aug_n_tests":     store.get("aug_n_tests"),
                    "fdr_threshold":   store.get("fdr_threshold", 0.05),
                    "source_run_id":    store.get("source_run_id", ""),
                })
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False, float_format="%.6f", na_rep="NA")

        # Avoid path separators and other awkward filename characters.
        safe_suppressor = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in suppressor
        ).strip("._") or "result"
        return dcc.send_string(buf.getvalue(),
                               filename=f"suppressor_{safe_suppressor}.csv")

    @app.callback(
        Output("p2-pca-download", "data"),
        Input("btn-p2-pca-export", "n_clicks"),
        State("p2-pca-store", "data"),
        prevent_initial_call=True,
    )
    def export_p2_pca(_, store):
        if not store:
            return no_update
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            outs  = store.get("outcomes", [])
            preds = store.get("predictors", [])
            load  = store.get("outcome_loadings", {})
            scr   = store.get("predictor_scores", {})

            df_lo = pd.DataFrame(load, index=outs)
            df_lo.index.name = "outcome"
            buf1 = io.StringIO()
            df_lo.reset_index().to_csv(buf1, index=False, float_format="%.6f")
            zf.writestr("pca_outcome_loadings.csv", buf1.getvalue())

            df_sc = pd.DataFrame(scr, index=preds)
            df_sc.index.name = "predictor"
            buf2 = io.StringIO()
            df_sc.reset_index().to_csv(buf2, index=False, float_format="%.6f")
            zf.writestr("pca_predictor_scores.csv", buf2.getvalue())

            var_e = store.get("var_explained", [])
            var_df = pd.DataFrame({
                "PC":               [f"PC{i+1}" for i in range(len(var_e))],
                "variance_explained": [round(v, 6) for v in var_e],
                "pct_variance":       [round(v * 100, 2) for v in var_e],
            })
            buf3 = io.StringIO()
            var_df.to_csv(buf3, index=False)
            zf.writestr("pca_variance_explained.csv", buf3.getvalue())

        return dcc.send_bytes(buf.getvalue(), filename="pca_results.zip")


    # ── Step 5: Hubness Index ─────────────────────────────────────────────────

    @app.callback(
        Output("btn-p2-hub", "disabled"),
        Input("p2-results-store", "data"),
    )
    def enable_hub_btn(payload):
        return payload is None

    @app.callback(
        Output("p2-hub-content",   "children"),
        Output("btn-p2-hub-export", "disabled"),
        Input("btn-p2-hub",        "n_clicks"),
        State("p2-results-store",  "data"),
        State("p2-hub-fdr-thresh", "value"),
        State("p2-hub-sig-only",   "value"),
        State("theme-store",       "data"),
        prevent_initial_call=True,
    )
    def run_hubness(_, payload, fdr_thresh, sig_only_val, theme):
        from modules.hubness import compute_hubness
        import plotly.graph_objects as go

        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), True

        thresh   = float(fdr_thresh or 0.05)
        sig_only = (sig_only_val != "all")
        t        = theme or "dark"
        dark     = t == "dark"
        ax       = get_axis_style(t)
        layout   = get_plotly_layout(t)

        hub_df = compute_hubness(payload, fdr_thresh=thresh, sig_only=sig_only)

        if hub_df.empty:
            return dbc.Alert("No results to display.", color="warning"), True

        # ── Bar chart — hubness index ranked ──────────────────────────────
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=hub_df["hubness_index"],
            y=hub_df["predictor"].apply(
                lambda c: (c.replace("cbcl_", "CBCL ")
                            .replace("dcdq_", "DCDQ ")
                            .replace("rbs_", "RBS-R ")
                            .replace("scq_", "SCQ ")
                            .replace("_", " "))
            ),
            orientation="h",
            marker_color=hub_df["domain_color"],
            customdata=np.stack([
                hub_df["n_significant"],
                hub_df["mean_abs_effect"].round(3),
                hub_df["max_abs_effect"].round(3),
                hub_df["domain"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Hubness: %{x:.3f}<br>"
                "Significant outcomes: %{customdata[0]}<br>"
                "Mean |√ΔR²|: %{customdata[1]}<br>"
                "Max |√ΔR²|: %{customdata[2]}<br>"
                "Domain: %{customdata[3]}<extra></extra>"
            ),
        ))

        fig_bar.update_layout(
            **layout,
            xaxis_title="Hubness Index (Σ|√ΔR²| across significant outcomes)",
            yaxis=dict(**ax, autorange="reversed"),
            xaxis=dict(**ax),
            height=max(300, len(hub_df) * 32),
            margin=dict(l=160, r=40, t=30, b=40),
        )

        # ── Summary table ─────────────────────────────────────────────────
        table_rows = []
        for _, row in hub_df.iterrows():
            table_rows.append(html.Tr([
                html.Td(f"{int(row['rank'])}",
                        style={"fontWeight": "700", "textAlign": "center",
                               "width": "40px"}),
                html.Td(
                    row["predictor"].replace("dcdq_", "DCDQ ")
                                    .replace("rbs_", "RBS-R ")
                                    .replace("scq_", "SCQ ")
                                    .replace("cbcl_", "CBCL ")
                                    .replace("_", " "),
                    style={"color": row["domain_color"],
                           "fontWeight": "600", "fontSize": "11px"}
                ),
                html.Td(row["domain"],
                        style={"fontSize": "10px",
                               "color": "var(--text-muted)"}),
                html.Td(f"{row['hubness_index']:.3f}",
                        style={"fontWeight": "700", "textAlign": "right"}),
                html.Td(f"{int(row['n_significant'])}/{int(row['n_total'])}",
                        style={"textAlign": "center", "fontSize": "11px"}),
                html.Td(
                    f"{row['mean_abs_effect']:.3f}"
                    if not np.isnan(row["mean_abs_effect"]) else "—",
                    style={"textAlign": "right", "fontSize": "11px"}),
                html.Td(
                    f"{row['max_abs_effect']:.3f}"
                    if not np.isnan(row["max_abs_effect"]) else "—",
                    style={"textAlign": "right", "fontSize": "11px"}),
            ]))

        table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Rank"),
                    html.Th("Predictor"),
                    html.Th("Domain"),
                    html.Th("Hubness", style={"textAlign": "right"}),
                    html.Th("Sig / Total", style={"textAlign": "center"}),
                    html.Th("Mean |√ΔR²|", style={"textAlign": "right"}),
                    html.Th("Max |√ΔR²|", style={"textAlign": "right"}),
                ])),
                html.Tbody(table_rows),
            ],
            bordered=False, striped=True, hover=True, size="sm",
            style={"fontSize": "11px"},
        )

        # ── Method note ───────────────────────────────────────────────────
        n_sig_total = int(hub_df["n_significant"].sum())
        top = hub_df.iloc[0]
        method_note = html.Div([
            html.Span(
                f"Hubness = Σ|√ΔR²| across FDR-significant outcomes "
                f"(q < {thresh}). "
                f"Top hub: {top['predictor'].replace('rbs_','RBS-R ').replace('dcdq_','DCDQ ').replace('scq_','SCQ ').replace('_',' ')} "
                f"(hubness = {top['hubness_index']:.3f}, "
                f"{int(top['n_significant'])} significant outcomes). "
                f"Total significant predictor-outcome pairs: {n_sig_total}.",
                style={"fontSize": "10px", "color": "var(--text-muted)"},
            ),
        ], style={"marginTop": "8px", "marginBottom": "4px"})

        content = html.Div([
            dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": "var(--border)", "margin": "12px 0"}),
            table,
            method_note,
        ])

        return content, False

    @app.callback(
        Output("p2-hub-download", "data"),
        Input("btn-p2-hub-export", "n_clicks"),
        State("p2-results-store",  "data"),
        State("p2-hub-fdr-thresh", "value"),
        State("p2-hub-sig-only",   "value"),
        prevent_initial_call=True,
    )
    def export_hub(_, payload, fdr_thresh, sig_only_val):
        from modules.hubness import compute_hubness
        import io

        if not payload:
            return no_update

        thresh   = float(fdr_thresh or 0.05)
        sig_only = (sig_only_val != "all")
        hub_df   = compute_hubness(payload, fdr_thresh=thresh, sig_only=sig_only)

        buf = io.StringIO()
        hub_df.drop(columns=["domain_color"]).to_csv(buf, index=False,
                                                       float_format="%.6f")
        return dcc.send_string(buf.getvalue(), filename="hubness_index.csv")



    # ── Step 6: Age-Stratified Hub Stability ─────────────────────────────────

    @app.callback(
        Output("btn-p2-agehub", "disabled"),
        Input("p2-results-store", "data"),
    )
    def enable_agehub_btn(payload):
        return payload is None

    @app.callback(
        Output("p2-agehub-content",    "children"),
        Output("btn-p2-agehub-export", "disabled"),
        Input("btn-p2-agehub",         "n_clicks"),
        State("p2-results-store",      "data"),
        State("p2-agehub-fdr-thresh",  "value"),
        State("p2-covariates",         "value"),
        State("theme-store",           "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def run_age_hub(_, payload, fdr_thresh, covariates, theme,
                    dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        from modules.age_hub import run_age_stratified_hubness, AGE_BANDS
        from modules.domains import DOMAIN_COLORS, get_domain
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), True

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return dbc.Alert("No data loaded.", color="warning"), True

        if "age_months" not in mg.columns and "age_years" not in mg.columns:
            return dbc.Alert(
                "Age data not found. Load a covariates file that includes "
                "age_at_registration_months or age_at_registration_years.",
                color="warning"), True

        thresh      = float(fdr_thresh or 0.05)
        t           = theme or "dark"
        dark        = t == "dark"
        ax          = get_axis_style(t)
        layout_base = get_plotly_layout(t)

        # Resolve predictors and outcomes from the Step 1 payload
        preds       = payload.get("predictors", [])
        outs        = payload.get("outcomes", [])
        cov_present = [c for c in (covariates or []) if c in mg.columns]

        # Join any sensory columns not already in mg
        sdf = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds
                         if c in sdf.columns and c not in mg.columns]
            if sens_cols:
                mg = mg.join(sdf[sens_cols], how="left")

        all_preds = [p for p in preds if p in mg.columns]
        all_outs  = [o for o in outs  if o in mg.columns]

        result = run_age_stratified_hubness(
            mg, all_preds, all_outs, cov_present,
            age_col="age_months", fdr_thresh=thresh,
        )

        if "error" in result:
            return dbc.Alert(result["error"], color="danger"), True

        band_results  = result["band_results"]
        band_ns       = result["band_ns"]
        rank_matrix   = result["rank_matrix"]
        hub_matrix    = result["hub_matrix"]
        stability     = result["stability"]
        n_missing     = result["n_missing_age"]
        skipped       = result["skipped_bands"]
        band_labels   = list(band_results.keys())

        # ── Clean label helper ────────────────────────────────────────────
        def _cl(c):
            return (c.replace("dcdq_", "DCDQ ").replace("rbs_", "RBS-R ")
                     .replace("scq_", "SCQ ").replace("cbcl_", "CBCL ")
                     .replace("_", " "))

        # ── Figure 1: Hub rankings across age bands (grouped bar) ─────────
        fig_rank = go.Figure()
        n_bands  = len(band_labels)
        for i, band in enumerate(band_labels):
            hub_df = band_results[band]
            preds_sorted = hub_df["predictor"].tolist()
            fig_rank.add_trace(go.Bar(
                name=f"{band} (n={band_ns[band]:,})",
                x=[_cl(p) for p in preds_sorted],
                y=hub_df["hubness_index"].tolist(),
                marker_color=hub_df["domain_color"].tolist(),
                opacity=0.5 + 0.5 * (i / max(n_bands - 1, 1)),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Band: {band}<br>"
                    "Hubness: %{y:.3f}<extra></extra>"
                ),
            ))

        fig_rank.update_layout(
            **layout_base,
            barmode="group",
            xaxis=dict(**ax, tickangle=-35),
            yaxis=dict(**ax, title="Hubness Index (Σ|√ΔR²|)"),
            height=380,
            margin=dict(l=60, r=20, t=30, b=100),
        )
        fig_rank.update_layout(legend=dict(font=dict(size=10)))

        # ── Figure 2: Rank heatmap across bands ──────────────────────────
        # Rows = predictors sorted by mean rank across bands
        # Columns = age bands
        # Cell = rank within that band (1 = top hub)
        rm = rank_matrix[band_labels].copy()
        rm["mean_rank"] = rm.mean(axis=1)
        rm = rm.sort_values("mean_rank")
        rm = rm.drop(columns="mean_rank")

        pred_labels_sorted = [_cl(p) for p in rm.index.tolist()]

        fig_heat = go.Figure(go.Heatmap(
            z=rm.values.tolist(),
            x=band_labels,
            y=pred_labels_sorted,
            colorscale="RdBu",    # rank 1 (top hub) = red, rank 10 = blue
            reversescale=False,
            colorbar=dict(title="Rank<br>(1=top hub)",
                          thickness=12, tickfont=dict(size=9)),
            text=[[str(int(v)) if not np.isnan(v) else "—"
                   for v in row]
                  for row in rm.values.tolist()],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Band: %{x}<br>"
                "Rank: %{z}<extra></extra>"
            ),
        ))
        fig_heat.update_layout(
            **layout_base,
            xaxis=dict(**ax),
            yaxis=dict(**ax),
            height=max(300, len(rm) * 28 + 80),
            margin=dict(l=160, r=60, t=30, b=40),
        )

        # ── Stability summary table ───────────────────────────────────────
        pairs = stability["pairs"]
        mean_r = stability["mean_r"]

        pair_rows = [
            html.Tr([
                html.Td(p["band_a"], style={"fontSize": "11px"}),
                html.Td(p["band_b"], style={"fontSize": "11px"}),
                html.Td(f"{p['spearman_r']:.3f}",
                        style={"fontWeight": "700", "textAlign": "right"}),
                html.Td(
                    f"{p['p_value']:.4f}",
                    style={"textAlign": "right", "fontSize": "11px",
                           "color": ("var(--text-muted)"
                                     if p["p_value"] >= 0.05 else "inherit")}
                ),
                html.Td(str(p["n_predictors"]),
                        style={"textAlign": "center", "fontSize": "11px"}),
            ])
            for p in pairs
        ]

        stability_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Band A"),
                    html.Th("Band B"),
                    html.Th("Spearman r", style={"textAlign": "right"}),
                    html.Th("p-value",    style={"textAlign": "right"}),
                    html.Th("N predictors", style={"textAlign": "center"}),
                ])),
                html.Tbody(pair_rows),
            ],
            bordered=False, striped=True, hover=True, size="sm",
            style={"fontSize": "11px", "maxWidth": "600px"},
        )

        # ── Data availability note ────────────────────────────────────────
        avail_items = [
            html.Li(
                f"{label}: n = {band_ns[label]:,}"
                + (" ✓" if label in band_results else
                   f" ✗ skipped (below minimum n=50)"),
                style={"fontSize": "11px"}
            )
            for label in [b["label"] for b in AGE_BANDS]
        ]
        if skipped:
            avail_items += [
                html.Li(f"Skipped: {s}",
                        style={"fontSize": "11px",
                               "color": "var(--text-muted)"})
                for s in skipped
            ]

        avail_note = html.Div([
            html.Div("Sample sizes per age band:",
                     style={"fontWeight": "600", "fontSize": "11px",
                            "marginBottom": "4px"}),
            html.Ul(avail_items, style={"marginBottom": "8px",
                                        "paddingLeft": "20px"}),
            html.Div(
                f"Participants excluded due to missing age data: "
                f"{n_missing:,} of {result['n_total']:,} "
                f"({100*n_missing/max(result['n_total'],1):.1f}%)",
                style={"fontSize": "10px", "color": "var(--text-muted)"}
            ),
        ])

        # ── Mean stability badge ──────────────────────────────────────────
        r_color = ("#34d399" if mean_r >= 0.80
                   else "#fbbf24" if mean_r >= 0.60
                   else "#f87171")
        stability_badge = html.Div([
            html.Span("Mean hub ranking concordance across bands: ",
                      style={"fontSize": "12px"}),
            html.Span(f"Spearman r = {mean_r:.3f}",
                      style={"fontSize": "14px", "fontWeight": "700",
                             "color": r_color}),
            html.Span(
                "  (≥0.80 strong  ·  0.60–0.79 moderate  ·  <0.60 weak)",
                style={"fontSize": "10px", "color": "var(--text-muted)",
                       "marginLeft": "8px"}),
        ], style={"marginBottom": "12px",
                  "padding": "8px",
                  "borderRadius": "4px",
                  "border": f"1px solid {r_color}33"})

        content = html.Div([
            stability_badge,
            html.Div("Hub rankings across age bands",
                     style={"fontWeight": "600", "fontSize": "12px",
                            "marginBottom": "4px"}),
            dcc.Graph(figure=fig_rank, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": "var(--border)",
                           "margin": "16px 0"}),
            html.Div("Hub rank by age band (1 = top hub)",
                     style={"fontWeight": "600", "fontSize": "12px",
                            "marginBottom": "4px"}),
            dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": "var(--border)",
                           "margin": "16px 0"}),
            html.Div("Pairwise rank stability",
                     style={"fontWeight": "600", "fontSize": "12px",
                            "marginBottom": "8px"}),
            stability_table,
            html.Hr(style={"borderColor": "var(--border)",
                           "margin": "16px 0"}),
            avail_note,
        ])

        return content, False

    @app.callback(
        Output("p2-agehub-download", "data"),
        Input("btn-p2-agehub-export", "n_clicks"),
        State("p2-results-store",     "data"),
        State("p2-agehub-fdr-thresh", "value"),
        State("p2-covariates",        "value"),
        State("theme-store",          "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def export_agehub(_, payload, fdr_thresh, covariates, theme,
                      dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        from modules.age_hub import run_age_stratified_hubness
        import io, zipfile

        if not payload:
            return no_update

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return no_update

        thresh      = float(fdr_thresh or 0.05)
        preds       = payload.get("predictors", [])
        outs        = payload.get("outcomes", [])
        cov_present = [c for c in (covariates or []) if c in mg.columns]

        sdf = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds
                         if c in sdf.columns and c not in mg.columns]
            if sens_cols:
                mg = mg.join(sdf[sens_cols], how="left")

        all_preds = [p for p in preds if p in mg.columns]
        all_outs  = [o for o in outs  if o in mg.columns]

        result = run_age_stratified_hubness(
            mg, all_preds, all_outs, cov_present,
            age_col="age_months", fdr_thresh=thresh,
        )

        if "error" in result:
            return no_update

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Hub rankings per band
            for label, hub_df in result["band_results"].items():
                safe = label.replace(" ", "_").replace("/", "-")
                s = io.StringIO()
                hub_df.drop(columns=["domain_color"]).to_csv(
                    s, index=False, float_format="%.6f")
                zf.writestr(f"hub_{safe}.csv", s.getvalue())

            # Rank matrix
            s = io.StringIO()
            result["rank_matrix"].to_csv(s, float_format="%.1f")
            zf.writestr("rank_matrix.csv", s.getvalue())

            # Hub index matrix
            s = io.StringIO()
            result["hub_matrix"].to_csv(s, float_format="%.6f")
            zf.writestr("hub_matrix.csv", s.getvalue())

            # Stability
            s = io.StringIO()
            pd.DataFrame(result["stability"]["pairs"]).to_csv(
                s, index=False, float_format="%.6f")
            zf.writestr("stability_spearman.csv", s.getvalue())

        return dcc.send_bytes(buf.getvalue(),
                              filename="age_stratified_hubness.zip")



    # ── Step 7: Equal-N Sensitivity ──────────────────────────────────────────

    @app.callback(
        Output("btn-p2-equaln", "disabled"),
        Input("p2-results-store", "data"),
    )
    def enable_equaln_btn(payload):
        return payload is None

    @app.callback(
        Output("p2-equaln-content",    "children"),
        Output("btn-p2-equaln-export", "disabled"),
        Input("btn-p2-equaln",         "n_clicks"),
        State("p2-results-store",      "data"),
        State("p2-agehub-fdr-thresh",  "value"),
        State("p2-covariates",         "value"),
        State("p2-equaln-iters",       "value"),
        State("p2-equaln-seed",        "value"),
        State("theme-store",           "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def run_equaln(_, payload, fdr_thresh, covariates, n_iters, seed, theme,
                   dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        from modules.age_hub import run_equaln_sensitivity
        import plotly.graph_objects as go

        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), True

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return dbc.Alert("No data loaded.", color="warning"), True

        if "age_months" not in mg.columns and "age_years" not in mg.columns:
            return dbc.Alert(
                "Age data not found. Load covariates file.", color="warning"), True

        thresh      = float(fdr_thresh or 0.05)
        n_iters     = int(n_iters or 5)
        seed        = int(seed or 42)
        t           = theme or "dark"
        dark        = t == "dark"
        ax          = get_axis_style(t)
        layout_base = get_plotly_layout(t)

        preds       = payload.get("predictors", [])
        outs        = payload.get("outcomes",   [])
        cov_present = [c for c in (covariates or []) if c in mg.columns]

        sdf = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds
                         if c in sdf.columns and c not in mg.columns]
            if sens_cols:
                mg = mg.join(sdf[sens_cols], how="left")

        all_preds = [p for p in preds if p in mg.columns]
        all_outs  = [o for o in outs  if o in mg.columns]

        result = run_equaln_sensitivity(
            mg, all_preds, all_outs, cov_present,
            age_col="age_months", fdr_thresh=thresh,
            seed=seed, n_iterations=n_iters,
        )

        if "error" in result:
            return dbc.Alert(result["error"], color="danger"), True

        summary  = result["summary"]
        equaln_n = result["equaln_n"]
        band_ns  = result["band_ns_full"]
        t_ax     = get_axis_style(t)
        layout   = get_plotly_layout(t)

        # ── Bar chart: mean r per comparison ─────────────────────────────
        summary_sorted = summary.sort_values("mean_r", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=summary_sorted["comparison"],
            y=summary_sorted["mean_r"],
            error_y=dict(type="data", array=summary_sorted["sd_r"].tolist(),
                         visible=True, thickness=1.5, width=4),
            marker_color=[
                "#34d399" if r >= 0.80
                else "#fbbf24" if r >= 0.60
                else "#f87171" if r >= 0
                else "#94a3b8"
                for r in summary_sorted["mean_r"]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean r = %{y:.3f}<br>"
                "SD = %{error_y.array:.3f}<extra></extra>"
            ),
        ))
        fig.add_hline(y=0.80, line_dash="dot", line_color="#34d399",
                      line_width=1, annotation_text="Strong (0.80)",
                      annotation_font_size=9)
        fig.add_hline(y=0.60, line_dash="dot", line_color="#fbbf24",
                      line_width=1, annotation_text="Moderate (0.60)",
                      annotation_font_size=9)
        fig.add_hline(y=0, line_color="gray", line_width=0.8)

        fig.update_layout(
            **layout,
            xaxis=dict(**t_ax, tickangle=-20),
            yaxis=dict(**t_ax, title="Mean Spearman r (hub ranking)",
                       range=[-1, 1]),
            height=320,
            margin=dict(l=60, r=20, t=30, b=80),
        )

        # ── Summary table ─────────────────────────────────────────────────
        table_rows = [
            html.Tr([
                html.Td(row["comparison"], style={"fontSize": "11px"}),
                html.Td(f"{row['mean_r']:.3f}",
                        style={"fontWeight": "700", "textAlign": "right",
                               "color": (
                                   "#34d399" if row["mean_r"] >= 0.80
                                   else "#fbbf24" if row["mean_r"] >= 0.60
                                   else "#f87171" if row["mean_r"] >= 0
                                   else "#94a3b8"
                               )}),
                html.Td(f"± {row['sd_r']:.3f}",
                        style={"textAlign": "right", "fontSize": "11px",
                               "color": "var(--text-muted)"}),
                html.Td(str(row["n_iter"]),
                        style={"textAlign": "center", "fontSize": "11px"}),
            ])
            for _, row in summary_sorted.iterrows()
        ]

        table = dbc.Table(
            [html.Thead(html.Tr([
                html.Th("Comparison"),
                html.Th("Mean r",   style={"textAlign": "right"}),
                html.Th("± SD",     style={"textAlign": "right"}),
                html.Th("Iters",    style={"textAlign": "center"}),
            ])),
             html.Tbody(table_rows)],
            bordered=False, striped=True, hover=True, size="sm",
            style={"fontSize": "11px", "maxWidth": "500px"},
        )

        # ── Interpretation note ───────────────────────────────────────────
        # Compare full-sample pattern vs equal-N pattern
        full_ns_str = "  ·  ".join(
            f"{k}: n={v:,}" for k, v in band_ns.items()
        )

        # Determine interpretation
        childhood_key = "4–8 yrs vs 9–12 yrs"
        childhood_r   = summary[summary["comparison"] == childhood_key
                                 ]["mean_r"].values
        late_keys     = [k for k in summary["comparison"]
                         if "13–17" in k or "18+" in k]
        late_rs       = summary[summary["comparison"].isin(late_keys)
                                ]["mean_r"].values

        if len(childhood_r) > 0 and len(late_rs) > 0:
            if childhood_r[0] > 0.70 and np.mean(late_rs) < 0.50:
                interp = (
                    "The equal-N analysis confirms the full-sample finding: "
                    "strong hub ranking concordance in childhood (4–8 vs 9–12 yrs) "
                    "but weaker concordance involving adolescent/adult bands. "
                    "This suggests the developmental shift in hub structure reflects "
                    "genuine phenotypic change rather than a sample size artifact."
                )
                interp_color = "#34d399"
            elif np.mean(late_rs) >= 0.60:
                interp = (
                    "Equal-N subsampling partially attenuates the weaker concordance "
                    "in later bands, suggesting sample size differences contributed "
                    "to the pattern observed in the full analysis. "
                    "Developmental shift interpretation should be made with caution."
                )
                interp_color = "#fbbf24"
            else:
                interp = (
                    "Hub ranking concordance remains weak across bands even at "
                    "equal N, consistent with genuine developmental reorganization "
                    "of cross-domain behavioral structure."
                )
                interp_color = "#f87171"
        else:
            interp = "Insufficient band pairs for interpretation."
            interp_color = "#94a3b8"

        note = html.Div([
            html.Div(
                f"Equal-N subsample: n = {equaln_n:,} per band "
                f"(80% of smallest band) × {n_iters} iterations  ·  "
                f"Original Ns: {full_ns_str}",
                style={"fontSize": "10px", "color": "var(--text-muted)",
                       "marginBottom": "6px"}),
            html.Div(interp,
                     style={"fontSize": "11px", "color": interp_color,
                            "padding": "8px",
                            "border": f"1px solid {interp_color}44",
                            "borderRadius": "4px"}),
        ], style={"marginTop": "10px"})

        content = html.Div([
            html.Div(
                f"Mean Spearman r across {n_iters} equal-N iterations "
                f"(n = {equaln_n:,} per band)",
                style={"fontWeight": "600", "fontSize": "12px",
                       "marginBottom": "4px"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": "var(--border)",
                           "margin": "12px 0"}),
            table,
            note,
        ])

        return content, False

    @app.callback(
        Output("p2-equaln-download", "data"),
        Input("btn-p2-equaln-export", "n_clicks"),
        State("p2-results-store",     "data"),
        State("p2-agehub-fdr-thresh", "value"),
        State("p2-covariates",        "value"),
        State("p2-equaln-iters",      "value"),
        State("p2-equaln-seed",       "value"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def export_equaln(_, payload, fdr_thresh, covariates, n_iters, seed,
                      dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        from modules.age_hub import run_equaln_sensitivity
        import io

        if not payload:
            return no_update

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return no_update

        thresh      = float(fdr_thresh or 0.05)
        preds       = payload.get("predictors", [])
        outs        = payload.get("outcomes",   [])
        cov_present = [c for c in (covariates or []) if c in mg.columns]

        sdf = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds
                         if c in sdf.columns and c not in mg.columns]
            if sens_cols:
                mg = mg.join(sdf[sens_cols], how="left")

        all_preds = [p for p in preds if p in mg.columns]
        all_outs  = [o for o in outs  if o in mg.columns]

        result = run_equaln_sensitivity(
            mg, all_preds, all_outs, cov_present,
            age_col="age_months", fdr_thresh=thresh,
            seed=int(seed or 42), n_iterations=int(n_iters or 5),
        )

        if "error" in result:
            return no_update

        buf = io.StringIO()
        result["summary"].to_csv(buf, index=False, float_format="%.6f")
        return dcc.send_string(buf.getvalue(),
                               filename="equaln_sensitivity.csv")



    # ── Network Analysis Data Export ─────────────────────────────────────────

    @app.callback(
        Output("btn-p2-network-export", "disabled"),
        Input("p2-results-store", "data"),
    )
    def enable_network_export(payload):
        return payload is None

    @app.callback(
        Output("p2-network-download", "data"),
        Input("btn-p2-network-export", "n_clicks"),
        State("p2-results-store",      "data"),
        State("p2-agehub-fdr-thresh",  "value"),
        State("theme-store",           "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def export_for_network(_, payload, fdr_thresh, theme,
                           dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        from modules.hubness import compute_hubness
        import io, zipfile

        if not payload:
            return no_update

        mg = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        if mg is None:
            return no_update

        preds       = payload.get("predictors", [])
        outs        = payload.get("outcomes",   [])
        thresh      = float(fdr_thresh or 0.05)

        # Join sensory columns if needed
        sdf = df_from_store(sensory) if sensory else None
        if sdf is not None:
            sens_cols = [c for c in preds
                         if c in sdf.columns and c not in mg.columns]
            if sens_cols:
                mg = mg.join(sdf[sens_cols], how="left")

        all_preds = [p for p in preds if p in mg.columns]
        all_outs  = [o for o in outs  if o in mg.columns]
        all_vars  = all_preds + [o for o in all_outs
                                  if o not in all_preds]

        # Raw data: all predictors + outcomes, complete cases
        # Export with missingness intact — R uses pairwise correlations
        # Listwise deletion across 19 variables collapses N drastically
        raw_df = mg[all_vars]

        # Community assignments for bridge() in R
        from modules.domains import get_domain
        community_df = pd.DataFrame({
            "variable":  all_vars,
            "community": [get_domain(v) for v in all_vars],
        })

        # Hubness index
        hub_df = compute_hubness(payload, fdr_thresh=thresh, sig_only=True)

        # sqrt_dr2 matrix
        sqrt_dr2_df = pd.DataFrame(payload["sqrt_dr2"])

        # Covariate info for README
        cov_cols = payload.get("cov_cols", [])

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

            # 1. Raw data
            s = io.StringIO()
            raw_df.to_csv(s, index=True, float_format="%.6f")
            zf.writestr("raw_data.csv", s.getvalue())

            # 2. Community assignments
            s = io.StringIO()
            community_df.to_csv(s, index=False)
            zf.writestr("communities.csv", s.getvalue())

            # 3. Hubness index
            s = io.StringIO()
            hub_df.drop(columns=["domain_color"]).to_csv(
                s, index=False, float_format="%.6f")
            zf.writestr("hubness_index.csv", s.getvalue())

            # 4. sqrt_dr2 matrix
            s = io.StringIO()
            sqrt_dr2_df.to_csv(s, float_format="%.6f")
            zf.writestr("sqrt_dr2_matrix.csv", s.getvalue())

            # 5. README with variable lists
            readme = f"""Network Analysis Data Export
============================
Generated by SPARK Behavioral Fingerprint App

FILES
-----
raw_data.csv        : Participant-level scores (N={len(raw_df)} rows, pairwise complete used in R)
                      Columns: {', '.join(all_vars[:5])}... ({len(all_vars)} variables total)
                      Use for: partial correlation network estimation (qgraph)

communities.csv     : Community assignments per variable (domain labels)
                      Use for: bridge() community argument in networktools

hubness_index.csv   : Predictor hubness rankings from sqrt(DeltaR2) analysis
                      Use for: convergence test with bridge centrality

sqrt_dr2_matrix.csv : Full predictor x outcome effect size matrix
                      Use for: reference and supplementary reporting

COVARIATES USED IN sqrt(DeltaR2) ANALYSIS
------------------------------------------
{', '.join(cov_cols) if cov_cols else 'None'}
Note: Covariates are NOT included in raw_data.csv.
The network analysis uses raw scores (no covariate adjustment).
This is intentional — the convergence test asks whether network-derived
bridge rankings agree with covariate-adjusted sqrt(DeltaR2) hub rankings,
testing whether hub structure is method-invariant.

PREDICTORS ({len(all_preds)})
-----------
{chr(10).join(all_preds)}

OUTCOMES ({len(all_outs)})
----------
{chr(10).join(all_outs)}

R SCRIPT
--------
Run: Rscript network_analysis.R
(Requires: qgraph, networktools, bootnet — see install_packages.R)
"""
            zf.writestr("README.txt", readme)

        return dcc.send_bytes(buf.getvalue(),
                              filename="network_analysis_data.zip")

