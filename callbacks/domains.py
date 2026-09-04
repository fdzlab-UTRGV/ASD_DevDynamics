"""
callbacks/domains.py
─────────────────────────────────────────────────────────────────────────────
Domain √ΔR² tab callbacks.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from helpers.store import get_merged_data, df_from_store
from helpers.theme import get_plotly_layout, get_axis_style
from modules.mass_univariate import run_mass_univariate
from modules.domains import (
    PREDICTOR_DOMAINS, PREDICTOR_DOMAIN_ORDER, DOMAIN_COLORS,
    compute_domain_composites, sort_by_domain, get_domain,
    domain_band_shapes, domain_tick_colors, make_legend_traces,
)
from modules.split_half import split_sample_matched
from scipy import stats as scipy_stats

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
CSS_OUTCOMES = [
    ("css", "css_total",  "ADOS CSS-Total"),
    ("css", "css_sa",     "ADOS CSS-SA"),
    ("css", "css_rrb",    "ADOS CSS-RRB"),
]


def _get_merged(*vals):
    keys = ["dcdq", "rbs", "scq", "ados", "cbcl", "cov", "sensory", "css"]
    return get_merged_data(**dict(zip(keys, vals)))


def _sig_marker(q, thresh):
    if np.isnan(q): return ""
    if q < thresh * 0.001: return "***"
    if q < thresh * 0.01:  return "**"
    if q < thresh:         return "*"
    return ""


def _clean(c):
    return (c.replace("cbcl_", "CBCL ").replace("css_", "ADOS ")
             .replace("dcdq_", "DCDQ ").replace("rbs_", "RBS-R ")
             .replace("scq_", "SCQ ").replace("_", " "))


def _build_heatmap(mat, pval_fdr, n_obs, thresh, t):
    """Shared heatmap builder used by Step 1 and Split-half."""
    dark = t == "dark"
    ax   = get_axis_style(t)
    mid  = "#0d0f14" if dark else "#f1f5f9"

    # Domain-band columns (outcomes)
    out_order = sort_by_domain(mat.columns.tolist())
    mat      = mat[out_order]
    pval_fdr = pval_fdr[out_order]
    n_obs    = n_obs[out_order]

    # Row order = PREDICTOR_DOMAIN_ORDER (already composites)
    row_order = [d for d in PREDICTOR_DOMAIN_ORDER if d in mat.index]
    mat      = mat.loc[row_order]
    pval_fdr = pval_fdr.loc[row_order]
    n_obs    = n_obs.loc[row_order]

    x_labs = [_clean(c) for c in mat.columns]
    y_labs = mat.index.tolist()

    ann = []
    for pred in mat.index:
        row = []
        for out in mat.columns:
            v = mat.loc[pred, out]
            q = pval_fdr.loc[pred, out]
            n = n_obs.loc[pred, out]
            row.append(f"{v:.3f}{_sig_marker(q, thresh)}<br>"
                       f"<span style='font-size:8px'>n={int(n)}</span>"
                       if not np.isnan(v) else "")
        ann.append(row)

    # Row colours from domain palette
    row_cols = [DOMAIN_COLORS.get(d, "#94a3b8") for d in y_labs]
    col_shapes = domain_band_shapes(mat.columns.tolist(), axis="x", dark=dark)
    col_cols   = domain_tick_colors(mat.columns.tolist())

    fig = go.Figure(go.Heatmap(
        z=mat.values.tolist(),
        x=x_labs, y=y_labs,
        colorscale=[[0, "#34d399"], [0.5, mid], [1, "#f87171"]],
        zmid=0,
        text=ann, texttemplate="%{text}", textfont={"size": 9},
        colorbar={"title": "√ΔR²", "thickness": 12, "tickfont": {"size": 9}},
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>√ΔR² = %{z:.4f}<extra></extra>",
    ))

    # Coloured row tick annotations (by domain colour)
    row_anns = [
        {"x": -0.01, "y": i, "xref": "paper", "yref": "y",
         "text": f"<b><span style='color:{row_cols[i]}'>{lbl}</span></b>",
         "showarrow": False, "xanchor": "right", "font": {"size": 11}}
        for i, lbl in enumerate(y_labs)
    ]
    col_anns = [
        {"x": j, "y": -0.01, "xref": "x", "yref": "paper",
         "text": f"<span style='color:{col_cols[j]}'>{lbl}</span>",
         "showarrow": False, "yanchor": "top",
         "textangle": -35, "font": {"size": 9}}
        for j, lbl in enumerate(x_labs)
    ]

    # Domain legend
    for tr in make_legend_traces():
        fig.add_trace(go.Scatter(**tr))

    n_pred = len(mat.index)
    n_out  = len(mat.columns)
    fig.update_layout(**{
        **get_plotly_layout(t),
        "height": max(260, n_pred * 60 + 140),
        "margin": {"l": 180, "r": 80, "t": 40, "b": 130},
        "shapes":      col_shapes,
        "annotations": row_anns + col_anns,
        "xaxis": {**ax, "showticklabels": False, "tickfont": {"size": 9}},
        "yaxis": {**ax, "showticklabels": False, "tickfont": {"size": 9},
                  "autorange": "reversed"},
        "legend": {"font": {"size": 9,
                            "color": "#e2e8f0" if dark else "#1e293b"},
                   "x": 1.08},
    })
    return fig


def register(app):

    # ── Populate outcome checklist + feature counts ───────────────────────────
    @app.callback(
        Output("dom-outcomes",       "options"),
        Output("dom-outcomes",       "value"),
        Output("dom-feature-counts", "children"),
        Input("main-tabs",           "active_tab"),
        Input("cbcl-store",          "data"),   # re-populate when outcomes arrive
        Input("css-store",           "data"),   # re-populate when ADOS severity computed
        *SOURCE_STATES,
    )
    def populate_dom(tab, _cbcl_trigger, _css_trigger,
                     dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        if tab != "tab-domains":
            return no_update, no_update, no_update

        mg  = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        sdf = df_from_store(sensory) if sensory else None

        # Merge sensory columns in for feature count
        mg_full = mg
        if mg is not None and sdf is not None:
            mg_full = mg.join(sdf.drop(columns=[c for c in sdf.columns
                                                if c in mg.columns],
                                       errors="ignore"),
                              how="left")

        all_cols = set(mg_full.columns) if mg_full is not None else set()

        # Outcomes
        out_opts, out_vals = [], []
        for scale, domain in PART2_OUTCOMES:
            col = f"{scale}_{domain}"
            if col in all_cols:
                out_opts.append({"label": f"CBCL {domain}", "value": col})
                out_vals.append(col)
        for _, col, label in CSS_OUTCOMES:
            if col in all_cols:
                out_opts.append({"label": label, "value": col})
                out_vals.append(col)

        # Feature count per domain
        counts = []
        for domain in PREDICTOR_DOMAIN_ORDER:
            feats = [f for f in PREDICTOR_DOMAINS.get(domain, [])
                     if f in all_cols]
            col   = DOMAIN_COLORS.get(domain, "#94a3b8")
            counts.append(html.Div([
                html.Span(f"● {domain}: ",
                          style={"color": col, "fontWeight": "600"}),
                html.Span(f"{len(feats)} features",
                          style={"color": "var(--text-muted)"}),
            ]))

        return out_opts, out_vals, counts

    # ── Step 1: Run domain analysis ───────────────────────────────────────────
    @app.callback(
        Output("dom-content",       "children"),
        Output("dom-results-store", "data"),
        Output("btn-dom-export",    "disabled"),
        Output("btn-dom-pca",       "disabled"),
        Output("btn-dom-split",     "disabled"),
        Input("btn-dom-run",        "n_clicks"),
        State("dom-predictors",     "value"),
        State("dom-outcomes",       "value"),
        State("dom-covariates",     "value"),
        State("dom-fdr-thresh",     "value"),
        State("theme-store",        "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def run_domain(_, domains, out_vals, covariates, fdr_thresh, theme,
                   dcdq, rbs, scq, ados, cbcl, cov, sensory, css):

        if not domains or not out_vals:
            return (dbc.Alert("Select at least one domain and one outcome.",
                              color="warning"), None, True, True, True)

        mg  = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        sdf = df_from_store(sensory) if sensory else None
        if mg is None:
            return dbc.Alert("No data.", color="warning"), None, True, True, True

        # Join sensory columns
        if sdf is not None:
            sens_cols = [c for c in sdf.columns if c not in mg.columns]
            mg = mg.join(sdf[sens_cols], how="left")

        cov_present = [c for c in (covariates or []) if c in mg.columns]
        all_outs    = [o for o in out_vals if o in mg.columns]
        thresh      = float(fdr_thresh or 0.05)

        if not all_outs:
            return (dbc.Alert("None of the selected outcomes are in the data.",
                              color="warning"), None, True, True, True)

        # ── Compute composites ────────────────────────────────────────────
        composites = compute_domain_composites(mg, domains)
        if composites.empty:
            return (dbc.Alert("No constituent features found for any domain.",
                              color="warning"), None, True, True, True)

        # Join composites to mg for mass_univariate (it needs them as columns)
        analysis_df = mg.join(composites, rsuffix="_dom")
        dom_cols = composites.columns.tolist()

        result = run_mass_univariate(analysis_df, dom_cols, all_outs, cov_present)
        if "error" in result:
            return dbc.Alert(result["error"], color="danger"), None, True, True, True

        t = theme or "dark"
        fig = _build_heatmap(result["sqrt_dr2"], result["pval_fdr"],
                             result["n_obs"], thresh, t)

        # Summary
        flat_sr2 = result["sqrt_dr2"].values.flatten().astype(float)
        flat_fdr = result["pval_fdr"].values.flatten().astype(float)
        flat_sr2 = flat_sr2[~np.isnan(flat_sr2)]
        flat_fdr = flat_fdr[~np.isnan(flat_fdr)]
        n_sig    = int((flat_fdr < thresh).sum())

        cov_str = ", ".join(cov_present) or "none"
        summary = html.Div([
            html.Span(f"✓ {result['n_tests']} tests  ·  ",
                      style={"color": "var(--success)", "fontWeight": "700"}),
            html.Span(f"{n_sig} significant (FDR q < {thresh})  ·  ",
                      style={"color": "var(--accent)"}),
            html.Span(f"Covariates: {cov_str}  ·  ",
                      style={"color": "var(--text-muted)"}),
            html.Span(f"Max |√ΔR²| = {np.max(np.abs(flat_sr2)):.3f}  "
                      f"median = {np.median(np.abs(flat_sr2)):.3f}",
                      style={"color": "var(--text-muted)"}),
        ], style={"fontSize": "11px", "marginBottom": "10px"})

        note = html.Div([
            html.Div(
                "Each predictor row is a domain composite (z-score mean across "
                "all constituent features from all loaded instruments). "
                "Outcome columns are individual measures banded by construct. "
                f"* q<{thresh}  ** q<{thresh/5:.3f}  *** q<{thresh/500:.4f}",
                style={"fontSize": "10px", "color": "var(--text-muted)",
                       "marginTop": "8px"}),
        ])

        payload = {
            "sqrt_dr2":   result["sqrt_dr2"].to_dict(),
            "pval_fdr":   result["pval_fdr"].to_dict(),
            "n_obs":      result["n_obs"].to_dict(),
            "beta":       result["beta"].to_dict(),
            "domains":    dom_cols,
            "outcomes":   all_outs,
            "cov_cols":   cov_present,
            "n_tests":    result["n_tests"],
        }

        return (html.Div([summary,
                          dcc.Graph(figure=fig,
                                    config={"displayModeBar": True,
                                            "toImageButtonOptions": {
                                                "filename": "domain_sqrt_dr2",
                                                "format": "png", "scale": 2}}),
                          note]),
                payload, False, False, False)

    # ── Export CSV ────────────────────────────────────────────────────────────
    @app.callback(
        Output("dom-download", "data"),
        Input("btn-dom-export", "n_clicks"),
        State("dom-results-store", "data"),
        prevent_initial_call=True,
    )
    def export_dom(_, payload):
        if not payload: return no_update
        mat  = pd.DataFrame(payload["sqrt_dr2"])
        long = mat.reset_index().melt(id_vars="index",
                                      var_name="outcome",
                                      value_name="sqrt_dr2")
        long.rename(columns={"index": "domain"}, inplace=True)
        fdr  = pd.DataFrame(payload["pval_fdr"])
        long["pval_fdr"] = fdr.reset_index().melt(
            id_vars="index", var_name="outcome",
            value_name="pval_fdr")["pval_fdr"].values
        return dcc.send_data_frame(long.to_csv, "domain_sqrt_dr2.csv",
                                   index=False)

    # ── Step 2: PCA ───────────────────────────────────────────────────────────
    @app.callback(
        Output("dom-pca-content",      "children"),
        Output("dom-pca-store",        "data"),
        Output("btn-dom-pca-export",   "disabled"),
        Input("btn-dom-pca",           "n_clicks"),
        State("dom-results-store","data"),
        State("theme-store",      "data"),
        prevent_initial_call=True,
    )
    def dom_pca(_, payload, theme):
        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), None, True

        mat = pd.DataFrame(payload["sqrt_dr2"]).astype(float).fillna(0)
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return dbc.Alert("Not enough data for PCA.", color="warning"), None, True

        Z   = mat.values.copy()
        Z  -= Z.mean(axis=0)
        std = Z.std(axis=0); std[std == 0] = 1; Z /= std
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        var_exp  = (s**2) / (s**2).sum()

        t  = theme or "dark"
        ax = get_axis_style(t)

        # Outcome loadings on PC1
        fig_load = go.Figure(go.Bar(
            x=mat.columns.tolist(),
            y=Vt[0].tolist(),
            marker_color=[DOMAIN_COLORS.get(get_domain(c), "#94a3b8")
                          for c in mat.columns],
        ))
        fig_load.update_layout(**{
            **get_plotly_layout(t),
            "title": {"text": f"PC1 outcome loadings ({var_exp[0]*100:.1f}% variance)",
                      "font": {"size": 11}},
            "height": 240,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 90},
            "xaxis": {**ax, "tickangle": -35, "tickfont": {"size": 9}},
            "yaxis": {**ax, "title": "Loading"},
        })

        # Domain scores on PC1
        domain_scores = (U[:, 0] * s[0]).tolist()
        dom_names = mat.index.tolist()
        dom_cols  = [DOMAIN_COLORS.get(d, "#94a3b8") for d in dom_names]

        fig_score = go.Figure(go.Bar(
            x=dom_names, y=domain_scores,
            marker_color=dom_cols,
            text=[f"{v:.3f}" for v in domain_scores],
            textposition="outside",
        ))
        fig_score.update_layout(**{
            **get_plotly_layout(t),
            "title": {"text": "PC1 domain scores", "font": {"size": 11}},
            "height": 220,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "xaxis": {**ax}, "yaxis": {**ax, "title": "Score"},
        })

        # ── Domain PCA store ──────────────────────────────────────────────────
        n_pcs_d = min(5, len(s))
        dom_pca_store = {
            "var_explained":    var_exp[:n_pcs_d].tolist(),
            "outcomes":         mat.columns.tolist(),
            "predictors":       mat.index.tolist(),
            "outcome_loadings": {f"PC{k+1}": Vt[k].tolist() for k in range(n_pcs_d)},
            "predictor_scores": {f"PC{k+1}": (U[:, k] * s[k]).tolist() for k in range(n_pcs_d)},
        }
        return html.Div([
            html.Div(
                f"PCA of {mat.shape[0]} domains × {mat.shape[1]} outcomes √ΔR² matrix. "
                f"PC1 explains {var_exp[0]*100:.1f}% of total variance.",
                style={"fontSize": "11px", "color": "var(--text-muted)",
                       "marginBottom": "10px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_score,
                                  config={"displayModeBar": False}), width=4),
                dbc.Col(dcc.Graph(figure=fig_load,
                                  config={"displayModeBar": False}), width=8),
            ]),
        ]), dom_pca_store, False

    # ── Step 3: Split-half ────────────────────────────────────────────────────
    @app.callback(
        Output("dom-split-content",      "children"),
        Output("dom-split-store",        "data"),
        Output("btn-dom-split-export",   "disabled"),
        Input("btn-dom-split",           "n_clicks"),
        State("dom-results-store",   "data"),
        State("dom-split-seed",      "value"),
        State("dom-covariates",      "value"),
        State("theme-store",         "data"),
        *SOURCE_STATES,
        prevent_initial_call=True,
    )
    def dom_split(_, payload, seed, covariates, theme,
                  dcdq, rbs, scq, ados, cbcl, cov, sensory, css):
        if not payload:
            return dbc.Alert("Run Step 1 first.", color="warning"), None, True

        mg  = _get_merged(dcdq, rbs, scq, ados, cbcl, cov, sensory, css)
        sdf = df_from_store(sensory) if sensory else None
        if mg is None:
            return dbc.Alert("No data.", color="warning"), None, True
        if sdf is not None:
            mg = mg.join(sdf.drop(columns=[c for c in sdf.columns
                                           if c in mg.columns],
                                  errors="ignore"), how="left")

        cov_present = [c for c in (covariates or []) if c in mg.columns]
        all_outs    = [o for o in payload.get("outcomes", []) if o in mg.columns]
        domains     = payload.get("domains", PREDICTOR_DOMAIN_ORDER)
        thresh      = 0.05
        t           = theme or "dark"

        disc, rep = split_sample_matched(mg, seed=int(seed or 42))

        # Compute composites independently in each half
        def _run_half(half):
            comp = compute_domain_composites(half, domains)
            hdf  = half.join(comp, rsuffix="_dom")
            return run_mass_univariate(hdf, domains, all_outs, cov_present)

        disc_res = _run_half(disc)
        rep_res  = _run_half(rep)

        if "error" in disc_res or "error" in rep_res:
            return dbc.Alert("Analysis failed on one or both halves.",
                             color="danger"), None, True

        d_vec = disc_res["sqrt_dr2"].values.flatten().astype(float)
        r_vec = rep_res["sqrt_dr2"].values.flatten().astype(float)
        mask  = ~(np.isnan(d_vec) | np.isnan(r_vec))

        if mask.sum() >= 3:
            conc_r, conc_p = scipy_stats.pearsonr(d_vec[mask], r_vec[mask])
        else:
            conc_r, conc_p = np.nan, np.nan

        p_str   = ("p < .001" if conc_p < 0.001
                   else f"p = {conc_p:.3f}" if not np.isnan(conc_p) else "—")
        r_color = ("#34d399" if abs(conc_r) > 0.6
                   else "#fbbf24" if abs(conc_r) > 0.3
                   else "#f87171")
        ax = get_axis_style(t)

        # Concordance scatter
        labs  = [f"{d} → {o}"
                 for d in disc_res["sqrt_dr2"].index
                 for o in disc_res["sqrt_dr2"].columns]
        labs_m = [labs[i] for i, m in enumerate(mask) if m]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d_vec[mask].tolist(), y=r_vec[mask].tolist(),
            mode="markers",
            marker={"size": 8, "color": "#38bdf8", "opacity": 0.8},
            text=labs_m,
            hovertemplate="%{text}<br>Disc: %{x:.3f}  Rep: %{y:.3f}<extra></extra>",
        ))
        lo = float(min(d_vec[mask].min(), r_vec[mask].min())) - 0.01
        hi = float(max(d_vec[mask].max(), r_vec[mask].max())) + 0.01
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line={"color": "#334155", "dash": "dash",
                                       "width": 1},
                                 showlegend=False, hoverinfo="skip"))
        fig.update_layout(**{
            **get_plotly_layout(t),
            "height": 320,
            "title": {"text": (f"Domain √ΔR² concordance  "
                               f"r = <span style='color:{r_color}'>"
                               f"{conc_r:.3f}</span>  ({p_str})"),
                      "font": {"size": 11}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
            "xaxis": {**ax, "title": "Discovery √ΔR²"},
            "yaxis": {**ax, "title": "Replication √ΔR²"},
        })

        # Side-by-side heatmaps
        fig_disc = _build_heatmap(disc_res["sqrt_dr2"], disc_res["pval_fdr"],
                                   disc_res["n_obs"], thresh, t)
        fig_rep  = _build_heatmap(rep_res["sqrt_dr2"],  rep_res["pval_fdr"],
                                   rep_res["n_obs"],  thresh, t)
        fig_disc.update_layout(title={"text": f"Discovery (n={len(disc):,})",
                                       "font": {"size": 11}},
                               height=340)
        fig_rep.update_layout(title={"text": f"Replication (n={len(rep):,})",
                                      "font": {"size": 11}},
                              height=340)

        # ── Domain split store ────────────────────────────────────────────────
        dom_split_store = {
            "concordance_r":   float(conc_r),
            "concordance_p":   float(conc_p) if not np.isnan(conc_p) else None,
            "n_disc":          len(disc),
            "n_rep":           len(rep),
            "n_pairs":         int(mask.sum()),
            "disc_dr2":        disc_res["sqrt_dr2"].to_dict(),
            "rep_dr2":         rep_res["sqrt_dr2"].to_dict(),
            "disc_pval_fdr":   disc_res["pval_fdr"].to_dict(),
            "rep_pval_fdr":    rep_res["pval_fdr"].to_dict(),
            "disc_n_obs":      disc_res["n_obs"].to_dict(),
            "rep_n_obs":       rep_res["n_obs"].to_dict(),
        }
        return html.Div([
            html.Div([
                html.Span("Concordance r = ",
                          style={"color": "var(--text-muted)",
                                 "fontSize": "12px"}),
                html.Span(f"{conc_r:.3f}",
                          style={"fontSize": "20px", "fontWeight": "700",
                                 "color": r_color}),
                html.Span(f"  ({p_str})  ·  {int(mask.sum())} domain×outcome pairs  "
                          f"·  Disc n={len(disc):,}  Rep n={len(rep):,}",
                          style={"fontSize": "11px",
                                 "color": "var(--text-muted)"}),
            ], style={"marginBottom": "14px"}),

            # Concordance scatter — full width
            dcc.Graph(figure=fig,
                      config={"displayModeBar": False},
                      style={"marginBottom": "16px"}),

            # Discovery and replication heatmaps side by side
            html.Div("Discovery half",
                     style={"fontSize": "12px", "fontWeight": "700",
                            "marginBottom": "4px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_disc,
                                  config={"displayModeBar": False}),
                        width=6),
                dbc.Col(dcc.Graph(figure=fig_rep,
                                  config={"displayModeBar": False}),
                        width=6),
            ]),
        ]), dom_split_store, False

    # ── Export callbacks ──────────────────────────────────────────────────────

    @app.callback(
        Output("dom-pca-download", "data"),
        Input("btn-dom-pca-export", "n_clicks"),
        State("dom-pca-store", "data"),
        prevent_initial_call=True,
    )
    def export_dom_pca(_, store):
        if not store:
            return no_update
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            outs  = store.get("outcomes", [])
            doms  = store.get("predictors", [])
            load  = store.get("outcome_loadings", {})
            scr   = store.get("predictor_scores", {})

            df_lo = pd.DataFrame(load, index=outs)
            df_lo.index.name = "outcome"
            buf1 = io.StringIO()
            df_lo.reset_index().to_csv(buf1, index=False, float_format="%.6f")
            zf.writestr("domain_pca_outcome_loadings.csv", buf1.getvalue())

            df_sc = pd.DataFrame(scr, index=doms)
            df_sc.index.name = "domain"
            buf2 = io.StringIO()
            df_sc.reset_index().to_csv(buf2, index=False, float_format="%.6f")
            zf.writestr("domain_pca_domain_scores.csv", buf2.getvalue())

            var_e = store.get("var_explained", [])
            var_df = pd.DataFrame({
                "PC":                [f"PC{i+1}" for i in range(len(var_e))],
                "variance_explained": [round(v, 6) for v in var_e],
                "pct_variance":       [round(v * 100, 2) for v in var_e],
            })
            buf3 = io.StringIO()
            var_df.to_csv(buf3, index=False)
            zf.writestr("domain_pca_variance_explained.csv", buf3.getvalue())

        return dcc.send_bytes(buf.getvalue(), filename="domain_pca_results.zip")

    @app.callback(
        Output("dom-split-download", "data"),
        Input("btn-dom-split-export", "n_clicks"),
        State("dom-split-store", "data"),
        prevent_initial_call=True,
    )
    def export_dom_split(_, store):
        if not store:
            return no_update
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            disc   = pd.DataFrame(store["disc_dr2"]).astype(float)
            rep    = pd.DataFrame(store["rep_dr2"]).astype(float)
            disc_q = pd.DataFrame(store.get("disc_pval_fdr", {})).astype(float) if store.get("disc_pval_fdr") else None
            rep_q  = pd.DataFrame(store.get("rep_pval_fdr",  {})).astype(float) if store.get("rep_pval_fdr")  else None

            rows = []
            for dom in disc.index:
                for out in disc.columns:
                    row = {
                        "domain":               dom,
                        "outcome":              out,
                        "discovery_sqrt_dr2":   round(float(disc.loc[dom, out]), 6),
                        "replication_sqrt_dr2": round(float(rep.loc[dom, out]),  6),
                    }
                    if disc_q is not None and dom in disc_q.index and out in disc_q.columns:
                        row["discovery_q"]   = round(float(disc_q.loc[dom, out]), 6)
                        row["replication_q"] = round(float(rep_q.loc[dom, out]),  6)
                    rows.append(row)

            conc_df = pd.DataFrame(rows)
            buf1 = io.StringIO()
            conc_df.to_csv(buf1, index=False, float_format="%.6f", na_rep="NA")
            zf.writestr("domain_split_half_concordance.csv", buf1.getvalue())

        return dcc.send_bytes(buf.getvalue(), filename="domain_split_half_results.zip")
