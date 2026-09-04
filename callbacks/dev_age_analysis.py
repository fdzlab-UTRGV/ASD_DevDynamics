"""callbacks/dev_age_analysis.py"""
from __future__ import annotations
import io, datetime, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, html, dcc, no_update
import dash_bootstrap_components as dbc

from helpers.store import get_merged_data, has_data
from helpers.theme import get_plotly_layout
from modules.dev_age_analysis import (
    run_all, DOMAINS, DOMAIN_ORDER, T1_BANDS,
)

BANDS = list(T1_BANDS.keys())


def register(app):

    @app.callback(
        Output("daa-run","children"), Output("daa-run","color"),
        Input("daa-sections","value"), Input("daa-run","n_clicks"),
    )
    def stale(*_):
        from dash import ctx
        return ("Run analysis","primary") if ctx.triggered_id=="daa-run" \
               else ("⟳ Re-run","warning")

    @app.callback(
        Output("daa-results","children"),
        Output("dev-age-results-store","data"),
        Input("daa-run","n_clicks"),
        State("daa-sections","value"),
        State("dcdq-store","data"), State("rbs-store","data"),
        State("scq-store","data"),  State("ados-store","data"),
        State("cbcl-store","data"), State("cov-store","data"),
        State("sensory-store","data"), State("css-store","data"),
        prevent_initial_call=True,
    )
    def run(n, sections, dcdq_s, rbs_s, scq_s, ados_s,
            cbcl_s, cov_s, sensory_s, css_s):
        if not has_data(cbcl_s):
            return _err("Upload CBCL data to run this analysis."), no_update
        merged = get_merged_data(
            dcdq=dcdq_s, rbs=rbs_s, scq=scq_s, ados=ados_s,
            cbcl=cbcl_s, cov=cov_s, sensory=sensory_s, css=css_s)
        if merged is None or merged.empty:
            return _err("No data after merging."), no_update

        result = run_all(merged)
        if "error" in result:
            return _err(result["error"]), no_update

        children = _render(result, sections or [])

        # Serialise (drop _sub column before JSON)
        cells_ser = result["cells"].drop(columns=["_sub"], errors="ignore")
        store = {
            "cells":      cells_ser.to_dict("records"),
            "anova":      result["anova"],
            "tukey":      result["tukey"].to_dict("records")
                          if hasattr(result["tukey"],"to_dict") else {},
            "segmented":  {k:v for k,v in result["segmented"].items()
                           if k not in ["x","y","b_lin","b_seg1","b_seg2"]},
            "bayes_cell": result["bayes_cell"],
            "bayes_ind":  result["bayes_ind"],
            "bayes_domain_trends": result["bayes_domain_trends"],
            "glm":        result["glm"],
            "glm_logistic": result["glm_logistic"],
            "mixed":        result["mixed"],
        }
        return children, json.dumps(store)

    @app.callback(
        Output("daa-download","data"),
        Input("daa-save","n_clicks"),
        State("dev-age-results-store","data"),
        prevent_initial_call=True,
    )
    def save(n, store_data):
        if not store_data: return no_update
        d  = json.loads(store_data)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xl:
            # Cells
            pd.DataFrame(d["cells"]).to_excel(
                xl, sheet_name="Cells", index=False)

            # ANOVA
            av = d.get("anova", {})
            if av and "error" not in av:
                # Domain main effect
                dm = av.get("domain", {})
                anova_rows = [{"effect": "Domain (main)",
                                "F": dm.get("F"), "df1": dm.get("df1"),
                                "df2": dm.get("df2"), "p": dm.get("p"),
                                "sig": dm.get("sig","")}]
                # Per-domain trends
                for t in av.get("domain_trends", []):
                    anova_rows.append({
                        "effect": f"Age trend — {t['domain']}",
                        "slope":  t.get("slope"),
                        "F":      t.get("F"),
                        "df1":    t.get("df1"),
                        "df2":    t.get("df2"),
                        "p":      t.get("p"),
                        "sig":    t.get("sig",""),
                    })
                pd.DataFrame(anova_rows).to_excel(
                    xl, sheet_name="ANOVA", index=False)

            # Tukey HSD
            if d.get("tukey"):
                pd.DataFrame(d["tukey"]).to_excel(
                    xl, sheet_name="Tukey HSD", index=False)

            # Segmented regression
            sg = d.get("segmented", {})
            if sg and "error" not in sg:
                seg_rows = []
                for r in sg.get("ranked", []):
                    seg_rows.append({
                        "rank":     r.get("rank"),
                        "domain":   r.get("domain"),
                        "band":     r.get("band"),
                        "sqrt_dr2": r.get("sqrt_dr2"),
                        "segment":  r.get("segment", ""),
                    })
                if seg_rows:
                    seg_df = pd.DataFrame(seg_rows)
                    # Add summary stats
                    summary_rows = [
                        {"metric": "F",              "value": sg.get("F")},
                        {"metric": "df1",            "value": sg.get("df1")},
                        {"metric": "df2",            "value": sg.get("df2")},
                        {"metric": "p",              "value": sg.get("p")},
                        {"metric": "breakpoint_rank","value": sg.get("breakpoint")},
                        {"metric": "breakpoint_label","value": sg.get("bp_label")},
                        {"metric": "seg1_mean",      "value": sg.get("seg1_mean")},
                        {"metric": "seg2_mean",      "value": sg.get("seg2_mean")},
                        {"metric": "seg1_labels",
                         "value": ", ".join(sg.get("seg1_labels", []))},
                        {"metric": "seg2_labels",
                         "value": ", ".join(sg.get("seg2_labels", []))},
                    ]
                    seg_df.to_excel(xl, sheet_name="Segmented regression",
                                    index=False)
                    pd.DataFrame(summary_rows).to_excel(
                        xl, sheet_name="Segmented summary", index=False)

            # Bayesian models
            if d.get("bayes_cell", {}).get("params"):
                pd.DataFrame(d["bayes_cell"]["params"]).to_excel(
                    xl, sheet_name="Bayes cell", index=False)
            if d.get("bayes_ind", {}).get("params"):
                pd.DataFrame(d["bayes_ind"]["params"]).to_excel(
                    xl, sheet_name="Bayes individual", index=False)
            bdt = d.get("bayes_domain_trends", {})
            if bdt and "by_domain" in bdt:
                all_trend_rows = []
                for dom, res in bdt["by_domain"].items():
                    if "params" in res:
                        for p in res["params"]:
                            p["domain"] = dom
                            all_trend_rows.append(p)
                if all_trend_rows:
                    pd.DataFrame(all_trend_rows).to_excel(
                        xl, sheet_name="Bayes domain trends", index=False)

            # GLM
            gm = d.get("glm", {})
            if gm and "error" not in gm:
                glm_rows = []
                for effect, label in [
                    ("domain_score",      "domain_score (main)"),
                    ("domain_main",       "Domain (main)"),
                    ("band_main",         "Band (main)"),
                    ("score×domain",      "score × Domain"),
                    ("score×band",        "score × Band"),
                    ("score×domain×band", "score × Domain × Band"),
                ]:
                    e = gm.get(effect, {})
                    if e and isinstance(e, dict):
                        p = e.get("p", "n/a")
                        glm_rows.append({
                            "effect": label,
                            "F":      e.get("F", "n/a"),
                            "df1":    e.get("df1", "n/a"),
                            "df2":    e.get("df2", "n/a"),
                            "p":      p,
                            "sig":    (e.get("sig","n/a") if "sig" in e else
                                       "n/a" if p == "n/a" else
                                       "***" if float(p) < 0.001 else
                                       "**"  if float(p) < 0.01  else
                                       "*"   if float(p) < 0.05  else "ns"),
                        })
                if glm_rows:
                    pd.DataFrame(glm_rows).to_excel(
                        xl, sheet_name="GLM Type III", index=False)
                if gm.get("coefficients"):
                    _sanitize_for_excel(pd.DataFrame(gm["coefficients"])).to_excel(
                        xl, sheet_name="GLM coefficients", index=False)

            # Logistic GLM
            gl = d.get("glm_logistic", {})
            if gl and "error" not in gl:
                if gl.get("coefficients"):
                    pd.DataFrame(gl["coefficients"]).to_excel(
                        xl, sheet_name="GLM logistic coef", index=False)
                if gl.get("predictions"):
                    pd.DataFrame(gl["predictions"]).to_excel(
                        xl, sheet_name="GLM logistic pred", index=False)

            # Mixed models
            mx = d.get("mixed", {})
            if mx and "error" not in mx:
                if mx.get("glm",{}).get("coefficients"):
                    df_glm = _sanitize_for_excel(
                        pd.DataFrame(mx["glm"]["coefficients"]))
                    df_glm["model"] = "Population GLM"
                    df_slope_glm = _sanitize_for_excel(
                        pd.DataFrame(mx["glm"]["slopes"]))
                    df_slope_glm["model"] = "Population GLM"
                    df_glm.to_excel(xl, sheet_name="Mixed - GLM coef", index=False)
                    df_slope_glm.to_excel(xl, sheet_name="Mixed - GLM slopes", index=False)
                if mx.get("mixed",{}).get("coefficients"):
                    df_mix = _sanitize_for_excel(
                        pd.DataFrame(mx["mixed"]["coefficients"]))
                    df_mix["model"] = "Mixed-effects"
                    # Debug: print any cell value that might be illegal
                    for col in df_mix.columns:
                        for val in df_mix[col]:
                            if isinstance(val, str):
                                for ci, ch in enumerate(val):
                                    if ord(ch) < 32 or (0xD800 <= ord(ch) <= 0xDFFF):
                                        print(f"[Excel debug] Illegal char U+{ord(ch):04X} "
                                              f"at pos {ci} in: {repr(val)}")
                    df_slope_mix = _sanitize_for_excel(
                        pd.DataFrame(mx["mixed"]["slopes"]))
                    df_slope_mix["model"] = "Mixed-effects"
                    df_mix.to_excel(xl, sheet_name="Mixed - LME coef", index=False)
                    df_slope_mix.to_excel(xl, sheet_name="Mixed - LME slopes", index=False)

        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), f"dev_age_{ts}.xlsx")


# ── Rendering ─────────────────────────────────────────────────────────────────

def _err(msg):
    return dbc.Alert(msg, color="warning", style={"fontSize":"12px"})

def _sig_badge(sig):
    if sig in ("***","**","*"):
        return html.Span(sig, style={"color":"white","fontWeight":"700",
                                     "fontSize":"11px"})
    return html.Span("ns", style={"color":"#666","fontSize":"10px"})

def _render(result, sections):
    from modules.domains import DOMAIN_COLORS
    panels = []
    cells  = result["cells"]

    # ── Main figure: √ΔR² trajectories ────────────────────────────────────
    fig = go.Figure()
    mode = get_plotly_layout("light")
    mode["legend"] = dict(
        orientation="h", y=-0.22, xanchor="center", x=0.5,
        font=dict(size=11, family="Arial"),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    )

    for dom in DOMAIN_ORDER:
        color = DOMAIN_COLORS.get(
            "Sensory" if dom == "Sensory-Repetitive" else dom, "#94a3b8")
        sub   = cells[cells["domain"] == dom]
        ys, es, sigs, ns = [], [], [], []
        for band in BANDS:
            row = sub[sub["band"] == band]
            if row.empty:
                ys.append(None); es.append(None)
                sigs.append(""); ns.append("")
            else:
                r = row.iloc[0]
                ys.append(r["sqrt_dr2"])
                # cap SE display to avoid giant bars
                es.append(min(r["se_sqrt_dr2"], 0.08))
                sigs.append(r["sig"] if r["sig"] != "ns" else "")
                ns.append(f"{int(r['n']):,}")

        fig.add_trace(go.Scatter(
            x=BANDS, y=ys,
            mode="lines+markers+text",
            name=dom,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color,
                        line=dict(color="white", width=1.5)),
            error_y=dict(type="data", array=es, visible=True,
                         color=color, thickness=1.5, width=5),
            text=[f" {s}" if s else "" for s in sigs],
            textposition="top center",
            textfont=dict(color="white", size=12, family="Arial"),
            hovertemplate=(
                f"<b>{dom}</b><br>"
                "Band: %{x}<br>"
                "√ΔR²: %{y:.4f}<br>"
                "n: %{customdata}<extra></extra>"
            ),
            customdata=ns,
        ))

    fig.add_hline(y=0, line_width=1, line_dash="dot",
                  line_color="#888", opacity=0.5)
    fig.update_layout(
        **mode,
        height=360,
        title=dict(
            text="√ΔR² per domain × age band → Psychopathology (CBCL)",
            font=dict(size=13, family="Arial", color="#1e293b"),
            x=0.01, xanchor="left",
        ),
        xaxis=dict(
            title=dict(text="Predictor age band",
                       font=dict(size=11, family="Arial")),
            tickfont=dict(size=10, family="Arial"),
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="#cbd5e1",
        ),
        yaxis=dict(
            title=dict(text="√ΔR² (above age + sex)",
                       font=dict(size=11, family="Arial")),
            tickfont=dict(size=10, family="Arial"),
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="#cbd5e1",
            zeroline=False,
        ),
        margin=dict(t=50, b=100, l=65, r=20),
    )
    panels.append(dcc.Graph(figure=fig,
                            config={"displayModeBar": False},
                            style={"marginBottom": "14px"}))

    # ── Cell table ─────────────────────────────────────────────────────────
    if "cells" in sections:
        head = ["Domain", "Band", "n", "√ΔR²", "SE", "p (FDR)", ""]
        trows = [html.Tr([
            html.Th(h, style={"fontSize": "9px",
                              "color": "var(--text-muted)",
                              "paddingRight": "12px",
                              "textAlign": "left",
                              "paddingBottom": "4px",
                              "fontFamily": "Arial"})
            for h in head])]
        prev_dom = None
        for _, r in cells.sort_values(["domain", "band"]).iterrows():
            c = DOMAIN_COLORS.get(
                "Sensory" if r["domain"] == "Sensory-Repetitive"
                else r["domain"], "#94a3b8")
            dom_cell = html.Td(
                html.Span(r["domain"],
                          style={"color": c, "fontWeight": "700",
                                 "fontSize": "10px"})
                if r["domain"] != prev_dom else "",
                style={"paddingRight": "12px"})
            prev_dom = r["domain"]
            trows.append(html.Tr([
                dom_cell,
                html.Td(r["band"],
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "color": "var(--text-muted)"}),
                html.Td(f"{int(r['n']):,}",
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "color": "var(--text-muted)"}),
                html.Td(f"{r['sqrt_dr2']:+.3f}",
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "fontFamily": "monospace"}),
                html.Td(f"{r['se_sqrt_dr2']:.3f}",
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "fontFamily": "monospace"}),
                html.Td("<0.001" if r["pval_fdr"] < 0.001
                        else f"{r['pval_fdr']:.3f}",
                        style={"fontSize": "10px", "paddingRight": "8px"}),
                html.Td(_sig_badge(r["sig"])),
            ]))
        panels.append(_section("Cell √ΔR² estimates",
            html.Table(trows,
                       style={"borderCollapse": "collapse",
                              "marginBottom": "4px"})))

    # ── ANOVA ──────────────────────────────────────────────────────────────
    if "anova" in sections:
        av = result["anova"]
        if "error" not in av:
            from modules.dev_age_analysis import DOMAIN_ORDER as _DO
            from modules.domains import DOMAIN_COLORS as _DCA

            # ── Part 1: Domain main effect ────────────────────────────────
            dm = av["domain"]
            p_dm = dm["p"]
            part1 = html.Div([
                html.Div("Part 1 — Do the three domain trajectories differ in level?",
                         style={"fontWeight":"600","fontSize":"11px",
                                "marginBottom":"5px"}),
                html.Table([html.Tr([
                    html.Td("Domain main effect",
                            style={"fontSize":"11px","paddingRight":"16px",
                                   "fontWeight":"600"}),
                    html.Td(f"F({dm['df1']},{dm['df2']}) = {dm['F']}",
                            style={"fontSize":"11px","paddingRight":"16px",
                                   "fontFamily":"monospace"}),
                    html.Td(f"p = {'<0.001' if p_dm<0.001 else p_dm}",
                            style={"fontSize":"11px","paddingRight":"12px"}),
                    html.Td(_sig_badge(dm["sig"])),
                ])], style={"borderCollapse":"collapse","marginBottom":"4px"}),
                html.Div("One-way ANOVA on 11 cell estimates. "
                         "Tests whether Sensory-Repetitive, Motor, and Social "
                         "trajectories operate at significantly different levels.",
                         style={"fontSize":"9px","color":"var(--text-muted)",
                                "marginBottom":"12px"}),
            ])

            # ── Part 2: Per-domain age trends ─────────────────────────────
            trends = av.get("domain_trends", [])
            trend_rows = []
            for t in trends:
                color = _DCA.get(
                    "Sensory" if t["domain"]=="Sensory-Repetitive"
                    else t["domain"], "#94a3b8")
                F_str = f"F({t.get('df1','?')},{t.get('df2','?')}) = {t.get('F','n/a')}"
                p_val = t.get("p", np.nan)
                p_str = (f"p = {'<0.001' if isinstance(p_val,float) and p_val<0.001 else p_val}"
                         if p_val is not None and not (isinstance(p_val,float) and np.isnan(p_val))
                         else "n/a")
                sl_str = (f"slope = {t['slope']:+.4f}" if t.get("slope") is not None
                          and not np.isnan(t.get("slope",np.nan)) else "")
                trend_rows.append(html.Tr([
                    html.Td(t["domain"],
                            style={"fontSize":"11px","paddingRight":"16px",
                                   "fontWeight":"600","color":color}),
                    html.Td(sl_str,
                            style={"fontSize":"11px","paddingRight":"16px",
                                   "fontFamily":"monospace"}),
                    html.Td(F_str,
                            style={"fontSize":"11px","paddingRight":"16px",
                                   "fontFamily":"monospace"}),
                    html.Td(p_str,
                            style={"fontSize":"11px","paddingRight":"12px"}),
                    html.Td(_sig_badge(t.get("sig","ns"))
                            if p_val is not None and not (isinstance(p_val,float) and np.isnan(p_val))
                            else html.Span()),
                    html.Td(t.get("note",""),
                            style={"fontSize":"9px","color":"var(--text-muted)"}),
                ]))

            part2 = html.Div([
                html.Div("Part 2 — Does each domain's trajectory change with age?",
                         style={"fontWeight":"600","fontSize":"11px",
                                "marginBottom":"5px"}),
                html.Table(trend_rows,
                           style={"borderCollapse":"collapse","marginBottom":"4px"}),
                html.Div("Weighted linear regression within each domain "
                         "(√ΔR² ~ age band index, weighted by 1/SE²). "
                         "Tests whether the domain line has a significant slope across bands. "
                         "Motor × 0–4y excluded.",
                         style={"fontSize":"9px","color":"var(--text-muted)",
                                "marginBottom":"12px"}),
            ])

            # ── Part 3: Note on interaction ───────────────────────────────
            part3 = html.Div(
                "Part 3 — Does the age slope differ between domains? "
                "Tested by GLM on individual data (score × Domain × Band, "
                "F(6,62826) = 2.66, p = 0.014). "
                "Cell-level ANOVA has df_error = 5 for the interaction — "
                "too low to interpret reliably.",
                style={"fontSize":"9px","color":"var(--text-muted)",
                       "fontStyle":"italic"})

            panels.append(_section(
                f"ANOVA  ·  n = {av['n_cells']} cells  ·  weighted by 1/SE²",
                html.Div([part1, part2, part3])))


    # ── Tukey HSD ──────────────────────────────────────────────────────────
    if "tukey" in sections:
        tk = result["tukey"]
        if isinstance(tk, pd.DataFrame) and not tk.empty:
            sig_only = tk[tk["sig"] != "ns"].head(20)
            if sig_only.empty:
                panels.append(_section("Tukey HSD post-hoc",
                    html.Div("No significant pairwise differences.",
                             style={"fontSize": "11px",
                                    "color": "var(--text-muted)"})))
            else:
                head = ["Cell A", "Cell B", "Δ√ΔR²", "SE", "p (adj)", ""]
                trows = [html.Tr([
                    html.Th(h, style={"fontSize": "9px",
                                      "color": "var(--text-muted)",
                                      "paddingRight": "12px",
                                      "textAlign": "left",
                                      "paddingBottom": "3px"})
                    for h in head])]
                for _, r in sig_only.iterrows():
                    trows.append(html.Tr([
                        html.Td(r["cell_A"],
                                style={"fontSize": "10px",
                                       "paddingRight": "12px"}),
                        html.Td(r["cell_B"],
                                style={"fontSize": "10px",
                                       "paddingRight": "12px"}),
                        html.Td(f"{r['diff']:+.3f}",
                                style={"fontSize": "10px",
                                       "paddingRight": "12px",
                                       "fontFamily": "monospace"}),
                        html.Td(f"{r['se_diff']:.3f}",
                                style={"fontSize": "10px",
                                       "paddingRight": "12px",
                                       "fontFamily": "monospace"}),
                        html.Td("<0.001" if r["p_adj"] < 0.001
                                else f"{r['p_adj']:.3f}",
                                style={"fontSize": "10px",
                                       "paddingRight": "8px"}),
                        html.Td(_sig_badge(r["sig"])),
                    ]))
                panels.append(_section(
                    f"Tukey HSD — significant comparisons "
                    f"({len(sig_only)}/{len(tk)})",
                    html.Table(trows,
                               style={"borderCollapse": "collapse"})))

    # ── Segmented regression ───────────────────────────────────────────────
    if "seg" in sections:
        sg = result["segmented"]
        if "error" not in sg:
            x   = np.array(sg["x"])
            y   = np.array(sg["y"])
            bp  = sg["breakpoint"]
            pts_colors = [
                DOMAIN_COLORS.get(
                    "Sensory" if r["domain"] == "Sensory-Repetitive"
                    else r["domain"], "#94a3b8")
                for r in sg["ranked"]
            ]
            fig_seg = go.Figure()
            fig_seg.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=11, color=pts_colors,
                            line=dict(color="white", width=1.5)),
                text=[f"{r['domain'][:5]}… {r['band']}"
                      for r in sg["ranked"]],
                hovertemplate="%{text}<br>√ΔR²: %{y:.4f}<extra></extra>",
                showlegend=False,
            ))
            x1 = x[:bp + 1]; x2 = x[bp:]
            fig_seg.add_trace(go.Scatter(
                x=x1, y=np.polyval(sg["b_seg1"], x1),
                mode="lines",
                line=dict(color="#64748b", width=2, dash="solid"),
                name=f"Segment 1  (mean={sg['seg1_mean']:.3f})",
            ))
            fig_seg.add_trace(go.Scatter(
                x=x2, y=np.polyval(sg["b_seg2"], x2),
                mode="lines",
                line=dict(color="#64748b", width=2, dash="dash"),
                name=f"Segment 2  (mean={sg['seg2_mean']:.3f})",
            ))
            fig_seg.add_vline(x=bp - 0.5, line_dash="dot",
                              line_color="#94a3b8", opacity=0.7)
            mode2 = get_plotly_layout("light")
            mode2["legend"] = dict(
                orientation="h", y=-0.22,
                xanchor="center", x=0.5,
                font=dict(size=10, family="Arial"))
            fig_seg.update_layout(
                **mode2, height=300,
                title=dict(
                    text=(f"Segmented regression  ·  "
                          f"F({sg['df1']},{sg['df2']}) = {sg['F']}  ·  "
                          f"p = {'<0.001' if sg['p'] < 0.001 else sg['p']}"),
                    font=dict(size=11, family="Arial", color="#1e293b"),
                    x=0.01, xanchor="left",
                ),
                xaxis=dict(
                    title=dict(text="Rank (highest → lowest √ΔR²)",
                               font=dict(size=10, family="Arial")),
                    tickfont=dict(size=9),
                    gridcolor="rgba(0,0,0,0.06)",
                    linecolor="#cbd5e1",
                ),
                yaxis=dict(
                    title=dict(text="√ΔR²",
                               font=dict(size=10, family="Arial")),
                    tickfont=dict(size=9),
                    gridcolor="rgba(0,0,0,0.06)",
                    linecolor="#cbd5e1",
                    zeroline=False,
                ),
                margin=dict(t=45, b=80, l=55, r=20),
            )
            panels.append(_section("Segmented regression", html.Div([
                dcc.Graph(figure=fig_seg,
                          config={"displayModeBar": False}),
                html.Div(
                    f"Breakpoint after rank {bp} ({sg['bp_label']}).  "
                    f"Segment 1: {', '.join(sg['seg1_labels'])}.  "
                    f"Segment 2: {', '.join(sg['seg2_labels'])}.",
                    style={"fontSize": "10px",
                           "color": "var(--text-muted)",
                           "marginTop": "4px"}),
            ])))

    # ── Bayesian tables ────────────────────────────────────────────────────
    for key, label, sec_key in [
        ("bayes_cell", "Bayesian model — cell level",       "bayes_cell"),
        ("bayes_ind",  "Bayesian model — individual level", "bayes_ind"),
    ]:
        if sec_key not in sections:
            continue
        bm = result[key]
        if "error" in bm:
            panels.append(_section(label, _err(bm["error"])))
            continue
        params = bm.get("params", [])
        n_str  = (f"n = {bm.get('n_obs', bm.get('n_cells', '?'))} "
                  f"{'obs' if 'n_obs' in bm else 'cells'}")
        extra  = (f"{n_str}  ·  ref: {bm['ref_domain']}, {bm['ref_band']}"
                  + (f"  ·  σ = {bm['sigma']}" if "sigma" in bm else ""))
        head = ["Parameter", "Mean", "SE", "95% CI", "P(dir)"]
        trows = [html.Tr([
            html.Th(h, style={"fontSize": "9px",
                              "color": "var(--text-muted)",
                              "paddingRight": "12px",
                              "textAlign": "left",
                              "paddingBottom": "3px"})
            for h in head])]
        for p in params:
            trows.append(html.Tr([
                html.Td(p["parameter"],
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "fontFamily": "monospace"}),
                html.Td(f"{p['mean']:+.4f}",
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "fontFamily": "monospace"}),
                html.Td(f"{p['se']:.4f}",
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "fontFamily": "monospace"}),
                html.Td(f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]",
                        style={"fontSize": "10px", "paddingRight": "12px",
                               "fontFamily": "monospace"}),
                html.Td(f"{p['P_direction']:.3f}",
                        style={"fontSize": "10px"}),
            ]))
        domain_trends = result.get("bayes_domain_trends", {})
        panels.append(_section(label, html.Div([
            # ── Bayesian forest plot ───────────────────────────────────
            _bayes_forest_figure(params, label, domain_trends),
            html.Table(trows,
                       style={"borderCollapse": "collapse",
                              "marginTop": "12px",
                              "marginBottom": "4px"}),
            html.Div(extra,
                     style={"fontSize": "10px",
                            "color": "var(--text-muted)"}),
        ])))

    # ── GLM (Type III SS) ──────────────────────────────────────────────────
    if "glm" in sections:
        from modules.domains import DOMAIN_COLORS as _DC
        gm = result.get("glm", {})
        if "error" in gm:
            panels.append(_section("GLM — Type III SS", _err(gm["error"])))
        elif gm:
            # ── Type III table ────────────────────────────────────────────
            effect_labels = [
                ("domain_score",      "Domain score (overall association)"),
                ("domain_main",       "Domain main effect"),
                ("band_main",         "Band main effect"),
                ("score×domain",      "Score × Domain (slope differs by domain?)"),
                ("score×band",        "Score × Band (slope changes across development?)"),
                ("score×domain×band", "Score × Domain × Band (developmental trajectories differ?)"),
            ]
            t3_rows = [html.Tr([
                html.Th(h, style={"fontSize":"9px","color":"var(--text-muted)",
                                  "paddingRight":"14px","paddingBottom":"3px"})
                for h in ["Effect","F","df1","df2","p",""]])]
            for key, label in effect_labels:
                e = gm.get(key, {})
                if not e:
                    continue
                F   = e.get("F","n/a"); df1 = e.get("df1","n/a")
                df2 = e.get("df2","n/a"); p  = e.get("p","n/a")
                sig = e.get("sig","")
                t3_rows.append(html.Tr([
                    html.Td(label, style={"fontSize":"10px","paddingRight":"14px",
                                          "fontWeight":"500"}),
                    html.Td(f"{F:.3f}" if isinstance(F,float) else str(F),
                            style={"fontSize":"10px","fontFamily":"monospace",
                                   "paddingRight":"10px"}),
                    html.Td(str(df1), style={"fontSize":"10px","paddingRight":"8px",
                                             "color":"var(--text-muted)"}),
                    html.Td(str(df2), style={"fontSize":"10px","paddingRight":"10px",
                                             "color":"var(--text-muted)"}),
                    html.Td("<0.001" if isinstance(p,float) and p<0.001
                            else f"{p:.4f}" if isinstance(p,float) else str(p),
                            style={"fontSize":"10px","paddingRight":"8px",
                                   "fontFamily":"monospace"}),
                    html.Td(_sig_badge(sig)),
                ]))

            # ── Slope figure: domain score × CBCL by domain and band ──────
            coefs = gm.get("coefficients", [])
            coef_map = {c["parameter"]: c for c in coefs}

            from modules.dev_age_analysis import MISSING_CELLS as _MC
            BANDS_ORDER = list(T1_BANDS.keys())
            other_doms  = [d for d in DOMAIN_ORDER if d != DOMAIN_ORDER[0]]
            other_bnds  = [b for b in BANDS_ORDER  if b != BANDS_ORDER[0]]

            fig_glm = go.Figure()
            mode_glm = get_plotly_layout("light")
            mode_glm["legend"] = dict(
                orientation="h", y=-0.22, xanchor="center", x=0.5,
                font=dict(size=11, family="Arial"))

            for dom in DOMAIN_ORDER:
                color = _DC.get(
                    "Sensory" if dom == "Sensory-Repetitive" else dom, "#94a3b8")
                # Only observed bands for this domain
                obs_bands = [b for b in BANDS_ORDER if (dom, b) not in _MC]
                slopes = []
                for band in obs_bands:
                    # base slope for reference domain at reference band
                    base = coef_map.get("domain_score",{}).get("estimate", 0)
                    # add domain offset
                    if dom != DOMAIN_ORDER[0]:
                        base += coef_map.get(f"score×{dom}",{}).get("estimate", 0)
                    # add band offset
                    if band != BANDS_ORDER[0]:
                        base += coef_map.get(f"score×{band}",{}).get("estimate", 0)
                    # add domain×band offset
                    if dom != DOMAIN_ORDER[0] and band != BANDS_ORDER[0]:
                        base += coef_map.get(f"score×{dom}×{band}",{}).get("estimate", 0)
                    slopes.append(round(base, 4))

                # get p-values for significance markers
                sigs = []
                for band in obs_bands:
                    if dom == DOMAIN_ORDER[0] and band == BANDS_ORDER[0]:
                        p_val = coef_map.get("domain_score",{}).get("p", 1)
                    elif dom != DOMAIN_ORDER[0] and band == BANDS_ORDER[0]:
                        p_val = coef_map.get(f"score×{dom}",{}).get("p", 1)
                    elif dom == DOMAIN_ORDER[0] and band != BANDS_ORDER[0]:
                        p_val = coef_map.get(f"score×{band}",{}).get("p", 1)
                    else:
                        p_val = coef_map.get(f"score×{dom}×{band}",{}).get("p", 1)
                    sigs.append("***" if p_val<0.001 else "**" if p_val<0.01
                                 else "*" if p_val<0.05 else "")

                fig_glm.add_trace(go.Scatter(
                    x=obs_bands, y=slopes,
                    mode="lines+markers+text",
                    name=dom,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=9, color=color,
                                line=dict(color="white", width=1.5)),
                    text=[f" {s}" for s in sigs],
                    textposition="top center",
                    textfont=dict(color="white", size=11, family="Arial"),
                    hovertemplate=(
                        f"<b>{dom}</b><br>Band: %{{x}}<br>"
                        "β (slope): %{y:.4f}<extra></extra>"
                    ),
                ))

            fig_glm.add_hline(y=0, line_width=1, line_dash="dot",
                              line_color="#888", opacity=0.5)
            fig_glm.update_layout(
                **mode_glm, height=340,
                title=dict(
                    text="GLM: domain score → CBCL slope by domain and age band",
                    font=dict(size=12, family="Arial", color="#1e293b"),
                    x=0.01, xanchor="left"),
                xaxis=dict(
                    title=dict(text="Age band",
                               font=dict(size=11, family="Arial")),
                    tickfont=dict(size=10, family="Arial"),
                    gridcolor="rgba(0,0,0,0.06)", linecolor="#cbd5e1"),
                yaxis=dict(
                    title=dict(text="β (domain score → CBCL)",
                               font=dict(size=11, family="Arial")),
                    tickfont=dict(size=10, family="Arial"),
                    gridcolor="rgba(0,0,0,0.06)", linecolor="#cbd5e1",
                    zeroline=False),
                margin=dict(t=45, b=90, l=65, r=20),
            )

            # ── Coefficient table ─────────────────────────────────────────
            coef_head = ["Parameter","β","SE","t","p",""]
            coef_trows = [html.Tr([
                html.Th(h, style={"fontSize":"9px","color":"var(--text-muted)",
                                  "paddingRight":"10px","paddingBottom":"3px"})
                for h in coef_head])]
            for c in coefs:
                coef_trows.append(html.Tr([
                    html.Td(c["parameter"],
                            style={"fontSize":"10px","paddingRight":"12px",
                                   "fontFamily":"monospace"}),
                    html.Td(f"{c['estimate']:+.4f}",
                            style={"fontSize":"10px","paddingRight":"10px",
                                   "fontFamily":"monospace"}),
                    html.Td(f"{c['se']:.4f}",
                            style={"fontSize":"10px","paddingRight":"10px",
                                   "fontFamily":"monospace"}),
                    html.Td(f"{c['t']:.3f}",
                            style={"fontSize":"10px","paddingRight":"10px",
                                   "fontFamily":"monospace"}),
                    html.Td("<0.001" if c["p"]<0.001 else f"{c['p']:.4f}",
                            style={"fontSize":"10px","paddingRight":"8px",
                                   "fontFamily":"monospace"}),
                    html.Td(_sig_badge(c["sig"])),
                ]))

            panels.append(_section(
                f"GLM — Type III SS  ·  {gm.get('model','')}",
                html.Div([
                    html.Table(t3_rows,
                               style={"borderCollapse":"collapse",
                                      "marginBottom":"12px"}),
                    html.Div(
                        f"n = {gm.get('n_obs','?'):,} obs  ·  "
                        f"df residual = {gm.get('df_resid','?'):,}  ·  "
                        f"Ref: {gm.get('ref_domain','?')}, {gm.get('ref_band','?')}",
                        style={"fontSize":"10px","color":"var(--text-muted)",
                               "marginBottom":"12px"}),
                    dcc.Graph(figure=fig_glm,
                              config={"displayModeBar": False},
                              style={"marginBottom":"12px"}),
                    html.Div("Slopes (β) represent the change in CBCL per 1 SD "
                             "increase in domain score, estimated from all individual "
                             "observations in each domain × band cell.",
                             style={"fontSize":"9px","color":"var(--text-muted)",
                                    "marginBottom":"10px"}),
                    html.Table(coef_trows,
                               style={"borderCollapse":"collapse"}),
                ])))

    # ── GLM Logistic ──────────────────────────────────────────────────────
    if "glm_logistic" in sections:
        gl = result.get("glm_logistic", {})
        if "error" in gl:
            panels.append(_section("GLM Logistic", _err(gl["error"])))
        elif gl:
            from modules.domains import DOMAIN_COLORS as _DC2
            # ── Header info ───────────────────────────────────────────────
            panels.append(_section(
                f"GLM Logistic  ·  {gl.get('model','')}",
                html.Div([
                    html.Div(gl.get("threshold_note",""),
                             style={"fontSize":"10px",
                                    "color":"var(--text-muted)",
                                    "marginBottom":"10px"}),

                    # ── OR forest plot ────────────────────────────────────
                    _logistic_or_figure(
                        gl.get("coefficients",[]),
                        gl.get("predictions",[])),

                    # ── Predicted probabilities heatmap-style table ───────
                    html.Div("Predicted probability of elevated late CBCL "
                             "by domain, age band, and domain score (z)",
                             style={"fontWeight":"600","fontSize":"11px",
                                    "marginBottom":"6px"}),
                    _logistic_pred_table(
                        gl.get("predictions",[]), _DC2),

                    # ── Coefficients table ────────────────────────────────
                    html.Div("Logistic regression coefficients (log OR)",
                             style={"fontWeight":"600","fontSize":"11px",
                                    "marginTop":"14px",
                                    "marginBottom":"6px"}),
                    _logistic_coef_table(gl.get("coefficients",[])),

                    html.Div(
                        f"n = {gl.get('n_obs',0):,} obs  ·  "
                        f"Elevated: {gl.get('n_elevated',0):,} "
                        f"({gl.get('prevalence_pct',0):.1f}%)  ·  "
                        f"Ref: {gl.get('ref_domain','Sensory-Repetitive')}, "
                        f"0–4y",
                        style={"fontSize":"9px",
                               "color":"var(--text-muted)",
                               "marginTop":"8px"}),
                ])))


    # ── Mixed Models ──────────────────────────────────────────────────────
    if "mixed" in sections:
        mx = result.get("mixed", {})
        if "error" in mx:
            panels.append(_section("Mixed Models", _err(mx["error"])))
        elif mx:
            glm_r   = mx.get("glm", {})
            mixed_r = mx.get("mixed", {})

            # Show mixed model error prominently if it failed
            mixed_err_div = html.Div()
            if isinstance(mixed_r, dict) and "error" in mixed_r:
                mixed_err_div = dbc.Alert(
                    [html.Strong("Mixed model error: "),
                     mixed_r["error"]],
                    color="warning",
                    style={"fontSize":"11px","marginBottom":"10px"})
                mixed_r = {}  # skip broken model block

            def _model_block(m, label):
                if not m:
                    return html.Div()
                coefs = m.get("coefficients", [])

                # Stats header
                icc_str = (f"  ·  ICC = {m['icc']:.3f}"
                           if m.get("icc") is not None else "")
                su_str  = (f"  ·  σ²_u = {m['sigma2_u']:.4f}"
                           if m.get("sigma2_u") is not None else "")
                deff_str= (f"  ·  DEFF = {m['deff']:.3f}  ·  n̄ = {m['n_bar']:.1f}"
                           if m.get("deff") is not None else "")
                header  = html.Div(
                    (f"AIC = {m.get('aic','?')}  ·  "
                     f"σ²_e = {m.get('sigma2_e','?')}"
                     f"{su_str}{icc_str}{deff_str}"),
                    style={"fontSize":"10px",
                           "color":"var(--text-muted)",
                           "marginBottom":"6px"})

                # Coefficient table
                head = ["Parameter","β","SE","t","p",""]
                trows = [html.Tr([
                    html.Th(h, style={"fontSize":"9px",
                                      "color":"var(--text-muted)",
                                      "paddingRight":"10px",
                                      "paddingBottom":"3px"})
                    for h in head])]
                for c in coefs:
                    trows.append(html.Tr([
                        html.Td(c["parameter"],
                                style={"fontSize":"10px",
                                       "paddingRight":"12px",
                                       "fontFamily":"monospace"}),
                        html.Td(f"{c['estimate']:+.4f}",
                                style={"fontSize":"10px",
                                       "paddingRight":"10px",
                                       "fontFamily":"monospace"}),
                        html.Td(f"{c['se']:.4f}" if c.get('se') is not None else "—",
                                style={"fontSize":"10px",
                                       "paddingRight":"10px",
                                       "fontFamily":"monospace"}),
                        html.Td(f"{c['t']:.3f}" if c.get('t') is not None else "—",
                                style={"fontSize":"10px",
                                       "paddingRight":"10px",
                                       "fontFamily":"monospace"}),
                        html.Td(("<0.001" if c["p"]<0.001
                                else f"{c['p']:.4f}") if c.get('p') is not None else "—",
                                style={"fontSize":"10px",
                                       "paddingRight":"8px",
                                       "fontFamily":"monospace"}),
                        html.Td(_sig_badge(c["sig"]) if c.get("sig") else html.Span()),
                    ]))

                # Slopes table
                slopes = m.get("slopes", [])
                slope_head = ["Domain","Band","β (1 SD)"]
                strows = [html.Tr([
                    html.Th(h, style={"fontSize":"9px",
                                      "color":"var(--text-muted)",
                                      "paddingRight":"10px",
                                      "paddingBottom":"3px"})
                    for h in slope_head])]
                from modules.domains import DOMAIN_COLORS as _DC3
                prev_dom = None
                for s in slopes:
                    c = _DC3.get(
                        "Sensory" if s["domain"]=="Sensory-Repetitive"
                        else s["domain"], "#94a3b8")
                    dom_td = html.Td(
                        html.Span(s["domain"],
                                  style={"color":c,"fontWeight":"700",
                                         "fontSize":"10px"})
                        if s["domain"] != prev_dom else "",
                        style={"paddingRight":"10px"})
                    prev_dom = s["domain"]
                    strows.append(html.Tr([
                        dom_td,
                        html.Td(s["band"],
                                style={"fontSize":"10px",
                                       "paddingRight":"10px",
                                       "color":"var(--text-muted)"}),
                        html.Td(f"{s['slope_per_1SD']:+.4f}",
                                style={"fontSize":"10px",
                                       "fontFamily":"monospace"}),
                    ]))

                return html.Div([
                    html.Div(label,
                             style={"fontWeight":"600",
                                    "fontSize":"11px",
                                    "marginBottom":"4px"}),
                    header,
                    html.Div("Coefficients:", style={"fontSize":"10px",
                                                      "marginBottom":"3px",
                                                      "color":"var(--text-muted)"}),
                    html.Table(trows,
                               style={"borderCollapse":"collapse",
                                      "marginBottom":"10px"}),
                    html.Div("Slopes (CBCL change per 1 SD domain score):",
                             style={"fontSize":"10px","marginBottom":"3px",
                                    "color":"var(--text-muted)"}),
                    html.Table(strows,
                               style={"borderCollapse":"collapse"}),
                ], style={"marginBottom":"16px"})

            panels.append(_section(
                f"Mixed Models  ·  "
                f"n = {mx.get('n_obs',0):,} obs, "
                f"{mx.get('n_people',0):,} people",
                html.Div([
                    html.Div(
                        f"Model: {mx.get('model','')}",
                        style={"fontSize":"10px",
                               "color":"var(--text-muted)",
                               "marginBottom":"12px",
                               "fontFamily":"monospace"}),
                    _model_block(glm_r,   "Model 1 — Population GLM (no grouping)"),
                    html.Hr(style={"borderColor":"var(--border)",
                                   "margin":"8px 0"}),
                    mixed_err_div,
                    _model_block(mixed_r, "Model 2 — Mixed-effects (random intercept per person)"),

                    html.Hr(style={"borderColor":"var(--border)", "margin":"8px 0"}),

                    # ── Slope figures ────────────────────────────────────
                    html.Div("Slopes: CBCL change per 1 SD domain score",
                             style={"fontWeight":"600","fontSize":"11px",
                                    "marginBottom":"8px","marginTop":"4px"}),
                    _mixed_slope_figures(glm_r, mixed_r),
                ])))


    # ── Footer note ────────────────────────────────────────────────────────
    panels.append(html.Div(
        "* q < .05  ** q < .01  *** q < .001  (FDR, BH). "
        "ANOVA weighted by 1/SE².  Tukey with Bonferroni correction. "
        "Bayesian posteriors: normal–normal conjugate, analytical. "
        "Covariates: T1 age (domain eval), sex.",
        style={"fontSize": "9px", "color": "var(--text-muted)",
               "marginTop": "10px", "fontFamily": "Arial"}))

    return html.Div(panels)




def _mixed_slope_figures(glm_r, mixed_r):
    """
    GLM and Mixed in SEPARATE panels.
    All domains on ONE graph per panel. Motor x 0-4y excluded.
    Fig 1: Coefficient forest — GLM left | Mixed right
    Fig 2: All domains slopes — GLM left | Mixed right
    """
    from modules.dev_age_analysis import DOMAIN_ORDER, T1_BANDS, MISSING_CELLS
    from modules.domains import DOMAIN_COLORS as _DC
    from plotly.subplots import make_subplots as _msp

    if not glm_r:
        return html.Div("No GLM data.", style={"fontSize":"10px","color":"var(--text-muted)"})

    BANDS     = list(T1_BANDS.keys())
    DOMAINS   = DOMAIN_ORDER
    has_mixed = bool(mixed_r and mixed_r.get("slopes"))
    glm_sm    = {(r["domain"],r["band"]): r for r in glm_r.get("slopes",[])}
    mixed_sm  = {(r["domain"],r["band"]): r for r in mixed_r.get("slopes",[])} if has_mixed else {}

    def _color(dom):
        return _DC.get("Sensory" if dom=="Sensory-Repetitive" else dom, "#94a3b8")

    def _obs_bands(dom, smap):
        return [b for b in BANDS if (dom,b) not in MISSING_CELLS and (dom,b) in smap]

    # ── Fig 1: Coefficient forest — GLM | Mixed ───────────────────────────
    glm_coefs   = {c["parameter"]: c for c in glm_r.get("coefficients",[])}
    mixed_coefs = {c["parameter"]: c for c in mixed_r.get("coefficients",[])} if has_mixed else {}
    skip   = {"intercept","z_age","sex"}
    params = [c["parameter"] for c in glm_r.get("coefficients",[])
              if c["parameter"] not in skip]

    n_cols1 = 2 if has_mixed else 1
    titles1 = ["Population GLM"] + (["Mixed-effects (DEFF-corrected)"] if has_mixed else [])
    fig1 = _msp(rows=1, cols=n_cols1, subplot_titles=titles1, shared_yaxes=True)

    for col_i, (cmap, filled) in enumerate(
            [(glm_coefs, True)] + ([(mixed_coefs, False)] if has_mixed else []), 1):
        ests   = [cmap.get(p,{}).get("estimate") for p in params]
        ses    = [cmap.get(p,{}).get("se",0) or 0 for p in params]
        ci_los = [(cmap.get(p,{}).get("ci_lo") or ((e or 0)-1.96*s))
                  for p,e,s in zip(params,ests,ses)]
        ci_his = [(cmap.get(p,{}).get("ci_hi") or ((e or 0)+1.96*s))
                  for p,e,s in zip(params,ests,ses)]
        colors = ["#38bdf8" if "Motor" in p
                  else "#34d399" if "Social" in p
                  else "#94a3b8" for p in params]
        # Separate trace per point for correct colors
        for pi, (p, e, lo, hi, c) in enumerate(zip(params, ests, ci_los, ci_his, colors)):
            if e is None: continue
            fig1.add_trace(go.Scatter(
                x=[e], y=[p], mode="markers",
                marker=dict(size=9,
                            color=c if filled else "white",
                            line=dict(color=c, width=2)),
                error_x=dict(type="data", symmetric=False,
                             array=[hi-e], arrayminus=[e-lo],
                             visible=True, thickness=1.8,
                             width=4, color=c),
                showlegend=False,
                hovertemplate=f"{p}<br>β=%{{x:.4f}}<extra></extra>",
            ), row=1, col=col_i)
        fig1.add_vline(x=0, line_width=1, line_dash="dot",
                       line_color="#888", opacity=0.5, row=1, col=col_i)

    mode1 = get_plotly_layout("light")
    fig1.update_layout(**mode1, showlegend=False,
        height=max(320, len(params)*22+100),
        title=dict(text="Coefficient estimates ± 95% CI",
                   font=dict(size=11,family="Arial",color="#1e293b"),
                   x=0.01, xanchor="left"),
        margin=dict(t=50,b=40,l=200,r=20))
    for ci in range(1, n_cols1+1):
        fig1.update_xaxes(title_text="β (95% CI)", tickfont=dict(size=8),
                          gridcolor="rgba(0,0,0,0.06)", zeroline=False, row=1, col=ci)
        fig1.update_yaxes(tickfont=dict(size=8), autorange="reversed", row=1, col=ci)

    # ── Fig 2: All domains — GLM left | Mixed right ───────────────────────
    n_cols2 = 2 if has_mixed else 1
    titles2  = ["Population GLM"] + (["Mixed-effects (DEFF-corrected)"] if has_mixed else [])
    fig2 = _msp(rows=1, cols=n_cols2, subplot_titles=titles2, shared_yaxes=True)

    for col_i, smap in enumerate([glm_sm] + ([mixed_sm] if has_mixed else []), 1):
        for dom in DOMAINS:
            color  = _color(dom)
            obs_b  = _obs_bands(dom, smap)
            if not obs_b: continue
            ys = [smap[(dom,b)]["slope_per_1SD"] for b in obs_b]
            lo = [smap[(dom,b)]["ci_lo"]         for b in obs_b]
            hi = [smap[(dom,b)]["ci_hi"]         for b in obs_b]
            fig2.add_trace(go.Scatter(
                x=obs_b, y=ys,
                mode="lines+markers",
                name=dom, legendgroup=dom, showlegend=(col_i==1),
                line=dict(color=color, width=2.5),
                marker=dict(size=9, color=color, line=dict(color="white",width=1.5)),
                error_y=dict(type="data", symmetric=False,
                             array=[h-y for y,h in zip(ys,hi)],
                             arrayminus=[y-l for y,l in zip(ys,lo)],
                             visible=True, color=color, thickness=1.2, width=4),
                hovertemplate=f"<b>{dom}</b><br>%{{x}}<br>β=%{{y:.4f}}<extra></extra>",
            ), row=1, col=col_i)
            fig2.add_hline(y=0, line_width=1, line_dash="dot",
                           line_color="#888", opacity=0.4, row=1, col=col_i)

    mode2 = get_plotly_layout("light")
    mode2["legend"] = dict(orientation="h", y=-0.20, xanchor="center", x=0.5,
                           font=dict(size=11, family="Arial"))
    fig2.update_layout(**mode2, height=350,
        title=dict(text="Slopes ± 95% CI  ·  all domains  ·  GLM vs Mixed (separate panels)",
                   font=dict(size=11,family="Arial",color="#1e293b"),
                   x=0.01, xanchor="left"),
        margin=dict(t=50,b=90,l=70,r=20))
    for ci in range(1, n_cols2+1):
        fig2.update_xaxes(title_text="Age band", tickfont=dict(size=9), row=1, col=ci)
        fig2.update_yaxes(title_text="β (1 SD → CBCL)" if ci==1 else "",
                          tickfont=dict(size=9),
                          gridcolor="rgba(0,0,0,0.06)", zeroline=False, row=1, col=ci)

    return html.Div([
        dcc.Graph(figure=fig1, config={"displayModeBar":False},
                  style={"marginBottom":"14px"}),
        dcc.Graph(figure=fig2, config={"displayModeBar":False},
                  style={"marginBottom":"4px"}),
        html.Div(
            "Motor × 0–4y excluded.  GLM = Population OLS.  "
            "Mixed = DEFF-corrected SEs.  Error bars = 95% CI.",
            style={"fontSize":"9px","color":"var(--text-muted)"}),
    ])


def _sanitize_for_excel(df):
    """Strip all characters that openpyxl cannot write to Excel cells."""
    def _clean(val):
        if not isinstance(val, str):
            return val
        val = val.replace("\u00d7", "x").replace("\u2013", "-").replace("\u2014", "-")
        val = val.replace("\u2019", "'").replace("\u2018", "'")
        out = []
        for ch in val:
            cp = ord(ch)
            if cp in (0x09, 0x0a, 0x0d): out.append(ch)
            elif 0x20 <= cp <= 0x7e: out.append(ch)
            elif 0x00a0 <= cp <= 0xfffd: out.append(ch)
        return "".join(out)
    try:
        return df.applymap(_clean)
    except AttributeError:
        return df.map(_clean)


def _bayes_forest_figure(params, label, domain_trends=None):
    """
    Bayesian slope trajectory plot:
    x = age band, y = posterior mean sqrt(DeltaR2), error bars = 95% CrI.
    One line per domain. Motor starts at 4-8y.
    Data from within-domain models (Option A, run_bayes_domain_trends).
    Falls back to cross-domain cell-level model if domain_trends unavailable.
    """
    from modules.dev_age_analysis import DOMAIN_ORDER, T1_BANDS, MISSING_CELLS
    from modules.domains import DOMAIN_COLORS as _DC

    BANDS   = list(T1_BANDS.keys())
    DOMAINS = DOMAIN_ORDER

    def _color(dom):
        return _DC.get("Sensory" if dom=="Sensory-Repetitive" else dom, "#94a3b8")

    # ── Build per-cell posterior means from within-domain models ──────────
    cell_posts = {}   # (dom, band) -> {mean, ci_lo, ci_hi}

    if domain_trends and "by_domain" in domain_trends:
        for dom in DOMAINS:
            res = domain_trends["by_domain"].get(dom, {})
            if "error" in res or not res.get("params"):
                continue
            ref_band  = res["ref_band"]
            intercept = None
            offsets   = {}

            for p in res["params"]:
                if "intercept" in p["parameter"]:
                    intercept = p
                else:
                    # band offset vs ref_band
                    band_lbl = p.get("band", "")
                    offsets[band_lbl] = p

            if intercept is None:
                continue

            # Reference band cell
            cell_posts[(dom, ref_band)] = {
                "mean":  intercept["mean"],
                "ci_lo": intercept["ci_lo"],
                "ci_hi": intercept["ci_hi"],
            }

            # Other bands: intercept + offset, propagate uncertainty
            for band_lbl, off in offsets.items():
                # Find matching band in T1_BANDS
                matched = None
                for b in BANDS:
                    if band_lbl in b or b in band_lbl:
                        matched = b
                        break
                if matched is None:
                    # Try exact match from parameter name
                    for b in BANDS:
                        clean = b.replace("–","-").replace("—","-")
                        if clean in off["parameter"] or b in off["parameter"]:
                            matched = b
                            break
                if matched is None:
                    matched = band_lbl

                mean_cell  = intercept["mean"]  + off["mean"]
                # SE propagation: sqrt(se_int^2 + se_off^2)
                se_int = (intercept["ci_hi"] - intercept["ci_lo"]) / (2*1.96)
                se_off = (off["ci_hi"] - off["ci_lo"]) / (2*1.96)
                se_cell = (se_int**2 + se_off**2)**0.5
                cell_posts[(dom, matched)] = {
                    "mean":  mean_cell,
                    "ci_lo": mean_cell - 1.96*se_cell,
                    "ci_hi": mean_cell + 1.96*se_cell,
                }

    # Fallback: use cross-domain cell-level params to reconstruct cell means
    if not cell_posts and params:
        p_map = {p["parameter"]: p for p in params}
        intercept_mean = p_map.get("intercept", {}).get("mean", 0)
        intercept_se   = (p_map.get("intercept",{}).get("ci_hi",0) -
                          p_map.get("intercept",{}).get("ci_lo",0)) / (2*1.96)
        ref_band = BANDS[0]
        for dom in DOMAINS:
            for band in BANDS:
                if (dom, band) in MISSING_CELLS:
                    continue
                mean_cell = intercept_mean
                se_cell   = intercept_se
                if dom != DOMAINS[0]:
                    dk = f"domain_{dom}"
                    dp = p_map.get(dk, {})
                    mean_cell += dp.get("mean", 0)
                    dse = (dp.get("ci_hi",0)-dp.get("ci_lo",0))/(2*1.96)
                    se_cell = (se_cell**2 + dse**2)**0.5
                if band != ref_band:
                    bk = f"band_{band}"
                    bp = p_map.get(bk, {})
                    mean_cell += bp.get("mean", 0)
                    bse = (bp.get("ci_hi",0)-bp.get("ci_lo",0))/(2*1.96)
                    se_cell = (se_cell**2 + bse**2)**0.5
                cell_posts[(dom, band)] = {
                    "mean":  mean_cell,
                    "ci_lo": mean_cell - 1.96*se_cell,
                    "ci_hi": mean_cell + 1.96*se_cell,
                }

    if not cell_posts:
        return html.Div("No Bayesian cell data available.",
                        style={"fontSize":"10px","color":"var(--text-muted)"})

    # ── Build figure ──────────────────────────────────────────────────────
    fig = go.Figure()
    mode = get_plotly_layout("light")

    for dom in DOMAINS:
        color   = _color(dom)
        obs_b   = [b for b in BANDS
                   if (dom,b) not in MISSING_CELLS and (dom,b) in cell_posts]
        if not obs_b:
            continue
        means  = [cell_posts[(dom,b)]["mean"]  for b in obs_b]
        ci_los = [cell_posts[(dom,b)]["ci_lo"] for b in obs_b]
        ci_his = [cell_posts[(dom,b)]["ci_hi"] for b in obs_b]

        fig.add_trace(go.Scatter(
            x=obs_b, y=means,
            mode="lines+markers",
            name=dom,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color,
                        line=dict(color="white", width=1.5)),
            error_y=dict(
                type="data", symmetric=False,
                array=[h-m for m,h in zip(means,ci_his)],
                arrayminus=[m-l for m,l in zip(means,ci_los)],
                visible=True,
                color=color,
                thickness=1.5, width=5,
            ),
            hovertemplate=(
                f"<b>{dom}</b><br>"
                "Band: %{x}<br>"
                "Posterior mean: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_width=1, line_dash="dot",
                  line_color="#888", opacity=0.5)

    mode["legend"] = dict(
        orientation="h", y=-0.22, xanchor="center", x=0.5,
        font=dict(size=11, family="Arial"))
    fig.update_layout(
        **mode,
        height=320,
        title=dict(
            text="√ΔR² posterior means ± 95% CrI  ·  "
                 "within-domain Bayesian models",
            font=dict(size=11, family="Arial", color="#1e293b"),
            x=0.01, xanchor="left"),
        xaxis=dict(
            title=dict(text="Age band",
                       font=dict(size=10, family="Arial")),
            tickfont=dict(size=9, family="Arial"),
            gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(
            title=dict(text="√ΔR² (posterior mean)",
                       font=dict(size=10, family="Arial")),
            tickfont=dict(size=9, family="Arial"),
            gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        margin=dict(t=45, b=80, l=70, r=20),
    )

    src_note = ("Within-domain Bayesian models (separate fit per domain)."
                if domain_trends and "by_domain" in domain_trends
                else "Cross-domain Bayesian model (single fit).")
    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"marginBottom": "4px"}),
        html.Div(
            f"{src_note}  "
            "Point = posterior mean √ΔR².  "
            "Error bars = 95% credible interval.  "
            "Motor × 0–4y excluded.  "
            "Motor reference band = 4–8y.",
            style={"fontSize": "9px", "color": "var(--text-muted)"}),
    ])


def _logistic_pred_table(pred_rows, DC):
    """Predicted probability table: domains x bands x z-scores."""
    from modules.dev_age_analysis import DOMAIN_ORDER, T1_BANDS, MISSING_CELLS
    if not pred_rows:
        return html.Div("No predictions available.",
                        style={"fontSize":"10px","color":"var(--text-muted)"})
    pred_df = pd.DataFrame(pred_rows)
    z_vals  = sorted(pred_df["z_score"].unique())
    DOMAINS = DOMAIN_ORDER
    head = ["Domain","Band"] + [f"z={z:+.0f}" for z in z_vals]
    trows = [html.Tr([html.Th(h, style={"fontSize":"9px","color":"var(--text-muted)",
             "paddingRight":"10px","paddingBottom":"3px","textAlign":"left"})
             for h in head])]
    for dom in DOMAINS:
        color = DC.get("Sensory" if dom=="Sensory-Repetitive" else dom, "#94a3b8")
        prev_dom = None
        dom_rows = pred_df[pred_df["domain"]==dom]
        for band in dom_rows["band"].unique():
            if (dom, band) in MISSING_CELLS:
                continue
            sub = dom_rows[dom_rows["band"]==band]
            dom_cell = html.Td(
                html.Span(dom, style={"color":color,"fontWeight":"700","fontSize":"10px"})
                if dom != prev_dom else "",
                style={"paddingRight":"10px"})
            prev_dom = dom
            prob_cells = []
            for z in z_vals:
                row = sub[sub["z_score"]==z]
                if row.empty:
                    prob_cells.append(html.Td("—"))
                else:
                    pct = float(row["pct_elevated"].iloc[0])
                    alpha = min(pct/50.0, 1.0)
                    bg = (f"rgba(251,146,60,{alpha*0.35:.2f})" if dom=="Sensory-Repetitive"
                          else f"rgba(56,189,248,{alpha*0.35:.2f})" if dom=="Motor"
                          else f"rgba(52,211,153,{alpha*0.35:.2f})")
                    prob_cells.append(html.Td(f"{pct:.1f}%",
                        style={"fontSize":"10px","fontFamily":"monospace",
                               "paddingRight":"10px","background":bg,
                               "borderRadius":"3px","textAlign":"right"}))
            trows.append(html.Tr([dom_cell,
                html.Td(band, style={"fontSize":"10px","paddingRight":"10px",
                                     "color":"var(--text-muted)"}),
                *prob_cells]))
    return html.Table(trows, style={"borderCollapse":"collapse","marginBottom":"4px"})


def _logistic_coef_table(coefs):
    if not coefs:
        return html.Span("")
    head = ["Parameter","log OR","OR","SE","z","p",""]
    trows = [html.Tr([html.Th(h, style={"fontSize":"9px","color":"var(--text-muted)",
             "paddingRight":"10px","paddingBottom":"3px","textAlign":"left"})
             for h in head])]
    for c in coefs:
        trows.append(html.Tr([
            html.Td(c["parameter"], style={"fontSize":"10px","paddingRight":"12px",
                                           "fontFamily":"monospace"}),
            html.Td(f"{c['log_OR']:+.4f}", style={"fontSize":"10px","paddingRight":"10px",
                                                    "fontFamily":"monospace"}),
            html.Td(f"{c['OR']:.3f}", style={"fontSize":"10px","paddingRight":"10px",
                                              "fontFamily":"monospace"}),
            html.Td(f"{c['se']:.4f}", style={"fontSize":"10px","paddingRight":"10px",
                                              "fontFamily":"monospace"}),
            html.Td(f"{c['z']:.3f}", style={"fontSize":"10px","paddingRight":"10px",
                                             "fontFamily":"monospace"}),
            html.Td("<0.001" if c["p"]<0.001 else f"{c['p']:.4f}",
                    style={"fontSize":"10px","paddingRight":"8px","fontFamily":"monospace"}),
            html.Td(_sig_badge(c["sig"])),
        ]))
    return html.Table(trows, style={"borderCollapse":"collapse"})


def _logistic_or_figure(coefs, pred_rows):
    """
    Lollipop OR forest plot for logistic GLM.
    11 observed cells, stem from OR=1 to point, colored by domain.
    Layout exactly as:
        SR · 0-4y    ●─────  [2.1, 3.0]
        SR · 4-8y    ●────   [1.9, 2.7]
        ...
        Motor · 4-8y ●───    [1.4, 1.9]   (starts at 4-8y)
        ...
        Social · 0-4y●─      [1.1, 1.5]
        ...
    """
    from modules.dev_age_analysis import DOMAIN_ORDER, T1_BANDS, MISSING_CELLS
    from modules.domains import DOMAIN_COLORS as _DC
    import math

    if not coefs:
        return html.Div()

    coef_map = {c["parameter"]: c for c in coefs}
    BANDS    = list(T1_BANDS.keys())
    ref_dom  = DOMAIN_ORDER[0]
    ref_band = BANDS[0]

    # Build 11-cell OR table
    rows = []
    for dom in DOMAIN_ORDER:
        color = _DC.get(
            "Sensory" if dom=="Sensory-Repetitive" else dom, "#94a3b8")
        short = "SR" if dom=="Sensory-Repetitive" else dom
        for band in BANDS:
            if (dom, band) in MISSING_CELLS:
                continue

            log_or   = coef_map.get("domain_score",{}).get("log_OR",0) or 0
            se_logOR = coef_map.get("domain_score",{}).get("se",0) or 0

            if dom != ref_dom:
                k = f"score×{dom}"
                log_or   += coef_map.get(k,{}).get("log_OR",0) or 0
                se_logOR  = math.sqrt(se_logOR**2 +
                            (coef_map.get(k,{}).get("se",0) or 0)**2)
            if band != ref_band:
                k = f"score×{band}"
                log_or   += coef_map.get(k,{}).get("log_OR",0) or 0
                se_logOR  = math.sqrt(se_logOR**2 +
                            (coef_map.get(k,{}).get("se",0) or 0)**2)
            if dom != ref_dom and band != ref_band:
                k = f"score×{dom}×{band}"
                log_or   += coef_map.get(k,{}).get("log_OR",0) or 0
                se_logOR  = math.sqrt(se_logOR**2 +
                            (coef_map.get(k,{}).get("se",0) or 0)**2)

            OR    = math.exp(log_or)
            OR_lo = math.exp(log_or - 1.96*se_logOR)
            OR_hi = math.exp(log_or + 1.96*se_logOR)

            rows.append({
                "label":  f"{short} · {band}",
                "dom":    dom,
                "band":   band,
                "OR":     OR,
                "OR_lo":  OR_lo,
                "OR_hi":  OR_hi,
                "color":  color,
            })

    if not rows:
        return html.Div()

    # Build SVG-style lollipop chart using Plotly shapes + scatter
    labels = [r["label"]  for r in rows]
    ORs    = [r["OR"]     for r in rows]
    OR_los = [r["OR_lo"]  for r in rows]
    OR_his = [r["OR_hi"]  for r in rows]
    colors = [r["color"]  for r in rows]

    fig = go.Figure()

    # ── Stems from OR=1 to the point ─────────────────────────────────────
    for i, r in enumerate(rows):
        fig.add_shape(
            type="line",
            x0=1.0, x1=r["OR"],
            y0=r["label"], y1=r["label"],
            line=dict(color=r["color"], width=2.5),
            layer="below",
        )

    # ── CI bars ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ORs, y=labels,
        mode="markers",
        marker=dict(
            size=12,
            color=colors,
            line=dict(color="white", width=1.8),
            symbol="circle",
        ),
        error_x=dict(
            type="data", symmetric=False,
            array=[hi-OR for OR,hi in zip(ORs,OR_his)],
            arrayminus=[OR-lo for OR,lo in zip(ORs,OR_los)],
            visible=True,
            thickness=1.8, width=5,
        ),
        customdata=[[r["OR_lo"], r["OR_hi"]] for r in rows],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "OR = %{x:.3f}<br>"
            "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # ── Vertical line at OR = 1 ───────────────────────────────────────────
    fig.add_vline(x=1.0, line_width=1.5, line_dash="dash",
                  line_color="#475569", opacity=0.7)

    # ── Domain separator horizontal lines ─────────────────────────────────
    dom_counts = {}
    for r in rows:
        dom_counts[r["dom"]] = dom_counts.get(r["dom"],0) + 1
    cumulative = 0
    for dom in DOMAIN_ORDER:
        if dom not in dom_counts:
            continue
        cumulative += dom_counts[dom]
        if cumulative < len(rows):
            # Between-domain separator
            sep_label_idx = cumulative - 1
            next_label_idx = cumulative
            sep_y = (labels[sep_label_idx] + "~" + labels[next_label_idx]
                     if False else labels[sep_label_idx])
            fig.add_shape(
                type="line",
                x0=0, x1=1, xref="paper",
                y0=cumulative - 0.5, y1=cumulative - 0.5,
                yref="y",
                line=dict(color="#e2e8f0", width=1.5, dash="solid"),
            )

    # ── Domain color annotations on right ─────────────────────────────────
    prev_dom = None
    x_ann = max(OR_his) * 1.08
    for r in rows:
        if r["dom"] != prev_dom:
            fig.add_annotation(
                x=x_ann, y=r["label"],
                text=f"<b>{r['dom'].replace('Sensory-Repetitive','Sensory-Rep')}</b>",
                showarrow=False,
                font=dict(size=9.5, color=r["color"], family="Arial"),
                xanchor="left", yanchor="middle",
            )
            prev_dom = r["dom"]

    mode = get_plotly_layout("light")
    fig.update_layout(
        **mode,
        height=max(350, len(rows)*48 + 80),
        showlegend=False,
        xaxis=dict(
            title=dict(
                text="Odds Ratio per 1 SD domain score (95% CI)",
                font=dict(size=10, family="Arial")),
            type="log",
            tickfont=dict(size=9, family="Arial"),
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            tickvals=[0.5, 1, 1.5, 2, 2.5, 3, 4],
            ticktext=["0.5","1.0","1.5","2.0","2.5","3.0","4.0"],
        ),
        yaxis=dict(
            tickfont=dict(size=10, family="Arial"),
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=30, b=55, l=140, r=180),
    )

    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"marginBottom": "4px"}),
        html.Div(
            "OR = odds of elevated CBCL per 1 SD increase in domain score.  "
            "Lollipop stem from OR = 1.0 (null) to estimate.  "
            "Log scale.  Motor × 0–4y excluded.  "
            "Ref: Sensory-Repetitive, 0–4y.",
            style={"fontSize": "9px", "color": "var(--text-muted)"}),
    ])


def _section(title, content):
    return html.Div([
        html.Div(title,
                 style={"fontWeight": "700", "fontSize": "12px",
                        "marginBottom": "6px", "marginTop": "14px",
                        "color": "#1e293b", "fontFamily": "Arial",
                        "borderBottom": "1px solid var(--border)",
                        "paddingBottom": "3px"}),
        content,
    ])