"""
helpers/theme.py
─────────────────────────────────────────────────────────────────────────────
Theme tokens and Plotly layout integration.

CSS variables in assets/theme.css define the actual colors per theme.
This module provides:
- The Plotly layout overrides for charts to match the active theme
- Color tokens accessible from Python for non-CSS contexts

Single source of truth: change a color here AND in theme.css to update.
"""

# Color tokens - keep in sync with assets/theme.css
DARK = {
    "bg":         "#0d0f14",
    "bg_alt":     "#13161e",
    "text":       "#e2e8f0",
    "text_muted": "#94a3b8",
    "border":     "#1f2433",
    "accent":     "#38bdf8",
    "success":    "#34d399",
    "warning":    "#fbbf24",
    "danger":     "#f87171",
    "grid":       "#1f2433",
}

LIGHT = {
    "bg":         "#f8fafc",
    "bg_alt":     "#f1f5f9",
    "text":       "#1e293b",
    "text_muted": "#64748b",
    "border":     "#e2e8f0",
    "accent":     "#0284c7",
    "success":    "#16a34a",
    "warning":    "#d97706",
    "danger":     "#dc2626",
    "grid":       "#e2e8f0",
}


def tokens(mode: str = "dark") -> dict:
    """Return color tokens for the active theme mode."""
    return LIGHT if mode == "light" else DARK


def get_plotly_layout(mode: str = "dark") -> dict:
    """
    Return Plotly layout overrides for the active theme.

    IMPORTANT: This dict includes a 'legend' key. If you need to customize
    the legend, do NOT pass legend= separately to update_layout — it will
    collide. Instead, merge first:

        base = get_plotly_layout(mode)
        base["legend"] = {"orientation": "h", "y": -0.2}
        fig.update_layout(**base)

    Returns ONLY top-level keys (no xaxis/yaxis). Use get_axis_style()
    for axis-level theming and merge into your axis dicts.
    """
    t = tokens(mode)
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font": {"color": t["text"], "family": "system-ui, -apple-system, sans-serif"},
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": t["border"],
            "font": {"color": t["text"]},
        },
    }


def get_axis_style(mode: str = "dark") -> dict:
    """Return axis-level theme overrides (gridcolor, linecolor, etc.).

    Merge into your xaxis/yaxis dicts:
        xaxis={**get_axis_style(mode), "tickangle": -45}
    """
    t = tokens(mode)
    return {
        "gridcolor": t["grid"],
        "linecolor": t["border"],
        "zerolinecolor": t["border"],
        "tickcolor": t["text_muted"],
    }


# Instrument-specific colors (don't change with theme)
SCALE_COLORS = {
    "dcdq": "#a78bfa",  # purple - motor
    "rbs":  "#38bdf8",  # blue - repetitive
    "scq":  "#4ade80",  # green - social comm
    "ados": "#f472b6",  # pink - ADOS
    "cbcl": "#fbbf24",  # yellow - internalizing
    "sp":   "#e879f9",  # magenta - sensory profile
    "seq":  "#f97316",  # orange - sensory experiences
    "isq":  "#06b6d4",  # cyan - infant sensory
    "css":  "#f87171",  # red - severity
}
