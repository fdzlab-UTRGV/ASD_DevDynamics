"""
callbacks/theme.py
─────────────────────────────────────────────────────────────────────────────
Theme toggle - pure clientside callback that flips the data-theme attribute
on <html>. CSS variables in theme.css do all the actual styling.

theme-store holds the current mode ("dark" | "light") for any callback
that needs to render Plotly figures (those need the mode to set colors).
"""

from dash import Input, Output, State


def register(app):

    # Clientside: toggle the data-theme attribute on <html> + persist in localStorage
    app.clientside_callback(
        """
        function(n_clicks, current_mode) {
            // First call (page load) - read from localStorage
            if (!n_clicks) {
                var saved = null;
                try { saved = localStorage.getItem("spark_dash_theme"); } catch(e) {}
                var theme = saved || "dark";
                document.documentElement.setAttribute("data-theme", theme);
                return [theme, theme === "light" ? "☾ Dark" : "☀ Light"];
            }
            // Toggle
            var next = (current_mode === "light") ? "dark" : "light";
            document.documentElement.setAttribute("data-theme", next);
            try { localStorage.setItem("spark_dash_theme", next); } catch(e) {}
            return [next, next === "light" ? "☾ Dark" : "☀ Light"];
        }
        """,
        Output("theme-store",  "data"),
        Output("btn-theme",    "children"),
        Input("btn-theme",     "n_clicks"),
        State("theme-store",   "data"),
    )
