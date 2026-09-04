/* theme.js - applies persisted theme on page load BEFORE Dash renders.
 * Avoids the flash-of-wrong-theme that would happen if Dash applied it later.
 */
(function() {
    var saved = null;
    try { saved = localStorage.getItem("spark_dash_theme"); } catch(e) {}
    var theme = saved || "dark";
    document.documentElement.setAttribute("data-theme", theme);
})();

/* Helper used by the toggle clientside callback */
window.sparkApplyTheme = function(mode) {
    var theme = (mode === "light") ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("spark_dash_theme", theme); } catch(e) {}
    return theme;
};
