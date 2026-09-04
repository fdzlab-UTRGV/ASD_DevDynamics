"""Layout modules - one per tab plus the sidebar."""
from layouts.sidebar import sidebar_layout

# Tab layouts - imported lazily by callbacks/main.py to keep this module light
__all__ = ["sidebar_layout"]
