"""
callbacks/__init__.py
─────────────────────────────────────────────────────────────────────────────
Single entry point for callback registration.

Calling register_all(app) wires every callback in every callback module.
All callbacks live inside register() functions — no callbacks are registered
outside of these functions.

Figure mapping (Fernandez et al., Cell Reports Medicine):
  correlations    → Figure 1:   unadjusted Pearson r matrix
  part2           → Figure 2:   √ΔR² hubness index + PCA
                    Figure 3A-B: split-half reproducibility + bootstrap
                    Figure 5:   anxiety-adjustment sensitivity
  domains         → Figure 4:   age-stratified hub reorganization
  dev_age_analysis → Figure 6:  developmental domain–psychopathology coupling
  ridge           → Figure 3C:  out-of-sample ridge regression (holdout)
"""

from callbacks import uploads, theme, main
from callbacks import correlations, part2, domains, dev_age_analysis, ridge, cohort


def register_all(app):
    uploads.register(app)
    theme.register(app)
    main.register(app)
    correlations.register(app)
    part2.register(app)
    domains.register(app)
    dev_age_analysis.register(app)
    ridge.register(app)
    cohort.register(app)
