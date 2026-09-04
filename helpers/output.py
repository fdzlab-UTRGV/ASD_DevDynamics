"""
helpers/output.py
─────────────────────────────────────────────────────────────────────────────
Save Run feature - writes a complete bundle to disk for a given analysis.

Each saved run goes into its own folder with parameters.json + result CSVs.
Old runs are never overwritten - new runs get fresh timestamped folders.
"""

import json
import datetime
from pathlib import Path
import pandas as pd


# Default output location. Honors the ASD_APP_OUTPUT_DIR environment variable
# (set by run_asd_app.sh) so "Save run" bundles land where the launcher expects.
import os

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "ASD_APP_OUTPUT_DIR",
        str(Path.home() / "Documents" / "asd_phenotypic_arch_outputs"),
    )
)


def get_output_dir() -> Path:
    """Return the configured output directory, creating it if needed."""
    path = DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(
    run_type: str,
    params: dict,
    dataframes: dict[str, pd.DataFrame] | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """
    Save an analysis run as an atomic bundle.

    Args:
        run_type: Short identifier ("correlations", "hubness", etc.)
        params: Parameters used (selected predictors/outcomes, covariates) - written as JSON
        dataframes: Dict of {csv_filename: DataFrame} to write as CSVs
        extra_files: Dict of {filename: bytes} for non-CSV outputs

    Returns:
        Path to the created run folder.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{run_type}_{timestamp}"
    run_dir = get_output_dir() / folder_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # Always write parameters.json with metadata
    params_full = {
        "run_type": run_type,
        "timestamp": timestamp,
        "iso_datetime": datetime.datetime.now().isoformat(),
        "params": params,
    }
    with open(run_dir / "parameters.json", "w") as f:
        json.dump(params_full, f, indent=2, default=str)

    # Write each DataFrame as CSV
    if dataframes:
        for filename, df in dataframes.items():
            if not filename.endswith(".csv"):
                filename = f"{filename}.csv"
            df.to_csv(run_dir / filename)

    # Write extra binary files (HDF5 models, etc.)
    if extra_files:
        for filename, content in extra_files.items():
            with open(run_dir / filename, "wb") as f:
                f.write(content)

    return run_dir


def list_runs(run_type: str | None = None) -> list[dict]:
    """List previously saved runs, optionally filtered by type."""
    output_dir = get_output_dir()
    if not output_dir.exists():
        return []
    runs = []
    for run_dir in sorted(output_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        params_file = run_dir / "parameters.json"
        if not params_file.exists():
            continue
        try:
            with open(params_file) as f:
                meta = json.load(f)
            if run_type and meta.get("run_type") != run_type:
                continue
            runs.append({
                "name": run_dir.name,
                "path": str(run_dir),
                "type": meta.get("run_type"),
                "timestamp": meta.get("timestamp"),
                "params": meta.get("params", {}),
            })
        except Exception:
            continue
    return runs
