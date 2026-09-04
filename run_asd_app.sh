#!/bin/zsh
# ── ASD Phenotypic Architecture — setup & run ─────────────────────────────────
# Fernandez et al., Cell Reports Medicine
#
#   zsh run_asd_app.sh
#
# Replaces the older SPARK Behavioral Fingerprint launcher. Three problems in
# that script made it unable to run this app:
#
#   1. It globbed for spark_dash_v200*.zip / spark_v200*.zip. The current app
#      ships as phenotypic_architecture_asd_v*.zip, so the glob missed it and
#      — if an old v200 zip was still in ~/Downloads — SILENTLY LAUNCHED THE
#      STALE APP. That is the dangerous failure: v3.0.0 and earlier carry known
#      statistical defects (sex never entered any adjustment model; ridge had
#      no intercept; FDR q-values inflated), so you could believe you were
#      running corrected code while running defective code.
#   2. Its pip list omitted scipy (a hard import-time dependency — the app
#      cannot boot without it) and statsmodels (mixed models), while installing
#      mofapy2 and umap-learn, which this app does not use at all.
#   3. It ran `rm -rf` on the install folder, destroying the previous install
#      including its venv and anything else kept in that folder.
#
# This script installs versions side by side and never deletes anything.

set -u
# zsh aborts a command when a glob matches nothing; we want an empty result
# instead so the checks below can print a useful message.
setopt NULL_GLOB 2>/dev/null || true

DOWNLOADS=~/Downloads
# Install root. Kept as SFARI_data_analysis to match existing habit; change
# this one line to relocate.
DOCUMENTS=~/Documents/SFARI_data_analysis

# Where "Save run" bundles are written. Exported so the app picks it up.
export ASD_APP_OUTPUT_DIR="${ASD_APP_OUTPUT_DIR:-$HOME/Documents/SFARI_data_analysis/_runs}"

# ── Find the newest app zip ───────────────────────────────────────────────────
# Expand the glob into an array first. Piping a bare glob into `ls` is unsafe:
# with NULL_GLOB the unmatched pattern disappears and `ls -t` silently lists the
# CURRENT directory instead, happily returning some unrelated file as the "zip".
CANDIDATES=("$DOWNLOADS"/phenotypic_architecture_asd*.zip)

LATEST_ZIP=""
if [ ${#CANDIDATES[@]} -gt 0 ]; then
  # Newest by mtime among the real matches only.
  LATEST_ZIP=$(ls -t -- "${CANDIDATES[@]}" 2>/dev/null | head -1)
fi

if [ -z "$LATEST_ZIP" ] || [ ! -f "$LATEST_ZIP" ]; then
  echo "❌  No phenotypic_architecture_asd*.zip found in $DOWNLOADS"
  echo "    Download the app zip, then re-run this script."
  # Deliberately does NOT fall back to older spark_dash_v200*/spark_v200* zips:
  # silently running superseded code is worse than not starting.
  OLD_ZIPS=("$DOWNLOADS"/spark_dash_v200*.zip "$DOWNLOADS"/spark_v200*.zip)
  if [ ${#OLD_ZIPS[@]} -gt 0 ]; then
    echo ""
    echo "    Note: ${#OLD_ZIPS[@]} older spark_*_v200 zip(s) are present in $DOWNLOADS."
    echo "    Those are a DIFFERENT, superseded app and are ignored on purpose —"
    echo "    launching them would silently run code with known statistical defects."
  fi
  exit 1
fi

echo "✓  Found: $LATEST_ZIP"

# NOTE: the running instance on :8050 is stopped LATER, just before launch.
# Killing it up front means a failure during extract or pip install leaves you
# with no app running at all — worse than the state you started in.

# ── Read the top-level folder name from inside the zip ───────────────────────
ZIP_INNER=$(unzip -Z1 "$LATEST_ZIP" 2>/dev/null | head -1 | cut -d/ -f1)
if [ -z "$ZIP_INNER" ]; then
  echo "❌  Could not read zip contents"
  exit 1
fi

DEST="$DOCUMENTS/$ZIP_INNER"

# ── Install without destroying anything ──────────────────────────────────────
# No `rm -rf` anywhere in this script. If the target exists it is left exactly
# as-is and this copy goes in beside it.
mkdir -p "$DOCUMENTS"

if [ -d "$DEST" ]; then
  echo "⚠  $DEST already exists — leaving it untouched."
  DEST="${DEST}__reinstall_$(date +%Y-%m-%d_%H-%M-%S)"
  echo "→  Installing this copy alongside it:"
  echo "   $DEST"
fi

echo "→  Extracting..."
TMP_EXTRACT=$(mktemp -d "${TMPDIR:-/tmp}/asd_app_XXXXXX")
if ! unzip -q "$LATEST_ZIP" -d "$TMP_EXTRACT"; then
  echo "❌  Unzip failed"
  rm -rf "$TMP_EXTRACT"
  exit 1
fi
if [ ! -d "$TMP_EXTRACT/$ZIP_INNER" ]; then
  echo "❌  Expected folder '$ZIP_INNER' not found inside the zip"
  rm -rf "$TMP_EXTRACT"
  exit 1
fi
# Move into place only after a successful extract, so a corrupt zip can never
# leave a half-installed folder behind.
mv "$TMP_EXTRACT/$ZIP_INNER" "$DEST"
rm -rf "$TMP_EXTRACT"

# ── Locate app.py ────────────────────────────────────────────────────────────
APP_PY=$(find "$DEST" -maxdepth 2 -name "app.py" 2>/dev/null | head -1)
if [ -z "$APP_PY" ]; then
  echo "❌  app.py not found in $DEST"
  exit 1
fi

APP_DIR=$(dirname "$APP_PY")
VERSION=$(grep -o '"v[0-9][^"]*"' "$APP_PY" 2>/dev/null | head -1 | tr -d '"')
cd "$APP_DIR"

# ── Shared virtual environment ───────────────────────────────────────────────
# One level above the version folders, so installing a new version does not
# rebuild it and removing a version folder does not delete it.
VENV="$DOCUMENTS/venv"

if [ ! -f "$VENV/bin/python" ]; then
  echo "→  Creating shared virtual environment at $VENV ..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

echo "→  Installing dependencies..."
if [ -f "$APP_DIR/requirements.txt" ]; then
  # Authoritative list, pinned by the app itself.
  python -m pip install --quiet -r "$APP_DIR/requirements.txt"
else
  python -m pip install --quiet \
    dash dash-bootstrap-components plotly pandas numpy \
    scipy statsmodels scikit-learn pyarrow openpyxl
fi

# ── Verify the app can actually import before launching ──────────────────────
# Catches a missing dependency here, with a readable message, instead of as a
# traceback in a browser tab that never loads.
if ! python -c "import dash, pandas, numpy, scipy, statsmodels, pyarrow" 2>/dev/null; then
  echo "❌  A required package is missing from $VENV"
  python -c "import dash, pandas, numpy, scipy, statsmodels, pyarrow"
  exit 1
fi

mkdir -p "$ASD_APP_OUTPUT_DIR"

# ── Stop anything already on :8050 (only now that everything above succeeded) ─
if lsof -ti:8050 >/dev/null 2>&1; then
  lsof -ti:8050 | xargs kill -9 2>/dev/null || true
  echo "→  Stopped previous instance on :8050"
  sleep 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ASD Phenotypic Architecture — Fernandez et al.  $VERSION"
echo "  http://127.0.0.1:8050"
echo ""
echo "  Running from : $APP_DIR"
echo "  Saved runs   : $ASD_APP_OUTPUT_DIR"
echo "  Environment  : $VENV"
echo ""
echo "  Installs present in $DOCUMENTS:"
ls -1d "$DOCUMENTS"/*/ 2>/dev/null \
  | grep -v -e '/venv/$' -e '/_runs/$' \
  | sed 's|.*/\([^/]*\)/$|    \1|'
echo ""
echo "  Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python app.py &
APP_PID=$!

# Open a browser once the server responds.
for i in {1..30}; do
  sleep 1
  if curl -s http://127.0.0.1:8050 > /dev/null 2>&1; then
    open -a "Google Chrome" http://127.0.0.1:8050 2>/dev/null \
      || open http://127.0.0.1:8050 2>/dev/null || true
    break
  fi
done

wait $APP_PID
