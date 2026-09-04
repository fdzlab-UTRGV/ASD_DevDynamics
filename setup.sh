#!/bin/zsh
# ── ASD Phenotypic Architecture — setup & run ─────────────────────────────────
# Fernandez et al., Cell Reports Medicine
#
# Usage: bash setup.sh
#
# Finds the latest app zip in ~/Downloads, extracts it, creates a venv,
# installs dependencies, and opens the app in Google Chrome.

DOWNLOADS=~/Downloads
DOCUMENTS=~/Documents/asd_phenotypic_arch

# ── Find latest zip ────────────────────────────────────────────────────────────
LATEST_ZIP=$(ls -t "$DOWNLOADS"/phenotypic_architecture_asd*.zip 2>/dev/null | head -1)

if [ -z "$LATEST_ZIP" ]; then
  echo "❌  No phenotypic_architecture_asd*.zip found in ~/Downloads"
  echo "    Place the app zip in ~/Downloads and re-run."
  exit 1
fi

echo "✓  Found: $LATEST_ZIP"

# ── Kill any running instance ──────────────────────────────────────────────────
lsof -ti:8050 | xargs kill -9 2>/dev/null && echo "→  Killed previous instance on :8050" || true
sleep 1

# ── Read top-level folder name from zip ───────────────────────────────────────
ZIP_INNER=$(unzip -Z1 "$LATEST_ZIP" 2>/dev/null | head -1 | cut -d/ -f1)
if [ -z "$ZIP_INNER" ]; then
  echo "❌  Could not read zip contents"
  exit 1
fi
echo "→  Zip folder: $ZIP_INNER"

DEST="$DOCUMENTS/$ZIP_INNER"

# ── Extract fresh ──────────────────────────────────────────────────────────────
mkdir -p "$DOCUMENTS"
echo "→  Removing old folder and extracting..."
rm -rf "$DEST"
unzip -q "$LATEST_ZIP" -d "$DOCUMENTS"

if [ ! -d "$DEST" ]; then
  echo "❌  Extraction failed — $DEST not found"
  exit 1
fi

# ── Locate app.py ─────────────────────────────────────────────────────────────
APP_PY=$(find "$DEST" -maxdepth 2 -name "app.py" 2>/dev/null | head -1)
if [ -z "$APP_PY" ]; then
  echo "❌  app.py not found in $DEST"
  exit 1
fi

SPARK_DIR=$(dirname "$APP_PY")
VERSION=$(grep -o '"v[0-9][^"]*"' "$APP_PY" 2>/dev/null | head -1 | tr -d '"')
echo "✓  App directory: $SPARK_DIR ($VERSION)"
cd "$SPARK_DIR"

# ── Virtual environment ────────────────────────────────────────────────────────
if [ ! -f "venv/bin/python" ]; then
  echo "→  Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip --quiet

echo "→  Installing dependencies..."
pip install --quiet \
  dash \
  dash-bootstrap-components \
  plotly \
  pandas \
  numpy \
  scipy \
  statsmodels \
  scikit-learn \
  pyarrow \
  openpyxl

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ASD Phenotypic Architecture — Fernandez et al.  $VERSION"
echo "  http://127.0.0.1:8050"
echo "  Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python app.py &
APP_PID=$!

# Wait for server to start, then open in Chrome
for i in {1..30}; do
  sleep 1
  if curl -s http://127.0.0.1:8050 > /dev/null 2>&1; then
    open -a "Google Chrome" http://127.0.0.1:8050
    echo "✓  Opened in Google Chrome"
    break
  fi
done

wait $APP_PID
