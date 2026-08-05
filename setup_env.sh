#!/usr/bin/env bash
# ============================================================================
# setup_env.sh  ·  ADME-Lab environment setup
# Creates ./.venv and installs the package + dependencies. Run:  bash setup_env.sh
# Robust to a missing ensurepip on Debian/Ubuntu (bootstraps pip without sudo).
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG="$HERE/install_core.log"

# 1) Create the virtual environment.
if [ ! -x .venv/bin/python ]; then
  if python3 -m venv .venv 2>/dev/null && [ -x .venv/bin/pip ]; then
    echo "venv created"
  else
    echo "ensurepip unavailable; bootstrapping pip via get-pip.py (no sudo)"
    python3 -m venv --without-pip .venv
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    .venv/bin/python /tmp/get-pip.py
  fi
fi
source .venv/bin/activate

echo "=== START $(date) ===" > "$LOG"
echo "Python: $(python --version 2>&1)" >> "$LOG"

echo "--- [1/2] pip/setuptools/wheel ---" | tee -a "$LOG"
python -m pip install -U pip setuptools wheel --progress-bar off >> "$LOG" 2>&1

echo "--- [2/2] admelab + dependencies (rdkit, admet-ai, jupyter...) ---" | tee -a "$LOG"
python -m pip install -e ".[notebook]" --progress-bar off >> "$LOG" 2>&1
echo "=== DONE $(date) exit=$? ===" >> "$LOG"

echo "Full log at: $LOG"
python -c "import admelab; print('admelab', admelab.__version__, 'installed OK')"
echo "Next (optional): bash tools_setup.sh   # JRE + OPSIN for verified IUPAC naming"
