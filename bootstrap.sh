#!/usr/bin/env bash
# AlphaEngine bootstrap.
# One-command setup: deps -> tests -> backtest -> dashboard.
# Idempotent: safe to re-run.
#
# Usage:
#   ./bootstrap.sh                    # full bootstrap, ends by starting dashboard
#   ./bootstrap.sh --no-dashboard     # skip starting the dashboard
#   ./bootstrap.sh --skip-backtest    # skip the backtest step

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

NO_DASH=false
SKIP_BT=false
for arg in "$@"; do
  case "$arg" in
    --no-dashboard) NO_DASH=true ;;
    --skip-backtest) SKIP_BT=true ;;
  esac
done

bold() { printf "\n\033[1m%s\033[0m\n" "$1"; }
ok() { printf "  \033[32mok\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!!\033[0m %s\n" "$1"; }
err() { printf "  \033[31mxx\033[0m %s\n" "$1"; }

bold "1. Python check"
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found. Install Python 3.10+."
  exit 1
fi
PYV="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ok "python3 ${PYV}"

bold "2. Virtual environment"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  ok "created .venv"
else
  ok ".venv already exists"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
ok "pip upgraded"

bold "3. Dependencies"
pip install -r requirements.txt --quiet
ok "core deps installed"

bold "4. .env file"
if [ ! -f ".env" ]; then
  cp .env.example .env
  warn "created .env from template. Open it and paste your Alpaca paper keys, or use the dashboard /settings page."
else
  ok ".env already exists"
fi

bold "5. Smoke tests"
python run_smoke.py "bootstrap" || { err "smoke tests failed"; exit 1; }
ok "smoke tests passed"

if [ "$SKIP_BT" = false ]; then
  bold "6. Backtest"
  if grep -q "^ALPACA_API_KEY=.\+" .env 2>/dev/null; then
    mkdir -p backtest_out
    set +e
    python -m alphaengine.backtest --config config.yaml --out backtest_out/equity.csv
    BT_RC=$?
    set -e
    if [ $BT_RC -eq 0 ]; then
      ok "backtest complete (see backtest_out/equity.csv)"
    else
      warn "backtest run returned non-zero. Check logs above."
    fi
  else
    warn "no ALPACA_API_KEY in .env yet, skipping backtest. Run again after pasting keys."
  fi
else
  warn "backtest skipped per flag"
fi

if [ "$NO_DASH" = false ]; then
  bold "7. Starting dashboard"
  echo "  Open http://localhost:8000"
  echo "  Press Ctrl-C to stop."
  python -m alphaengine.web.app
else
  bold "Done. To start the dashboard later:"
  echo "  source .venv/bin/activate && python -m alphaengine.web.app"
fi
