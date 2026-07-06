#!/usr/bin/env bash
# deskbot installer (Linux/macOS secondary path — primary target is Windows,
# see install.ps1). Verifies Python 3.11+, installs Ollama if missing, pulls
# the RAM-tiered models, installs the deskbot package, and runs the doctor.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_MODEL_PULL="${SKIP_MODEL_PULL:-0}"

step() { echo -e "\033[36m==> $1\033[0m"; }
ok()   { echo -e "\033[32m    OK: $1\033[0m"; }
warn() { echo -e "\033[33m    WARN: $1\033[0m"; }

step "Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.11+ and re-run." >&2
    exit 1
fi
pyver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
pymaj="${pyver%%.*}"; pymin="${pyver##*.}"
if [ "$pymaj" -lt 3 ] || { [ "$pymaj" -eq 3 ] && [ "$pymin" -lt 11 ]; }; then
    echo "Python 3.11+ required, found $pyver" >&2
    exit 1
fi
ok "python $pyver"

step "Checking Ollama"
if ! command -v ollama >/dev/null 2>&1; then
    warn "Ollama not found. Installing via the official install script..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    ok "Ollama already installed"
fi

step "Making sure the Ollama server is running"
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Starting ollama serve in the background"
    nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
    sleep 3
fi
ok "Ollama server reachable"

step "Detecting RAM and picking a model tier"
if [ "$(uname)" = "Darwin" ]; then
    ram_bytes="$(sysctl -n hw.memsize)"
else
    ram_bytes="$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)"
fi
ram_gb="$(( ram_bytes / 1024 / 1024 / 1024 ))"
if [ "$ram_gb" -ge 28 ]; then tier="32gb"
elif [ "$ram_gb" -ge 14 ]; then tier="16gb"
else tier="8gb"
fi
ok "Detected ${ram_gb} GB RAM -> tier '$tier'"

text_model="$(python3 - "$root/deskbot/defaults/config.yaml" "$tier" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["models"]["ram_tiers"][sys.argv[2]]["text"])
PYEOF
)"
vision_model="$(python3 - "$root/deskbot/defaults/config.yaml" "$tier" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["models"]["ram_tiers"][sys.argv[2]]["vision"])
PYEOF
)"
ok "text model: $text_model | vision model: $vision_model"

if [ "$SKIP_MODEL_PULL" != "1" ]; then
    step "Pulling models (this can take a while on first run)"
    ollama pull "$text_model"
    ollama pull "$vision_model"
    ok "Models pulled"
else
    warn "Skipping model pull (SKIP_MODEL_PULL=1)"
fi

step "Installing the deskbot Python package"
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install --user -e "$root[dev]"
ok "deskbot package installed (editable, --user)"

step "Checking the browser layer (Chrome/Edge)"
if ! command -v google-chrome >/dev/null 2>&1 && ! command -v microsoft-edge >/dev/null 2>&1 && ! command -v chromium >/dev/null 2>&1; then
    warn "No system Chrome/Edge/Chromium found — installing Playwright's bundled Chromium as a fallback"
    python3 -m playwright install chromium
else
    ok "Found an installed browser for the browser layer"
fi

step "Running deskbot doctor"
python3 -m deskbot.cli doctor || deskbot doctor || true

echo
echo "Install complete. If 'deskbot' isn't found, add your user Python bin dir to PATH, then try:"
echo "    deskbot chat -p friend"
echo "    deskbot persona create"
