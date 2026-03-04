#!/usr/bin/env bash
# PlexAI setup script
set -e

echo "🎬  PlexAI Setup"
echo "────────────────────────────────────────"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "✗  Python 3 is required. Install it with: sudo apt install python3 python3-full"
  exit 1
fi

# Ensure python3-venv module is available
if ! python3 -m venv --help &>/dev/null 2>&1; then
  echo "⟳  python3-venv not found, installing…"
  sudo apt install -y python3-full python3-venv
fi

# Remove broken venv dir if it exists but is incomplete
if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
  echo "⟳  Found incomplete venv, removing and recreating…"
  rm -rf venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "⟳  Creating virtual environment…"
  python3 -m venv venv
fi

# Final check
if [ ! -f "venv/bin/activate" ]; then
  echo "✗  Virtual environment creation failed. Try manually:"
  echo "    sudo apt install python3-full python3-venv"
  echo "    python3 -m venv venv"
  exit 1
fi

echo "✓  Virtual environment ready"

# Activate it
source venv/bin/activate

echo "⟳  Installing Python dependencies (this may take a minute on first run)…"
pip install -r requirements.txt

echo ""
echo "✓  Dependencies installed."
echo ""

# Create a convenient start script
cat > start.sh <<'EOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
source venv/bin/activate
exec python3 server.py
EOF
chmod +x start.sh

echo "🚀  To start PlexAI:"
echo "    bash start.sh"
echo ""
echo "Then open http://localhost:8000 in your browser."
