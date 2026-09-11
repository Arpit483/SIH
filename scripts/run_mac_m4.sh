#!/usr/bin/env bash
# ==============================================================================
# run_mac_m4.sh: Unified Startup Script for SatQuery AI on Apple Silicon (Mac M4)
# Smart India Hackathon SIH26167 | ISRO Earth Observation Scientist Suite
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "================================================================="
echo "🛰️  SatQuery AI — Unified Deployment on Apple Silicon (Mac M4)"
echo "   SIH26167 | Indian Space Research Organisation (ISRO)"
echo "================================================================="

# 1. Check Python
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.10+ on macOS."
    exit 1
fi
echo "✓ Python detected: $($PYTHON_BIN --version)"

# 2. Virtual Environment Setup
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment in .venv..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment active: $(which python)"

# 3. Install Python backend dependencies if needed
echo "🔍 Checking backend Python dependencies..."
pip install --upgrade pip --quiet
pip install -r "$PROJECT_ROOT/backend/requirements.txt" --quiet

# 4. Check Node.js & Next.js frontend dependencies
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm/node not found. Please install Node.js 18+."
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/node_modules" ]; then
    echo "📦 Installing Node.js frontend dependencies..."
    npm install
fi

# Create required runtime directories
mkdir -p "$PROJECT_ROOT/backend/uploads/thumbnails"
mkdir -p "$PROJECT_ROOT/backend/reports"
mkdir -p "$PROJECT_ROOT/backend/traces"

echo "-----------------------------------------------------------------"
echo "🚀 Starting Services:"
echo "   [1] FastAPI Backend (Apple Silicon MPS / PyTorch) -> http://localhost:8000"
echo "   [2] Next.js Dashboard (Mission-Control UI)        -> http://localhost:3001"
echo "-----------------------------------------------------------------"

cleanup() {
    echo ""
    echo "🛑 Shutting down SatQuery AI services..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Launch Backend
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend startup
sleep 2

# Launch Frontend
npm run dev &
FRONTEND_PID=$!

echo "================================================================="
echo "✅ SatQuery AI is running!"
echo "   Open Dashboard: http://localhost:3001"
echo "   API Docs:       http://localhost:8000/docs"
echo "   Press Ctrl+C to stop all services."
echo "================================================================="

wait
