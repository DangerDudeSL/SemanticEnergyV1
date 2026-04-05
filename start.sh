#!/usr/bin/env bash
# start.sh — Launch the Semantic Energy Live Application (Linux/macOS)

set -e

echo "Starting Semantic Energy Live Application..."

# 1. Start the FastAPI Backend
echo "Starting Backend on http://127.0.0.1:8000..."
(cd backend && python app.py) &
BACKEND_PID=$!

# Give the backend a few seconds to initialize
sleep 3

# 2. Start the Frontend Simple HTTP Server
echo "Starting Frontend on http://127.0.0.1:3000..."
(cd frontend && python -m http.server 3000) &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "Application is running!"
echo "Frontend: http://127.0.0.1:3000"
echo "Backend API: http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop both servers."
echo "========================================================"
echo ""

cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    echo "Shutdown complete."
}
trap cleanup EXIT INT TERM

wait
