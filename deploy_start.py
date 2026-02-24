"""
Combined deployment server for Hugging Face Spaces.
Serves both the FastAPI backend (/api/*) and the static frontend on a single port (7860).
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app import app as backend_app

# Create the combined app
combined_app = FastAPI(title="Semantic Energy")

combined_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the backend API routes under /api
# Re-register the backend endpoints under /api prefix
@combined_app.on_event("startup")
async def startup():
    """Trigger the backend's model loading on startup."""
    await backend_app.router.startup()

@combined_app.post("/chat")
async def chat_proxy(request: Request):
    """Proxy to the backend chat endpoint for backward compatibility."""
    from app import chat_endpoint
    return await chat_endpoint(request)

@combined_app.post("/api/chat")
async def api_chat_proxy(request: Request):
    """API-prefixed chat endpoint."""
    from app import chat_endpoint
    return await chat_endpoint(request)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
combined_app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@combined_app.get("/")
async def serve_index():
    """Serve the frontend index.html at root."""
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@combined_app.get("/{filename}")
async def serve_static(filename: str):
    """Serve any frontend file (styles.css, script.js, etc.)."""
    filepath = os.path.join(frontend_dir, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Deploy] Starting combined server on port {port}...", flush=True)
    uvicorn.run(combined_app, host="0.0.0.0", port=port)
