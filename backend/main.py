"""
Backend main entry point.

This file runs the FastAPI API defined in api/main.py.
"""

from api.main import app  # Import the FastAPI app from the api folder
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app locally
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,       # Port number
        reload=True      # Auto-reload when code changes (dev only)
    )
