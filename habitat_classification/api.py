"""
FastAPI endpoint for habitat classification predictions with API key authentication.

Usage:
    python api.py

The server will start on http://0.0.0.0:4321
"""

import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from model import predict
from utils import decode_patch

HOST = "0.0.0.0"
PORT = 4321

# Load API key from environment variable
API_KEY = os.getenv("API_KEY", "your-default-api-key-change-this")

# Security scheme
security = HTTPBearer()


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    patch: str  # base64-encoded numpy array (15, 35, 35) float32


class PredictResponse(BaseModel):
    """Response body from /predict endpoint."""
    prediction: int  # Class index 0-70


app = FastAPI(
    title="Habitat Classification API",
    description="Classify Icelandic satellite image patches into 71 habitat types",
    version="1.0.0"
)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify the bearer token matches the API key.
    
    Raises HTTPException if token is invalid.
    """
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@app.get("/")
def index():
    """Health check endpoint (no authentication required)."""
    return {"status": "running", "message": "Habitat Classification API"}


@app.get("/api")
def api_info():
    """API information endpoint (no authentication required)."""
    return {
        "service": "habitat-classification",
        "version": "1.0.0",
        "authentication": "Bearer token required for /predict endpoint",
        "endpoints": {
            "/": "Health check",
            "/api": "API information",
            "/predict": "POST - Classify a patch (requires authentication)"
        }
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    request: PredictRequest,
    token: str = Depends(verify_token)
):
    """
    Classify a satellite image patch.

    Requires: Bearer token in Authorization header
    
    The patch should be base64-encoded numpy array of shape (15, 35, 35) with dtype float32.

    Returns the predicted habitat class index (0-70).
    """
    # Decode base64 to numpy array
    patch = decode_patch(request.patch)

    # Get prediction from model
    prediction = predict(patch)

    return PredictResponse(prediction=int(prediction))


if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    print(f"API Key: {API_KEY}")
    print("Set API_KEY environment variable to change the authentication key")
    uvicorn.run("api:app", host=HOST, port=PORT, reload=False)