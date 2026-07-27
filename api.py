import os

from fastapi import FastAPI

from src.config import get_provider_name

app = FastAPI(title="DocuMind API", version="1.0.0")


@app.get("/")
def home():
    return {
        "message": "DocuMind API is ready",
        "status": "ok",
        "provider": get_provider_name(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
