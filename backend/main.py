"""MedMem0 API - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from api.routes import patients_router, search_router, chat_router

settings = get_settings()

app = FastAPI(
    title="MedMem0 API",
    description="Longitudinal Patient Memory API powered by Mem0",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://medical-mem0.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(patients_router)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "MedMem0 API",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
