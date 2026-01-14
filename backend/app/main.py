from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import router
from backend.app.config import *



app = FastAPI(
    title="AI Stock Market Analyzer",
    description="Multi-agent AI system for stock analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "API running"}
