import os
import asyncio
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from loguru import logger

from mcp.master_control import MCPOrchestrator

load_dotenv()

app = FastAPI(title="Multi-Agent Stock Market Analysis API")


@app.get("/")
def health():
    return {"status": "API is live 🚀"}


@app.post("/analyze")
async def analyze_stocks(payload: dict):
    try:
        tickers = payload.get("tickers", [])
        user_preferences = payload.get("preferences", {})

        if not tickers:
            raise HTTPException(status_code=400, detail="Tickers are required")

        mcp = MCPOrchestrator(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            newsapi_key=os.getenv("NEWSAPI_KEY"),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY")
        )

        result = await mcp.run_analysis(
            tickers=tickers,
            user_id="api_user",
            user_preferences=user_preferences
        )

        return {"success": True, "data": result}

    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
