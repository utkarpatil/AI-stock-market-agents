from fastapi import APIRouter, HTTPException
from backend.app.schemas import AnalyzeRequest, AnalyzeResponse
from backend.core.analysis import run_analysis

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_stock(request: AnalyzeRequest):
    try:
        return await run_analysis(request.company_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
