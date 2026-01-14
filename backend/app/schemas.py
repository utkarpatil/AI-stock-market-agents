from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    company_name: str


class AnalyzeResponse(BaseModel):
    company: str
    ticker: str
    action: Optional[str] = None
    confidence: Optional[int] = None
    expected_roi: Optional[float] = None
    horizon: Optional[str] = None
    entry: Optional[float] = None
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    reason: Optional[str] = None
