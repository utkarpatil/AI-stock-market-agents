import os
from dotenv import load_dotenv

from tools.company_mapper import company_to_ticker
from mcp.master_control import MCPOrchestrator

load_dotenv()


async def run_analysis(company_name: str) -> dict:
    ticker = company_to_ticker(company_name)

    if not ticker:
        raise ValueError("Invalid company name")

    orchestrator = MCPOrchestrator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY")
    )

    result = await orchestrator.run_analysis(
        tickers=[ticker],
        user_id="public_user",
        user_preferences={
            "risk_profile": "moderate",
            "time_horizon": "medium-term"
        }
    )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "Analysis failed"))

    recommendations = result.get("final_recommendations", [])
    if not recommendations:
        raise RuntimeError("No recommendations generated")

    top = recommendations[0]

    # -------------------------------
    # FALLBACK DATA SOURCES
    # -------------------------------
    technical = result.get("technical_signals", [{}])[0]
    risk = result.get("risk_assessments", [{}])[0]

    # -------------------------------
    # ACTION
    # -------------------------------
    action = (
        top.get("action")
        or technical.get("signal")
        or "HOLD"
    )

    # -------------------------------
    # CONFIDENCE
    # -------------------------------
    confidence = (
        top.get("confidence")
        or technical.get("confidence")
        or 50
    )

    # -------------------------------
    # REASON (VARIES BY CONFIDENCE)
    # -------------------------------
    reason = top.get("justification")

    if not reason:
        tech_signal = technical.get("signal", "mixed").lower()
        risk_level = risk.get("risk_category", "moderate").lower()
        roi = top.get("expected_roi", 0)

        direction = "downside pressure" if roi < 0 else "upside potential"

        if confidence >= 75:
            reason = (
                f"Strong technical alignment with a {tech_signal} trend and "
                f"{risk_level} risk profile supports this {action.lower()} view."
            )
        elif confidence >= 55:
            reason = (
                f"Current indicators show a {tech_signal} setup with "
                f"{risk_level} risk exposure, suggesting a {action.lower()} stance."
            )
        else:
            reason = (
                f"Mixed signals and {risk_level} risk conditions warrant a "
                f"cautious {action.lower()} approach."
            )

    # -------------------------------
    # FINAL RESPONSE (ONLY DATA)
    # -------------------------------
    return {
        "company": company_name,
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "expected_roi": top.get("expected_roi", 0),
        "horizon": top.get("time_horizon", "medium-term"),
        "entry": top.get("entry"),
        "target": top.get("target"),
        "stop_loss": top.get("stop_loss"),
        "reason": reason
    }
