# main.py
"""
Main execution script for Multi-Agent Stock Market Analysis System.
Can be run from command line or imported as a module.
"""

import asyncio
import argparse
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from mcp.master_control import MCPOrchestrator
from tools.company_mapper import company_to_ticker   # ✅ NEW


# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)
logger.add(
    f"logs/main_{datetime.now().strftime('%Y%m%d')}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------

def setup_environment():
    """Load environment variables and validate."""
    load_dotenv()

    required_keys = ["OPENAI_API_KEY", "NEWSAPI_KEY", "FINNHUB_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

    logger.info("Environment setup complete")


# -------------------------------------------------------------------
# CLI Argument Parsing (UNCHANGED)
# -------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Stock Market Analysis System"
    )

    parser.add_argument(
        "--stocks",
        type=str,
        default="TSLA,NVDA,NET",
        help="Comma-separated list of stock tickers"
    )

    parser.add_argument("--user-id", type=str, default="default")

    parser.add_argument(
        "--risk-profile",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate"
    )

    parser.add_argument(
        "--time-horizon",
        choices=["short-term", "medium-term", "long-term"],
        default="medium-term"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="analysis_output.json"
    )

    parser.add_argument("--test", action="store_true")

    return parser.parse_args()


# -------------------------------------------------------------------
# CORE ASYNC ANALYSIS (UNCHANGED)
# -------------------------------------------------------------------

async def run_analysis(
    tickers: list,
    user_id: str,
    user_preferences: dict,
    output_path: str
):
    logger.info("=" * 80)
    logger.info("STARTING MULTI-AGENT STOCK MARKET ANALYSIS")
    logger.info("=" * 80)

    mcp = MCPOrchestrator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY")
    )

    result = await mcp.run_analysis(
        tickers=tickers,
        user_id=user_id,
        user_preferences=user_preferences
    )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    return result


# -------------------------------------------------------------------
# 🔥 STREAMLIT-SAFE WRAPPER (NEW)
# -------------------------------------------------------------------

def run_analysis_sync(
    companies: list,
    risk_profile: str = "moderate",
    time_horizon: str = "medium-term",
    user_id: str = "public_user",
    output_path: str = "analysis_output.json"
):
    """
    Synchronous wrapper for Streamlit.
    Accepts COMPANY NAMES, converts them to tickers.
    """

    setup_environment()

    tickers = []
    for company in companies:
        ticker = company_to_ticker(company)
        if not ticker:
            raise ValueError(f"Unsupported company name: {company}")
        tickers.append(ticker)

    user_preferences = {
        "risk_profile": risk_profile,
        "time_horizon": time_horizon,
        "max_position_size": 5.0 if risk_profile == "aggressive" else 3.0
    }

    return asyncio.run(
        run_analysis(
            tickers=tickers,
            user_id=user_id,
            user_preferences=user_preferences,
            output_path=output_path
        )
    )


# -------------------------------------------------------------------
# CLI ENTRY POINT (UNCHANGED)
# -------------------------------------------------------------------

def main():
    try:
        setup_environment()
        args = parse_arguments()

        if args.test:
            asyncio.run(
                run_analysis(
                    tickers=["AAPL"],
                    user_id="test_user",
                    user_preferences={
                        "risk_profile": "moderate",
                        "time_horizon": "medium-term",
                        "max_position_size": 3.0
                    },
                    output_path="test_results.json"
                )
            )
            return 0

        tickers = [t.strip().upper() for t in args.stocks.split(",")]

        user_preferences = {
            "risk_profile": args.risk_profile,
            "time_horizon": args.time_horizon,
            "max_position_size": 5.0 if args.risk_profile == "aggressive" else 3.0
        }

        asyncio.run(
            run_analysis(
                tickers=tickers,
                user_id=args.user_id,
                user_preferences=user_preferences,
                output_path=args.output
            )
        )

        return 0

    except Exception as e:
        logger.error(str(e))
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
