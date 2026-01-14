# main.py
"""
Terminal-based execution script for Multi-Agent Stock Market Analysis System.
Prints final analysis results directly in the terminal.
"""

import asyncio
import argparse
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from mcp.master_control import MCPOrchestrator


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
    load_dotenv()

    required_keys = ["OPENAI_API_KEY", "NEWSAPI_KEY", "FINNHUB_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]

    if missing:
        raise ValueError(f"Missing required API keys: {', '.join(missing)}")

    logger.info("Environment setup complete")


# -------------------------------------------------------------------
# CLI Arguments
# -------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Stock Market Analysis System (Terminal)"
    )

    parser.add_argument(
        "--stocks",
        type=str,
        default="TSLA,NVDA,NET",
        help="Comma-separated stock tickers (e.g. TSLA,AAPL)"
    )

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
        "--save",
        action="store_true",
        help="Save output to JSON file"
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# Core Analysis
# -------------------------------------------------------------------

async def run_analysis(tickers, user_preferences):
    logger.info("=" * 80)
    logger.info("STARTING MULTI-AGENT STOCK MARKET ANALYSIS")
    logger.info(f"Tickers: {tickers}")
    logger.info("=" * 80)

    mcp = MCPOrchestrator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY")
    )

    result = await mcp.run_analysis(
        tickers=tickers,
        user_id="terminal_user",
        user_preferences=user_preferences
    )

    return result


# -------------------------------------------------------------------
# Terminal Output (FIXED & ROBUST)
# -------------------------------------------------------------------

def print_results(result: dict):
    print("\n" + "=" * 80)
    print("📊 FINAL STOCK ANALYSIS RESULT")
    print("=" * 80)

    # MCP wraps final output inside "data"
    data = result.get("data") or result

    recommendations = data.get("final_recommendations", [])

    if not recommendations:
        print("\n⚠️ No final recommendations returned.")
        return

    for i, rec in enumerate(recommendations, 1):
        print(f"\n#{i} 📈 {rec.get('ticker')}")
        print(f"  Action        : {rec.get('action')}")
        print(f"  Confidence    : {rec.get('confidence')}%")
        print(f"  Expected ROI  : {rec.get('expected_roi')}%")
        print(f"  Horizon       : {rec.get('time_horizon')}")
        print(f"  Consensus     : {rec.get('consensus_score')}")
        print(f"  Entry         : {rec.get('recommended_entry')}")
        print(f"  Target        : {rec.get('recommended_target')}")
        print(f"  Stop Loss     : {rec.get('recommended_stop')}")
        print(f"  Reason        : {rec.get('justification')}")

    allocation = data.get("portfolio_allocation", {})
    if allocation:
        print("\n💼 PORTFOLIO ALLOCATION")
        for a in allocation.get("allocations", []):
            print(f"  - {a['ticker']}: {a['allocation_pct']}%")

        print(f"  Cash Reserve  : {allocation.get('cash_reserve_pct')}%")

    print("\n" + "=" * 80)


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

def main():
    try:
        setup_environment()
        args = parse_arguments()

        tickers = [t.strip().upper() for t in args.stocks.split(",")]

        user_preferences = {
            "risk_profile": args.risk_profile,
            "time_horizon": args.time_horizon,
            "max_position_size": 5.0 if args.risk_profile == "aggressive" else 3.0
        }

        result = asyncio.run(
            run_analysis(tickers, user_preferences)
        )

        # Print results to terminal
        print_results(result)

        # Optional save
        if args.save:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to {filename}")

        return 0

    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
