# mcp/master_control.py
"""
MCP (Master Control Program): Orchestrates all agents using LangGraph.
Controls workflow, handles errors, and manages state.
"""

from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from datetime import datetime
from loguru import logger
import json
import asyncio


# Import all agents
import sys
sys.path.append('..')
from agents.macro_analyst import MacroAnalystAgent
from agents.quant_agent import QuantAgent
from agents.fundamental_agent import FundamentalAgent
from agents.risk_manger_agent import RiskManagerAgent
from agents.statergy_evaluator import StrategyEvaluatorAgent


class AgentState(TypedDict):
    """Shared state between all agents."""
    # Input
    tickers: List[str]
    user_id: str
    user_preferences: Dict[str, Any]
    
    # Agent outputs
    macro_report: Dict[str, Any]
    trade_signals: List[Dict[str, Any]]
    company_reports: List[Dict[str, Any]]
    risk_assessments: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
    
    # Metadata
    errors: Annotated[List[str], operator.add]
    timestamp: str
    workflow_complete: bool


class MCPOrchestrator:
    """
    Master Control Program that orchestrates all trading agents.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        newsapi_key: str,
        finnhub_api_key: str,
        model: str = "gpt-4-turbo-preview"
    ):
        """Initialize all agents and workflow."""
        logger.info("Initializing MCP Orchestrator")
        
        # Initialize all agents
        self.macro_agent = MacroAnalystAgent(openai_api_key, newsapi_key, model)
        self.quant_agent = QuantAgent(openai_api_key, model)
        self.fundamental_agent = FundamentalAgent(openai_api_key, finnhub_api_key, model)
        self.risk_agent = RiskManagerAgent(openai_api_key, model)
        self.strategy_agent = StrategyEvaluatorAgent(openai_api_key, model)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("MCP Orchestrator initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow connecting all agents.
        
        Workflow:
        START → Macro Analyst → [Quant, Fundamental] → Risk Manager → Strategy Evaluator → END
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("macro_analysis", self._macro_node)
        workflow.add_node("quant_analysis", self._quant_node)
        workflow.add_node("fundamental_analysis", self._fundamental_node)
        workflow.add_node("risk_analysis", self._risk_node)
        workflow.add_node("strategy_evaluation", self._strategy_node)
        
        # Define edges (workflow flow)
        workflow.set_entry_point("macro_analysis")
        
        # After macro, run quant and fundamental in parallel (both depend on macro)
        workflow.add_edge("macro_analysis", "quant_analysis")
        workflow.add_edge("macro_analysis", "fundamental_analysis")
        
        # After both quant and fundamental complete, run risk analysis
        workflow.add_edge("quant_analysis", "risk_analysis")
        workflow.add_edge("fundamental_analysis", "risk_analysis")
        
        # After risk analysis, run strategy evaluation
        workflow.add_edge("risk_analysis", "strategy_evaluation")
        
        # After strategy evaluation, end
        workflow.add_edge("strategy_evaluation", END)
        
        # Compile the graph
        app = workflow.compile()
        
        logger.info("Workflow graph compiled successfully")
        return app
    
    async def _macro_node(self, state: AgentState) -> AgentState:
        """Execute macro analysis node."""
        logger.info("Executing Macro Analysis Node")
        
        try:
            result = await self.macro_agent.analyze({
                "sectors": ["technology", "finance", "healthcare"],
                "lookback_days": 7,
                "include_global": True
            })
            
            if result.get("success"):
                state["macro_report"] = result.get("data", {})
                logger.info("Macro analysis completed successfully")
            else:
                error_msg = f"Macro analysis failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["macro_report"] = {}
        
        except Exception as e:
            error_msg = f"Macro node exception: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["macro_report"] = {}
        
        return {
    "macro_report": state["macro_report"]
    }

    
    async def _quant_node(self, state: AgentState) -> AgentState:
        """Execute quant analysis node."""
        logger.info("Executing Quant Analysis Node")
        
        try:
            result = await self.quant_agent.analyze({
                "tickers": state["tickers"],
                "period": "3mo",
                "macro_report": state.get("macro_report", {})
            })
            
            if result.get("success"):
                data = result.get("data", {})
                state["trade_signals"] = data.get("trade_signals", [])
                logger.info(f"Quant analysis completed: {len(state['trade_signals'])} signals")
            else:
                error_msg = f"Quant analysis failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["trade_signals"] = []
        
        except Exception as e:
            error_msg = f"Quant node exception: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["trade_signals"] = []
        
        return {
    "trade_signals": state["trade_signals"]
    }

    
    async def _fundamental_node(self, state: AgentState) -> AgentState:
        """Execute fundamental analysis node."""
        logger.info("Executing Fundamental Analysis Node")
        
        try:
            result = await self.fundamental_agent.analyze({
                "tickers": state["tickers"],
                "use_memory": True
            })
            
            if result.get("success"):
                data = result.get("data", {})
                state["company_reports"] = data.get("company_reports", [])
                logger.info(f"Fundamental analysis completed: {len(state['company_reports'])} reports")
            else:
                error_msg = f"Fundamental analysis failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["company_reports"] = []
        
        except Exception as e:
            error_msg = f"Fundamental node exception: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["company_reports"] = []
        
        return {
    "company_reports": state["company_reports"]
    }

    
    async def _risk_node(self, state: AgentState) -> AgentState:
        """Execute risk analysis node."""
        logger.info("Executing Risk Analysis Node")
        
        try:
            # Wait for both quant and fundamental to complete
            if not state.get("trade_signals") or not state.get("company_reports"):
                logger.warning("Waiting for quant and fundamental analysis...")
                # In a real implementation, LangGraph handles this automatically
            
            result = await self.risk_agent.analyze({
                "user_id": state.get("user_id", "default"),
                "trade_signals": state.get("trade_signals", []),
                "company_reports": state.get("company_reports", []),
                "macro_report": state.get("macro_report", {}),
                "simulate_scenarios": True
            })
            
            if result.get("success"):
                data = result.get("data", {})
                state["risk_assessments"] = data.get("risk_assessments", [])
                logger.info(f"Risk analysis completed: {len(state['risk_assessments'])} assessments")
            else:
                error_msg = f"Risk analysis failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["risk_assessments"] = []
        
        except Exception as e:
            error_msg = f"Risk node exception: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["risk_assessments"] = []
        
        return {
    "risk_assessments": state["risk_assessments"]
    }

    
    async def _strategy_node(self, state: AgentState) -> AgentState:
        """Execute strategy evaluation node."""
        logger.info("Executing Strategy Evaluation Node")
        
        try:
            result = await self.strategy_agent.analyze({
                "macro_report": state.get("macro_report", {}),
                "trade_signals": state.get("trade_signals", []),
                "company_reports": state.get("company_reports", []),
                "risk_assessments": state.get("risk_assessments", []),
                "user_preferences": state.get("user_preferences", {})
            })
            
            if result.get("success"):
                data = result.get("data", {})
                state["final_recommendations"] = data.get("final_recommendations", [])
                logger.info(f"Strategy evaluation completed: {len(state['final_recommendations'])} recommendations")
            else:
                error_msg = f"Strategy evaluation failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["final_recommendations"] = []
        
        except Exception as e:
            error_msg = f"Strategy node exception: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["final_recommendations"] = []
        
        state["workflow_complete"] = True
        return {
    "final_recommendations": state["final_recommendations"],
    "workflow_complete": True
    }

    
    async def run_analysis(
        self,
        tickers: List[str],
        user_id: str = "default",
        user_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run complete multi-agent analysis workflow.
        
        Args:
            tickers: List of stock tickers to analyze
            user_id: User identifier for risk profile
            user_preferences: User trading preferences
            
        Returns:
            Complete analysis results from all agents
        """
        logger.info(f"Starting MCP analysis for tickers: {tickers}")
        
        # Initialize state
        initial_state: AgentState = {
            "tickers": tickers,
            "user_id": user_id,
            "user_preferences": user_preferences or {},
            "macro_report": {},
            "trade_signals": [],
            "company_reports": [],
            "risk_assessments": [],
            "final_recommendations": [],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
            "workflow_complete": False
        }
        
        try:
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Format output
            output = {
                "success": final_state.get("workflow_complete", False),
                "timestamp": final_state.get("timestamp"),
                "tickers_analyzed": tickers,
                "macro_environment": final_state.get("macro_report", {}),
                "technical_signals": final_state.get("trade_signals", []),
                "fundamental_reports": final_state.get("company_reports", []),
                "risk_assessments": final_state.get("risk_assessments", []),
                "final_recommendations": final_state.get("final_recommendations", []),
                "errors": final_state.get("errors", []),
                "summary": self._generate_summary(final_state)
            }
            
            logger.info("MCP analysis completed successfully")
            return output
        
        except Exception as e:
            logger.error(f"MCP workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "tickers_analyzed": tickers
            }
    
    def _generate_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate executive summary of analysis."""
        recommendations = state.get("final_recommendations", [])
        
        if not recommendations:
            return {
                "message": "No recommendations generated",
                "status": "incomplete"
            }
        
        # Count by action
        actions = {}
        for rec in recommendations:
            action = rec.get("action", "UNKNOWN")
            actions[action] = actions.get(action, 0) + 1
        
        # Get top recommendation
        top_rec = recommendations[0] if recommendations else {}
        
        return {
            "total_opportunities": len(recommendations),
            "actions_breakdown": actions,
            "top_recommendation": {
                "ticker": top_rec.get("ticker", "N/A"),
                "action": top_rec.get("action", "N/A"),
                "confidence": top_rec.get("confidence", 0),
                "expected_roi": top_rec.get("expected_roi", 0)
            },
            "high_confidence_trades": len([r for r in recommendations if r.get("confidence", 0) >= 80]),
            "average_confidence": round(sum(r.get("confidence", 0) for r in recommendations) / len(recommendations), 2) if recommendations else 0,
            "errors_encountered": len(state.get("errors", [])),
            "status": "complete" if state.get("workflow_complete") else "incomplete"
        }
    
    def visualize_workflow(self, output_path: str = "workflow_graph.png"):
        """
        Generate visual representation of the workflow.
        Requires graphviz installation.
        """
        try:
            from IPython.display import Image, display
            
            # Get mermaid diagram
            mermaid = self.workflow.get_graph().draw_mermaid()
            
            logger.info("Workflow visualization:")
            print(mermaid)
            
            return mermaid
        
        except Exception as e:
            logger.warning(f"Could not generate visualization: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize orchestrator
    mcp = MCPOrchestrator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY")
    )
    
    async def run_test():
        """Test the complete workflow."""
        print("=" * 80)
        print("MULTI-AGENT STOCK MARKET ANALYSIS SYSTEM")
        print("=" * 80)
        
        # Define analysis parameters
        tickers = ["AAPL", "MSFT", "GOOGL"]
        user_prefs = {
            "risk_profile": "moderate",
            "time_horizon": "medium-term",
            "max_position_size": 5.0
        }
        
        print(f"\nAnalyzing: {', '.join(tickers)}")
        print(f"User Profile: {user_prefs['risk_profile']}")
        print("\nStarting analysis...\n")
        
        # Run analysis
        result = await mcp.run_analysis(
            tickers=tickers,
            user_id="trader_001",
            user_preferences=user_prefs
        )
        
        # Display results
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        if result.get("success"):
            print("\n✓ Analysis completed successfully!")
            
            summary = result.get("summary", {})
            print(f"\n📊 Summary:")
            print(f"  - Total Opportunities: {summary.get('total_opportunities', 0)}")
            print(f"  - High Confidence Trades: {summary.get('high_confidence_trades', 0)}")
            print(f"  - Average Confidence: {summary.get('average_confidence', 0)}%")
            print(f"  - Actions Breakdown: {summary.get('actions_breakdown', {})}")
            
            top = summary.get('top_recommendation', {})
            print(f"\n🎯 Top Recommendation:")
            print(f"  - Ticker: {top.get('ticker', 'N/A')}")
            print(f"  - Action: {top.get('action', 'N/A')}")
            print(f"  - Confidence: {top.get('confidence', 0)}%")
            print(f"  - Expected ROI: {top.get('expected_roi', 0):.2f}%")
            
            # Display all recommendations
            print(f"\n📋 All Recommendations:")
            for i, rec in enumerate(result.get("final_recommendations", []), 1):
                print(f"\n  {i}. {rec.get('ticker')} - {rec.get('action')}")
                print(f"     Confidence: {rec.get('confidence')}% | ROI: {rec.get('expected_roi')}%")
                print(f"     Justification: {rec.get('justification', 'N/A')[:100]}...")
            
            # Save full results
            with open('analysis_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("\n✓ Full results saved to 'analysis_results.json'")
        
        else:
            print("\n✗ Analysis failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        if result.get("errors"):
            print(f"\n⚠ Warnings/Errors encountered:")
            for error in result.get("errors", []):
                print(f"  - {error}")
        
        print("\n" + "=" * 80)
    
    # Run the test
    asyncio.run(run_test())