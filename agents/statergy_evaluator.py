# agents/strategy_evaluator.py
"""
Strategy Evaluator Agent: Reviews all agents' advice and generates final recommendations.
Uses LangChain conversation memory for context.
"""

from typing import Dict, Any, List
from datetime import datetime
from agents.base_agent import BaseAgent
from loguru import logger
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import json


class StrategyEvaluatorAgent(BaseAgent):
    """
    Final decision maker that evaluates all agent outputs and generates ranked recommendations.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4-turbo-preview"
    ):
        super().__init__("StrategyEvaluator", openai_api_key, model)
        
        # LangChain conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        logger.info("Initialized Strategy Evaluator with conversation memory")
    
    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all agent outputs and generate final recommendations.
        
        Args:
            inputs: {
                "macro_report": {...},
                "trade_signals": [...],
                "company_reports": [...],
                "risk_assessments": [...],
                "user_preferences": {...}
            }
            
        Returns:
            {
                "final_recommendations": [
                    {
                        "ticker": "AAPL",
                        "action": "BUY/SELL/HOLD",
                        "confidence": 0-100,
                        "priority": 1-N,
                        "justification": "...",
                        "expected_roi": float,
                        "time_horizon": "short/medium/long",
                        "consensus_score": 0-100
                    }
                ],
                "portfolio_allocation": {...},
                "warnings": [...],
                "conflicts_detected": [...]
            }
        """
        try:
            logger.info("Starting strategy evaluation")
            
            macro_report = inputs.get("macro_report", {})
            trade_signals = inputs.get("trade_signals", [])
            company_reports = inputs.get("company_reports", [])
            risk_assessments = inputs.get("risk_assessments", [])
            user_prefs = inputs.get("user_preferences", {})
            
            # 1. Check for conflicts
            conflicts = self._detect_conflicts(
                trade_signals, company_reports, risk_assessments
            )
            
            # 2. Combine all analyses
            combined_analysis = self._combine_analyses(
                macro_report, trade_signals, company_reports, risk_assessments
            )
            
            # 3. Generate final recommendations with LLM
            recommendations = self._generate_final_recommendations(
                combined_analysis, user_prefs, conflicts
            )
            
            # 4. Rank and sort recommendations
            ranked_recommendations = self._rank_recommendations(recommendations)
            
            # 5. Generate portfolio allocation
            allocation = self._generate_portfolio_allocation(
                ranked_recommendations, risk_assessments
            )
            
            # 6. Store in conversation memory
            self._update_memory(inputs, ranked_recommendations)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "final_recommendations": ranked_recommendations,
                "portfolio_allocation": allocation,
                "conflicts_detected": conflicts,
                "total_opportunities": len(ranked_recommendations),
                "high_confidence_count": len([r for r in ranked_recommendations if r.get('confidence', 0) >= 80])
            }
            
            self.log_analysis(inputs, result)
            return self.create_success_response(result)
            
        except Exception as e:
            return self.handle_error(e, "strategy evaluation")
    
    def _detect_conflicts(
        self,
        trade_signals: List[Dict],
        company_reports: List[Dict],
        risk_assessments: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between different agents' analyses.
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group by ticker
        tickers = set([s.get('ticker') for s in trade_signals])
        
        for ticker in tickers:
            signal = next((s for s in trade_signals if s.get('ticker') == ticker), {})
            fundamental = next((c for c in company_reports if c.get('ticker') == ticker), {})
            risk = next((r for r in risk_assessments if r.get('ticker') == ticker), {})
            
            # Check for technical vs fundamental conflict
            tech_signal = signal.get('signal', '')
            fund_rec = fundamental.get('recommendation', '')
            
            if tech_signal == 'BUY' and fund_rec in ['sell', 'strong_sell']:
                conflicts.append({
                    'ticker': ticker,
                    'type': 'technical_vs_fundamental',
                    'description': f'Technical says {tech_signal} but Fundamental says {fund_rec}',
                    'severity': 'high'
                })
            
            # Check for high risk on strong buy signal
            if tech_signal == 'BUY' and risk.get('risk_score', 0) >= 8:
                conflicts.append({
                    'ticker': ticker,
                    'type': 'signal_vs_risk',
                    'description': f'Buy signal but risk score is {risk.get("risk_score")}',
                    'severity': 'medium'
                })
            
            # Check for valuation concerns
            if fund_rec in ['buy', 'strong_buy'] and fundamental.get('valuation') == 'overvalued':
                conflicts.append({
                    'ticker': ticker,
                    'type': 'recommendation_vs_valuation',
                    'description': 'Recommended to buy but stock is overvalued',
                    'severity': 'medium'
                })
        
        if conflicts:
            logger.warning(f"Detected {len(conflicts)} conflicts")
        
        return conflicts
    
    def _combine_analyses(
        self,
        macro_report: Dict,
        trade_signals: List[Dict],
        company_reports: List[Dict],
        risk_assessments: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Combine all analyses for each ticker.
        
        Returns:
            List of combined analyses per ticker
        """
        combined = []
        
        tickers = set([s.get('ticker') for s in trade_signals])
        
        for ticker in tickers:
            analysis = {
                'ticker': ticker,
                'macro': macro_report,
                'technical': next((s for s in trade_signals if s.get('ticker') == ticker), {}),
                'fundamental': next((c for c in company_reports if c.get('ticker') == ticker), {}),
                'risk': next((r for r in risk_assessments if r.get('ticker') == ticker), {})
            }
            combined.append(analysis)
        
        return combined
    
    def _generate_final_recommendations(
        self,
        combined_analysis: List[Dict],
        user_prefs: Dict,
        conflicts: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to generate final recommendations considering all inputs.
        
        Returns:
            List of final recommendations
        """
        recommendations = []
        
        # Get conversation history for context
        history = self.memory.load_memory_variables({})
        past_context = self._format_memory_context(history)
        
        for analysis in combined_analysis:
            ticker = analysis['ticker']
            
            system_prompt = """You are the Chief Investment Officer making final trading decisions.
Review all analyst reports and generate a final recommendation.

Consider:
1. Alignment between technical, fundamental, and macro analysis
2. Risk-adjusted returns
3. User preferences and risk tolerance
4. Any conflicts or concerns
5. Past recommendations and their outcomes

Output in JSON format with clear justification."""

            user_message = f"""Make final recommendation for {ticker}

MACRO ENVIRONMENT:
- Sentiment: {analysis['macro'].get('macro_sentiment', 'neutral')}
- Risk Level: {analysis['macro'].get('risk_level', 'medium')}
- Key Factors: {', '.join(analysis['macro'].get('key_factors', [])[:3])}

TECHNICAL ANALYSIS:
- Signal: {analysis['technical'].get('signal', 'N/A')} ({analysis['technical'].get('strength', 'N/A')} strength)
- Confidence: {analysis['technical'].get('confidence', 'N/A')}%
- Entry: ${analysis['technical'].get('entry_price', 0):.2f}
- Target: ${analysis['technical'].get('target_price', 0):.2f}
- Stop Loss: ${analysis['technical'].get('stop_loss', 0):.2f}
- Risk/Reward: {analysis['technical'].get('risk_reward_ratio', 'N/A')}
- Reasons: {', '.join(analysis['technical'].get('technical_reasons', [])[:3])}

FUNDAMENTAL ANALYSIS:
- Health Score: {analysis['fundamental'].get('health_score', 'N/A')}/100
- Valuation: {analysis['fundamental'].get('valuation', 'N/A')}
- Recommendation: {analysis['fundamental'].get('recommendation', 'N/A')}
- Strengths: {', '.join(analysis['fundamental'].get('strengths', [])[:2])}
- Weaknesses: {', '.join(analysis['fundamental'].get('weaknesses', [])[:2])}

RISK ASSESSMENT:
- Risk Score: {analysis['risk'].get('risk_score', 'N/A')}/10
- Category: {analysis['risk'].get('risk_category', 'N/A')}
- VaR 95%: ${analysis['risk'].get('var_95', 0):.2f}
- Max Position: {analysis['risk'].get('max_position_size', {}).get('recommended_position_pct', 'N/A')}%
- Probability of Profit: {analysis['risk'].get('scenario_analysis', {}).get('probabilities', {}).get('profit', 'N/A')}%

USER PREFERENCES:
- Risk Profile: {user_prefs.get('risk_profile', 'moderate')}
- Time Horizon: {user_prefs.get('time_horizon', 'medium-term')}

DETECTED CONFLICTS:
{json.dumps([c for c in conflicts if c.get('ticker') == ticker], indent=2)}

PAST CONTEXT:
{past_context}

Provide final recommendation in JSON:
{{
    "action": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence": 0-100,
    "justification": "Clear 2-3 sentence explanation",
    "expected_roi": float (percentage),
    "time_horizon": "short-term/medium-term/long-term",
    "consensus_score": 0-100 (agreement between agents),
    "key_reasons": ["reason1", "reason2", "reason3"],
    "warnings": ["warning1", ...],
    "recommended_entry": float,
    "recommended_target": float,
    "recommended_stop": float,
    "position_sizing": float (% of portfolio)
}}"""

            response = self.call_llm(system_prompt, user_message, json_mode=True)
            recommendation = self.parse_json_response(response)
            
            # Add metadata
            recommendation['ticker'] = ticker
            recommendation['timestamp'] = datetime.now().isoformat()
            recommendation['analyzed_by'] = ['macro', 'quant', 'fundamental', 'risk']
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _rank_recommendations(
        self,
        recommendations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Rank recommendations by confidence and expected ROI.
        
        Returns:
            Sorted list of recommendations
        """
        # Calculate composite score
        for rec in recommendations:
            confidence = rec.get('confidence', 0)
            roi = rec.get('expected_roi', 0)
            consensus = rec.get('consensus_score', 0)
            
            # Weighted score
            composite_score = (confidence * 0.4) + (roi * 0.3) + (consensus * 0.3)
            rec['composite_score'] = round(composite_score, 2)
        
        # Sort by composite score
        sorted_recs = sorted(
            recommendations,
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )
        
        # Add priority ranking
        for i, rec in enumerate(sorted_recs, 1):
            rec['priority'] = i
        
        return sorted_recs
    
    def _generate_portfolio_allocation(
        self,
        recommendations: List[Dict],
        risk_assessments: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate portfolio allocation strategy.
        
        Returns:
            Allocation recommendations
        """
        total_allocation = 0
        allocations = []
        
        # Only allocate to BUY recommendations
        buy_recs = [r for r in recommendations if r.get('action') in ['BUY', 'STRONG_BUY']]
        
        for rec in buy_recs[:5]:  # Top 5 opportunities
            ticker = rec.get('ticker')
            risk_assessment = next((r for r in risk_assessments if r.get('ticker') == ticker), {})
            
            position_size = risk_assessment.get('max_position_size', {}).get('recommended_position_pct', 2.0)
            
            allocations.append({
                'ticker': ticker,
                'allocation_pct': position_size,
                'action': rec.get('action'),
                'confidence': rec.get('confidence')
            })
            
            total_allocation += position_size
        
        return {
            'allocations': allocations,
            'total_allocated_pct': round(total_allocation, 2),
            'cash_reserve_pct': round(100 - total_allocation, 2),
            'diversification': len(allocations)
        }
    
    def _update_memory(self, inputs: Dict, recommendations: List[Dict]):
        """Store analysis in conversation memory."""
        summary = f"Analyzed {len(recommendations)} stocks. "
        summary += f"Top recommendation: {recommendations[0].get('ticker') if recommendations else 'None'} "
        summary += f"({recommendations[0].get('action') if recommendations else 'N/A'})"
        
        self.memory.save_context(
            {"input": f"Analysis request: {json.dumps(inputs.get('tickers', []))}"},
            {"output": summary}
        )
    
    def _format_memory_context(self, history: Dict) -> str:
        """Format memory for LLM context."""
        messages = history.get('chat_history', [])
        if not messages:
            return "No previous analyses."
        
        context = "Recent analyses:\n"
        for msg in messages[-3:]:  # Last 3 interactions
            if isinstance(msg, HumanMessage):
                context += f"- Query: {msg.content[:100]}...\n"
            elif isinstance(msg, AIMessage):
                context += f"- Result: {msg.content[:100]}...\n"
        
        return context


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = StrategyEvaluatorAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    async def test():
        result = await agent.analyze({
            "macro_report": {
                "macro_sentiment": "bullish",
                "risk_level": "medium",
                "key_factors": ["Fed policy", "Tech sector strength"]
            },
            "trade_signals": [{
                "ticker": "AAPL",
                "signal": "BUY",
                "strength": "strong",
                "confidence": 85,
                "entry_price": 175.0,
                "target_price": 185.0,
                "stop_loss": 170.0,
                "risk_reward_ratio": 2.0,
                "technical_reasons": ["Strong uptrend", "RSI in buy zone"]
            }],
            "company_reports": [{
                "ticker": "AAPL",
                "health_score": 85,
                "valuation": "fair",
                "recommendation": "buy",
                "strengths": ["Strong financials", "Innovation"],
                "weaknesses": ["High valuation"]
            }],
            "risk_assessments": [{
                "ticker": "AAPL",
                "risk_score": 4,
                "risk_category": "medium",
                "var_95": 5.25,
                "max_position_size": {"recommended_position_pct": 3.5},
                "scenario_analysis": {"probabilities": {"profit": 72}}
            }],
            "user_preferences": {
                "risk_profile": "moderate",
                "time_horizon": "medium-term"
            }
        })
        
        print("Strategy Evaluation Result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())