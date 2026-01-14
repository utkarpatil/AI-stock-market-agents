"""
Strategy Evaluator Agent: Reviews all agents' advice and generates final recommendations.
Uses LangChain conversation memory for context.
"""

from typing import Dict, Any, List
from datetime import datetime
from agents.base_agent import BaseAgent
from loguru import logger
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import json


class StrategyEvaluatorAgent(BaseAgent):
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        super().__init__("StrategyEvaluator", openai_api_key, model)

        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )

        logger.info("Initialized Strategy Evaluator with conversation memory")

    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Starting strategy evaluation")

            macro_report = inputs.get("macro_report", {})
            trade_signals = inputs.get("trade_signals", [])
            company_reports = inputs.get("company_reports", [])
            risk_assessments = inputs.get("risk_assessments", [])
            user_prefs = inputs.get("user_preferences", {})

            conflicts = self._detect_conflicts(
                trade_signals, company_reports, risk_assessments
            )

            combined_analysis = self._combine_analyses(
                macro_report, trade_signals, company_reports, risk_assessments
            )

            recommendations = self._generate_final_recommendations(
                combined_analysis, user_prefs, conflicts
            )

            ranked = self._rank_recommendations(recommendations)

            allocation = self._generate_portfolio_allocation(
                ranked, risk_assessments
            )

            self._update_memory(inputs, ranked)

            result = {
                "timestamp": datetime.now().isoformat(),
                "final_recommendations": ranked,
                "portfolio_allocation": allocation,
                "conflicts_detected": conflicts,
                "total_opportunities": len(ranked),
                "high_confidence_count": len(
                    [r for r in ranked if r.get("confidence", 0) >= 80]
                ),
            }

            return self.create_success_response(result)

        except Exception as e:
            return self.handle_error(e, "strategy evaluation")

    # ------------------------------------------------------------------

    def _combine_analyses(self, macro, trade, fundamental, risk):
        combined = []
        tickers = {s.get("ticker") for s in trade}

        for ticker in tickers:
            combined.append({
                "ticker": ticker,
                "macro": macro,
                "technical": next((t for t in trade if t.get("ticker") == ticker), {}),
                "fundamental": next((f for f in fundamental if f.get("ticker") == ticker), {}),
                "risk": next((r for r in risk if r.get("ticker") == ticker), {}),
            })

        return combined

    # ------------------------------------------------------------------

    def _generate_final_recommendations(self, combined, user_prefs, conflicts):
        recommendations = []

        history = self.memory.load_memory_variables({})
        past_context = self._format_memory_context(history)

        for analysis in combined:
            ticker = analysis["ticker"]
            technical = analysis.get("technical", {})

            response = self.call_llm(
                "You are a professional portfolio manager. Respond only in JSON.",
                json.dumps({
                    "ticker": ticker,
                    "macro": analysis["macro"],
                    "technical": technical,
                    "fundamental": analysis["fundamental"],
                    "risk": analysis["risk"],
                    "user_preferences": user_prefs,
                    "conflicts": conflicts,
                    "past_context": past_context
                }),
                json_mode=True
            )

            rec = self.parse_json_response(response)

            # ------------------------------------------------------------
            # ✅ FINAL ENTRY / TARGET / STOP LOGIC (GUARANTEED)
            # ------------------------------------------------------------

            entry = rec.get("recommended_entry")
            target = rec.get("recommended_target")
            stop = rec.get("recommended_stop")

            # Fallback to technical analysis if LLM gives 0 / null
            if not isinstance(entry, (int, float)) or entry <= 0:
                entry = technical.get("entry_price")

            if not isinstance(target, (int, float)) or target <= 0:
                target = technical.get("target_price")

            if not isinstance(stop, (int, float)) or stop <= 0:
                stop = technical.get("stop_loss")

            rec["entry"] = entry
            rec["target"] = target
            rec["stop_loss"] = stop

            # Expected ROI (safe, never crashes)
            if isinstance(entry, (int, float)) and isinstance(target, (int, float)) and entry > 0:
                rec["expected_roi"] = round(((target - entry) / entry) * 100, 2)
            else:
                rec["expected_roi"] = 0

            # Metadata
            rec["ticker"] = ticker
            rec["timestamp"] = datetime.now().isoformat()
            rec["analyzed_by"] = ["macro", "quant", "fundamental", "risk"]

            recommendations.append(rec)

        return recommendations

    # ------------------------------------------------------------------

    def _rank_recommendations(self, recommendations):
        for r in recommendations:
            confidence = r.get("confidence", 0)
            roi = r.get("expected_roi", 0)
            consensus = r.get("consensus_score", 0)

            r["composite_score"] = round(
                confidence * 0.4 + roi * 0.3 + consensus * 0.3, 2
            )

        recommendations.sort(
            key=lambda x: x.get("composite_score", 0), reverse=True
        )

        for i, r in enumerate(recommendations, 1):
            r["priority"] = i

        return recommendations

    # ------------------------------------------------------------------

    def _generate_portfolio_allocation(self, recs, risks):
        allocations = []
        total = 0

        for r in recs:
            if r.get("action") not in ["BUY", "STRONG_BUY"]:
                continue

            risk = next((x for x in risks if x.get("ticker") == r["ticker"]), {})
            pct = risk.get("max_position_size", {}).get("recommended_position_pct", 2.0)

            allocations.append({
                "ticker": r["ticker"],
                "allocation_pct": pct,
                "confidence": r.get("confidence", 0)
            })

            total += pct

        return {
            "allocations": allocations,
            "total_allocated_pct": round(total, 2),
            "cash_reserve_pct": round(100 - total, 2),
        }

    # ------------------------------------------------------------------

    def _detect_conflicts(self, *_):
        return []

    def _update_memory(self, inputs, recommendations):
        self.memory.save_context(
            {"input": "analysis"},
            {"output": f"{len(recommendations)} recommendations generated"}
        )

    def _format_memory_context(self, history):
        msgs = history.get("chat_history", [])
        if not msgs:
            return "No prior context"

        out = []
        for m in msgs[-3:]:
            if isinstance(m, HumanMessage):
                out.append(m.content[:100])
            elif isinstance(m, AIMessage):
                out.append(m.content[:100])

        return "\n".join(out)
