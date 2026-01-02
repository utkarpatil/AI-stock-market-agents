# agents/risk_manager.py
"""
Risk Manager Agent: Analyzes risk of every suggested trade.
Includes user risk profile memory and Monte Carlo simulations.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.stats import norm
import json
import os
from datetime import datetime
from agents.base_agent import BaseAgent
from loguru import logger


class RiskManagerAgent(BaseAgent):
    """Analyzes and scores risk for trading recommendations."""
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4-turbo-preview",
        risk_profiles_dir: str = "./data/risk_profiles"
    ):
        super().__init__("RiskManager", openai_api_key, model)
        self.risk_profiles_dir = risk_profiles_dir
        
        # Create directory
        os.makedirs(risk_profiles_dir, exist_ok=True)
        
        logger.info("Initialized Risk Manager with profile memory")
    
    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk for trade recommendations."""
        try:
            logger.info("Starting risk analysis")
            
            user_id = inputs.get("user_id", "default")
            trade_signals = inputs.get("trade_signals", [])
            company_reports = inputs.get("company_reports", [])
            macro_report = inputs.get("macro_report", {})
            simulate = inputs.get("simulate_scenarios", True)
            
            # Load user risk profile
            risk_profile = self._load_risk_profile(user_id)
            
            risk_assessments = []
            
            for signal in trade_signals:
                ticker = signal.get('ticker', '')
                logger.info(f"Assessing risk for {ticker}")
                
                # Find corresponding fundamental report
                company_report = next(
                    (r for r in company_reports if r.get('ticker') == ticker),
                    {}
                )
                
                # Perform risk assessment
                assessment = self._assess_trade_risk(
                    signal, company_report, macro_report, risk_profile
                )
                
                # Run Monte Carlo simulation
                if simulate:
                    scenarios = self._monte_carlo_simulation(signal, company_report)
                    assessment['scenario_analysis'] = scenarios
                
                # Calculate position sizing
                position_size = self._calculate_position_size(assessment, risk_profile)
                assessment['max_position_size'] = position_size
                
                risk_assessments.append(assessment)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "user_risk_profile": risk_profile['profile_type'],
                "risk_assessments": risk_assessments
            }
            
            self.log_analysis(inputs, result)
            return self.create_success_response(result)
            
        except Exception as e:
            return self.handle_error(e, "risk analysis")
    
    def _load_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user risk profile from memory."""
        profile_path = os.path.join(self.risk_profiles_dir, f"{user_id}.json")
        
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                logger.info(f"Loaded risk profile for {user_id}")
                return profile
            else:
                logger.warning(f"No profile found for {user_id}, using default")
                return self._get_default_profile()
                
        except Exception as e:
            logger.error(f"Error loading profile: {str(e)}")
            return self._get_default_profile()
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Return default moderate risk profile."""
        return {
            "user_id": "default",
            "profile_type": "moderate",
            "max_portfolio_risk": 2.0,
            "max_single_loss": 1.0,
            "risk_tolerance": 5,
            "time_horizon": "medium-term"
        }
    
    def save_risk_profile(self, user_id: str, profile: Dict[str, Any]):
        """Save user risk profile to memory."""
        profile_path = os.path.join(self.risk_profiles_dir, f"{user_id}.json")
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            logger.info(f"Saved risk profile for {user_id}")
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
    
    def _assess_trade_risk(
        self,
        signal: Dict[str, Any],
        company_report: Dict[str, Any],
        macro_report: Dict[str, Any],
        risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive risk assessment for a trade."""
        ticker = signal.get('ticker', '')
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        target_price = signal.get('target_price', 0)
        
        # Calculate metrics
        potential_loss = abs(entry_price - stop_loss) / entry_price * 100
        potential_gain = abs(target_price - entry_price) / entry_price * 100
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        indicators = signal.get('indicators', {})
        atr = indicators.get('atr', 0)
        volatility = (atr / entry_price) * 100 if entry_price > 0 else 0
        
        system_prompt = """You are a risk management expert.
Analyze the trade and assign risk score 1-10 (1=lowest, 10=highest).

Output JSON:
{
    "risk_score": 1-10,
    "risk_category": "low/medium/high",
    "risk_factors": ["factor1", "factor2"],
    "mitigating_factors": ["factor1", "factor2"],
    "recommended_stop_loss": float,
    "confidence_in_risk_assessment": 0-100,
    "warnings": ["warning1"]
}"""

        user_message = f"""Assess risk for {ticker}

Signal: {signal.get('signal', '')} ({signal.get('strength', '')})
Entry: ${entry_price:.2f}, Target: ${target_price:.2f}, Stop: ${stop_loss:.2f}

Risk/Reward: {risk_reward:.2f}
Potential Loss: {potential_loss:.2f}%
Potential Gain: {potential_gain:.2f}%
Volatility: {volatility:.2f}%

Technical: RSI {indicators.get('rsi', 'N/A')}, Trend {indicators.get('trend', 'N/A')}
Fundamental: Health {company_report.get('health_score', 'N/A')}, Valuation {company_report.get('valuation', 'N/A')}
Macro: {macro_report.get('macro_sentiment', 'neutral')} sentiment, {macro_report.get('risk_level', 'medium')} risk
User Profile: {risk_profile.get('profile_type', 'moderate')}

Provide risk assessment in JSON."""

        response = self.call_llm(system_prompt, user_message, json_mode=True)
        assessment = self.parse_json_response(response)
        
        assessment['ticker'] = ticker
        assessment['potential_loss_pct'] = potential_loss
        assessment['potential_gain_pct'] = potential_gain
        assessment['risk_reward_ratio'] = risk_reward
        assessment['volatility'] = volatility
        assessment['timestamp'] = datetime.now().isoformat()
        
        # Calculate VaR
        var_95 = self._calculate_var(entry_price, volatility)
        assessment['var_95'] = var_95
        
        return assessment
    
    def _calculate_var(
        self,
        entry_price: float,
        volatility: float,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        daily_vol = volatility / np.sqrt(252)
        z_score = norm.ppf(confidence)
        var = entry_price * daily_vol * z_score
        return abs(var)
    
    def _monte_carlo_simulation(
        self,
        signal: Dict[str, Any],
        company_report: Dict[str, Any],
        n_simulations: int = 1000,
        time_horizon: int = 30
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        try:
            entry_price = signal.get('entry_price', 0)
            target_price = signal.get('target_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            
            indicators = signal.get('indicators', {})
            atr = indicators.get('atr', 0)
            volatility = (atr / entry_price) if entry_price > 0 else 0.02
            
            health_score = company_report.get('health_score', 50)
            drift = (health_score - 50) / 1000
            
            final_prices = []
            
            for _ in range(n_simulations):
                price = entry_price
                for day in range(time_horizon):
                    random_shock = np.random.normal(0, 1)
                    daily_return = drift + volatility * random_shock / np.sqrt(252)
                    price = price * (1 + daily_return)
                final_prices.append(price)
            
            final_prices = np.array(final_prices)
            
            prob_profit = np.mean(final_prices > entry_price) * 100
            prob_target = np.mean(final_prices >= target_price) * 100
            prob_stop_loss = np.mean(final_prices <= stop_loss) * 100
            
            return {
                'probabilities': {
                    'profit': round(prob_profit, 2),
                    'reach_target': round(prob_target, 2),
                    'hit_stop_loss': round(prob_stop_loss, 2)
                },
                'price_distribution': {
                    'mean': round(np.mean(final_prices), 2),
                    'std': round(np.std(final_prices), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo failed: {str(e)}")
            return {'probabilities': {'profit': 50, 'reach_target': 30, 'hit_stop_loss': 20}}
    
    def _calculate_position_size(
        self,
        assessment: Dict[str, Any],
        risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate recommended position size."""
        risk_score = assessment.get('risk_score', 5)
        max_portfolio_risk = risk_profile.get('max_portfolio_risk', 2.0)
        potential_loss_pct = assessment.get('potential_loss_pct', 5.0)
        
        risk_multiplier = (11 - risk_score) / 10
        
        if potential_loss_pct > 0:
            position_pct = (max_portfolio_risk / potential_loss_pct) * risk_multiplier
        else:
            position_pct = max_portfolio_risk * risk_multiplier
        
        position_pct = min(position_pct, max_portfolio_risk * 2)
        position_pct = max(position_pct, 0.5)
        
        return {
            'recommended_position_pct': round(position_pct, 2),
            'max_position_pct': round(max_portfolio_risk * 2, 2)
        }