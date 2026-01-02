# agents/quant_agent.py
"""
Quant Agent: Performs technical analysis using price charts, indicators, and patterns.
Uses yfinance for data and talib for technical indicators.
"""

from typing import Dict, Any, List
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from loguru import logger


class QuantAgent(BaseAgent):
    """
    Performs quantitative and technical analysis on stocks.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        super().__init__("QuantAgent", openai_api_key, model)
        
        # Technical indicators to calculate
        self.indicators = [
            'RSI', 'MACD', 'BB', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26', 'ATR', 'ADX', 'OBV', 'STOCH'
        ]
    
    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform technical analysis on stocks.
        
        Args:
            inputs: {
                "tickers": ["AAPL", "MSFT", ...],
                "period": "3mo",  # yfinance period
                "macro_report": {...}  # From Macro Analyst
            }
            
        Returns:
            {
                "trade_signals": [
                    {
                        "ticker": "AAPL",
                        "signal": "BUY/SELL/HOLD",
                        "strength": "weak/medium/strong",
                        "entry_price": 175.00,
                        "target_price": 185.00,
                        "stop_loss": 170.00,
                        "technical_reasons": [...],
                        "indicators": {...}
                    }
                ]
            }
        """
        try:
            logger.info(f"Starting quant analysis for: {inputs.get('tickers', [])}")
            
            tickers = inputs.get("tickers", [])
            period = inputs.get("period", "3mo")
            macro_report = inputs.get("macro_report", {})
            
            trade_signals = []
            
            for ticker in tickers:
                logger.info(f"Analyzing {ticker}")
                
                # 1. Fetch market data
                data = self._fetch_market_data(ticker, period)
                
                if data is None or len(data) < 50:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue
                
                # 2. Calculate technical indicators
                indicators = self._calculate_indicators(data)
                
                # 3. Detect patterns
                patterns = self._detect_patterns(data, indicators)
                
                # 4. Generate signal with LLM
                signal = self._generate_signal(
                    ticker, data, indicators, patterns, macro_report
                )
                
                trade_signals.append(signal)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "trade_signals": trade_signals,
                "analysis_period": period,
                "macro_bias": macro_report.get("macro_sentiment", "neutral")
            }
            
            self.log_analysis(inputs, result)
            return self.create_success_response(result)
            
        except Exception as e:
            return self.handle_error(e, "quant analysis")
    
    def _fetch_market_data(
        self,
        ticker: str,
        period: str = "3mo"
    ) -> pd.DataFrame:
        """
        Fetch historical price data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1mo, 3mo, 6mo, 1y, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            logger.info(f"Fetched {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:

        data = data.copy()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna()

        close = data["Close"].astype(np.float64).values
        high = data["High"].astype(np.float64).values
        low = data["Low"].astype(np.float64).values
        volume = data["Volume"].astype(np.float64).values

        MIN_BARS = 50
        if len(close) < MIN_BARS:
            logger.warning("Not enough data after cleaning for indicators")
            return {}

        indicators = {}  # ✅ MUST be before try

        try:
            indicators["rsi"] = talib.RSI(close, timeperiod=14)[-1]

            macd, signal, hist = talib.MACD(close, 12, 26, 9)
            indicators["macd"] = {
                "macd": macd[-1],
                "signal": signal[-1],
                "histogram": hist[-1],
            }

            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            indicators["bollinger_bands"] = {
                "upper": upper[-1],
                "middle": middle[-1],
                "lower": lower[-1],
                "price_position": (close[-1] - lower[-1]) / (upper[-1] - lower[-1]),
            }

            indicators["sma_20"] = talib.SMA(close, 20)[-1]
            indicators["sma_50"] = talib.SMA(close, 50)[-1]
            indicators["sma_200"] = talib.SMA(close, 200)[-1]
            indicators["ema_12"] = talib.EMA(close, 12)[-1]
            indicators["ema_26"] = talib.EMA(close, 26)[-1]

            indicators["atr"] = talib.ATR(high, low, close, 14)[-1]
            indicators["adx"] = talib.ADX(high, low, close, 14)[-1]

            slowk, slowd = talib.STOCH(high, low, close)
            indicators["stochastic"] = {"k": slowk[-1], "d": slowd[-1]}

            indicators["obv"] = talib.OBV(close, volume)[-1]

            indicators["current_price"] = close[-1]
            indicators["price_change_1d"] = ((close[-1] - close[-2]) / close[-2]) * 100
            indicators["price_change_5d"] = ((close[-1] - close[-6]) / close[-6]) * 100

            if len(close) >= 21:
                indicators["price_change_20d"] = ((close[-1] - close[-21]) / close[-21]) * 100
            else:
                indicators["price_change_20d"] = 0.0

            indicators["trend"] = self._determine_trend(indicators)

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}


    
    def _determine_trend(self, indicators: Dict[str, Any]) -> str:
        """
        Determine overall trend from indicators.
        
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        price = indicators.get('current_price', 0)
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        sma_200 = indicators.get('sma_200', 0)
        
        if price > sma_20 > sma_50 > sma_200:
            return 'uptrend'
        elif price < sma_20 < sma_50 < sma_200:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _detect_patterns(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> List[str]:
        """
        Detect candlestick patterns and chart patterns.
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        open_prices = data['Open'].values
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        try:
            # Candlestick patterns
            if talib.CDLDOJI(open_prices, high, low, close)[-1] != 0:
                patterns.append("Doji")
            
            if talib.CDLENGULFING(open_prices, high, low, close)[-1] != 0:
                patterns.append("Engulfing")
            
            if talib.CDLHAMMER(open_prices, high, low, close)[-1] != 0:
                patterns.append("Hammer")
            
            if talib.CDLMORNINGSTAR(open_prices, high, low, close)[-1] != 0:
                patterns.append("Morning Star")
            
            if talib.CDLEVENINGSTAR(open_prices, high, low, close)[-1] != 0:
                patterns.append("Evening Star")
            
            # Chart patterns based on indicators
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                patterns.append("Overbought (RSI)")
            elif rsi < 30:
                patterns.append("Oversold (RSI)")
            
            # MACD crossover
            macd = indicators.get('macd', {})
            if macd.get('macd', 0) > macd.get('signal', 0):
                patterns.append("MACD Bullish Crossover")
            elif macd.get('macd', 0) < macd.get('signal', 0):
                patterns.append("MACD Bearish Crossover")
            
            logger.info(f"Detected patterns: {patterns}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return patterns
    
    def _generate_signal(
        self,
        ticker: str,
        data: pd.DataFrame,
        indicators: Dict[str, Any],
        patterns: List[str],
        macro_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signal using LLM analysis.
        
        Returns:
            Trading signal with entry, target, stop loss
        """
        current_price = indicators.get('current_price', 0)
        atr = indicators.get('atr', 0)
        
        system_prompt = """You are an expert quantitative trader specializing in technical analysis.
Analyze the provided technical indicators and patterns to generate a trading signal.

Output format (JSON):
{
    "signal": "BUY/SELL/HOLD",
    "strength": "weak/medium/strong",
    "confidence": 0-100,
    "entry_price": float,
    "target_price": float,
    "stop_loss": float,
    "technical_reasons": ["reason1", "reason2", ...],
    "risk_reward_ratio": float,
    "timeframe": "short-term/medium-term/long-term"
}"""

        user_message = f"""Analyze {ticker} and generate a trading signal.

Current Price: ${current_price:.2f}

Technical Indicators:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', {})}
- Trend: {indicators.get('trend', 'unknown')}
- ADX: {indicators.get('adx', 'N/A')}
- Stochastic: {indicators.get('stochastic', {})}
- Bollinger Band Position: {indicators.get('bollinger_bands', {}).get('price_position', 'N/A')}
- SMA 20/50/200: {indicators.get('sma_20', 'N/A'):.2f} / {indicators.get('sma_50', 'N/A'):.2f} / {indicators.get('sma_200', 'N/A'):.2f}
- ATR (volatility): ${atr:.2f}
- Recent Performance: 1D: {indicators.get('price_change_1d', 0):.2f}%, 5D: {indicators.get('price_change_5d', 0):.2f}%, 20D: {indicators.get('price_change_20d', 0):.2f}%

Detected Patterns:
{', '.join(patterns) if patterns else 'None'}

Macro Environment:
- Sentiment: {macro_report.get('macro_sentiment', 'neutral')}
- Risk Level: {macro_report.get('risk_level', 'medium')}

Calculate entry, target (use ATR for targets), and stop loss based on technical analysis and risk management principles."""

        response = self.call_llm(system_prompt, user_message, json_mode=True)
        signal_data = self.parse_json_response(response)
        
        # Add ticker and indicators to response
        signal_data['ticker'] = ticker
        signal_data['indicators'] = indicators
        signal_data['patterns'] = patterns
        signal_data['timestamp'] = datetime.now().isoformat()
        
        return signal_data


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = QuantAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    async def test():
        result = await agent.analyze({
            "tickers": ["AAPL", "MSFT"],
            "period": "3mo",
            "macro_report": {
                "macro_sentiment": "bullish",
                "risk_level": "medium"
            }
        })
        
        print("Quant Analysis Result:")
        import json
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())