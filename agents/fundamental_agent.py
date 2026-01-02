# agents/fundamental_agent.py
"""
Fundamental Agent: Evaluates financial health using fundamental analysis.
Includes vector memory for storing past company analysis.
"""

from typing import Dict, Any, List, Optional
import finnhub
from datetime import datetime
from agents.base_agent import BaseAgent
from loguru import logger
import chromadb
from chromadb.utils import embedding_functions
import json
import os


class FundamentalAgent(BaseAgent):
    """Performs fundamental analysis on companies with persistent memory."""
    
    def __init__(
        self,
        openai_api_key: str,
        finnhub_api_key: str,
        model: str = "gpt-4-turbo-preview",
        chroma_persist_dir: str = "./data/chroma_db"
    ):
        super().__init__("FundamentalAgent", openai_api_key, model)
        
        # Finnhub client
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        
        # Create chroma directory if not exists
        os.makedirs(chroma_persist_dir, exist_ok=True)
        
        # Initialize vector memory
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        
        # OpenAI embedding function
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="company_fundamental_analysis",
            embedding_function=self.embedding_fn,
            metadata={"description": "Historical fundamental analysis"}
        )
        
        logger.info("Initialized Fundamental Agent with vector memory")
    
    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fundamental analysis on companies."""
        try:
            logger.info(f"Starting fundamental analysis for: {inputs.get('tickers', [])}")
            
            tickers = inputs.get("tickers", [])
            use_memory = inputs.get("use_memory", True)
            
            company_reports = []
            
            for ticker in tickers:
                logger.info(f"Analyzing {ticker}")
                
                # Check memory
                past_analysis = None
                if use_memory:
                    past_analysis = self._retrieve_from_memory(ticker)
                
                # Fetch financial data
                financial_data = self._fetch_financial_data(ticker)
                
                if not financial_data:
                    logger.warning(f"Could not fetch data for {ticker}")
                    continue
                
                # Perform analysis
                report = self._perform_fundamental_analysis(
                    ticker, financial_data, past_analysis
                )
                
                # Store in memory
                self._store_in_memory(ticker, report)
                
                company_reports.append(report)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "company_reports": company_reports
            }
            
            self.log_analysis(inputs, result)
            return self.create_success_response(result)
            
        except Exception as e:
            return self.handle_error(e, "fundamental analysis")
    
    def _fetch_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch financial data from Finnhub API."""
        try:
            data = {}
            
            # Company profile
            profile = self.finnhub_client.company_profile2(symbol=ticker)
            data['profile'] = profile
            
            # Financial metrics
            metrics = self.finnhub_client.company_basic_financials(ticker, 'all')
            data['metrics'] = metrics.get('metric', {})
            
            logger.info(f"Fetched financial data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            return {}
    
    def _retrieve_from_memory(self, ticker: str, limit: int = 3) -> Optional[List[Dict]]:
        """Retrieve past analyses from vector memory."""
        try:
            query_text = f"fundamental analysis of {ticker}"
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={"ticker": ticker}
            )
            
            if results['documents'] and results['documents'][0]:
                logger.info(f"Retrieved {len(results['documents'][0])} past analyses for {ticker}")
                
                past_analyses = []
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    past_analyses.append({
                        'analysis': doc,
                        'metadata': metadata
                    })
                
                return past_analyses
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from memory: {str(e)}")
            return None
    
    def _store_in_memory(self, ticker: str, report: Dict[str, Any]):
        """Store analysis in vector memory."""
        try:
            doc_text = f"""
Fundamental Analysis of {ticker}
Date: {report.get('timestamp', '')}
Health Score: {report.get('health_score', 0)}
Valuation: {report.get('valuation', '')}
Recommendation: {report.get('recommendation', '')}

Key Metrics:
{json.dumps(report.get('financial_metrics', {}), indent=2)}

Strengths: {', '.join(report.get('strengths', []))}
Weaknesses: {', '.join(report.get('weaknesses', []))}
"""
            
            doc_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.collection.add(
                documents=[doc_text],
                metadatas=[{
                    "ticker": ticker,
                    "timestamp": report.get('timestamp', ''),
                    "health_score": report.get('health_score', 0),
                    "valuation": report.get('valuation', '')
                }],
                ids=[doc_id]
            )
            
            logger.info(f"Stored analysis for {ticker} in memory")
            
        except Exception as e:
            logger.error(f"Error storing in memory: {str(e)}")
    
    def _perform_fundamental_analysis(
        self,
        ticker: str,
        financial_data: Dict[str, Any],
        past_analysis: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Perform fundamental analysis using LLM."""
        metrics = financial_data.get('metrics', {})
        profile = financial_data.get('profile', {})
        
        past_context = ""
        if past_analysis:
            past_context = "\n\nPast Analyses:\n"
            for pa in past_analysis[:2]:
                past_context += f"- {pa['metadata'].get('timestamp', 'Unknown')}: "
                past_context += f"Health {pa['metadata'].get('health_score', 'N/A')}, "
                past_context += f"Valuation: {pa['metadata'].get('valuation', 'N/A')}\n"
        
        system_prompt = """You are a senior fundamental analyst at an investment bank.
Analyze companies based on financial metrics and market position.

Output JSON with:
- health_score: 0-100
- valuation: "undervalued"/"fair"/"overvalued"
- financial_metrics: {"profitability": "good", "liquidity": "excellent", ...}
- strengths: list of top 3 strengths
- weaknesses: list of top 3 weaknesses
- recommendation: "strong_buy"/"buy"/"hold"/"sell"/"strong_sell"
- summary: brief 2-3 sentence summary"""

        user_message = f"""Analyze {ticker} ({profile.get('name', 'Unknown')})

Company Profile:
- Industry: {profile.get('finnhubIndustry', 'N/A')}
- Market Cap: ${profile.get('marketCapitalization', 0)}M
- Country: {profile.get('country', 'N/A')}

Key Financial Metrics:
- P/E Ratio: {metrics.get('peNormalizedAnnual', 'N/A')}
- Revenue Growth: {metrics.get('revenueGrowthTTMYoy', 'N/A')}%
- Profit Margin: {metrics.get('netProfitMarginTTM', 'N/A')}%
- ROE: {metrics.get('roeTTM', 'N/A')}%
- Current Ratio: {metrics.get('currentRatioAnnual', 'N/A')}
- Debt/Equity: {metrics.get('totalDebt/totalEquityAnnual', 'N/A')}
{past_context}

Provide analysis in JSON format."""

        response = self.call_llm(system_prompt, user_message, json_mode=True)
        analysis = self.parse_json_response(response)
        
        analysis['ticker'] = ticker
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['raw_metrics'] = metrics
        
        return analysis