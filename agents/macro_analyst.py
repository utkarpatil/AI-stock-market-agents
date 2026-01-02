# agents/macro_analyst.py
"""
Macro Analyst Agent: Analyzes global events, economic news, and policy decisions.
Sends macro environment reports to other agents.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from agents.base_agent import BaseAgent
from loguru import logger


class MacroAnalystAgent(BaseAgent):
    """Analyzes macroeconomic environment using news and economic data."""
    
    def __init__(
        self,
        openai_api_key: str,
        newsapi_key: str,
        model: str = "gpt-4-turbo-preview"
    ):
        super().__init__("MacroAnalyst", openai_api_key, model)
        self.news_client = NewsApiClient(api_key=newsapi_key)
        
        # Key topics to monitor
        self.topics = [
            "federal reserve interest rates",
            "inflation data",
            "GDP growth",
            "unemployment rate",
            "stock market trends",
            "economic outlook"
        ]
    
    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform macro analysis.
        
        Args:
            inputs: {
                "sectors": ["tech", "finance", ...],  # Optional
                "lookback_days": 7,  # How far back to look for news
                "include_global": True
            }
            
        Returns:
            {
                "macro_sentiment": "bullish/bearish/neutral",
                "risk_level": "low/medium/high",
                "key_factors": [...],
                "sector_outlook": {...},
                "summary": "..."
            }
        """
        try:
            logger.info(f"Starting macro analysis with inputs: {inputs}")
            
            # 1. Gather news data
            lookback_days = inputs.get("lookback_days", 7)
            news_data = self._fetch_news(lookback_days)
            
            # 2. Analyze with LLM
            analysis = self._analyze_macro_environment(news_data, inputs)
            
            # 3. Generate report for other agents
            report = self._generate_report(analysis)
            
            self.log_analysis(inputs, report)
            return self.create_success_response(report)
            
        except Exception as e:
            return self.handle_error(e, "macro analysis")
    
    def _fetch_news(self, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch relevant economic and market news.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            List of news articles
        """
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        all_articles = []
        
        for topic in self.topics:
            try:
                logger.info(f"Fetching news for topic: {topic}")
                
                response = self.news_client.get_everything(
                    q=topic,
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=5  # Limit to 5 per topic to avoid rate limits
                )
                
                articles = response.get('articles', [])
                
                for article in articles:
                    all_articles.append({
                        'topic': topic,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'url': article.get('url', '')
                    })
                
            except Exception as e:
                logger.warning(f"Failed to fetch news for {topic}: {str(e)}")
                continue
        
        logger.info(f"Fetched {len(all_articles)} total articles")
        return all_articles
    
    def _analyze_macro_environment(
        self,
        news_data: List[Dict[str, Any]],
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze macro environment from news.
        
        Args:
            news_data: List of news articles
            inputs: User inputs
            
        Returns:
            Structured analysis
        """
        # Prepare news summary for LLM (limit to avoid token limits)
        news_summary = "\n\n".join([
            f"Topic: {article['topic']}\n"
            f"Title: {article['title']}\n"
            f"Description: {article['description']}\n"
            f"Source: {article['source']}\n"
            f"Date: {article['published_at']}"
            for article in news_data[:20]  # Limit to 20 articles
        ])
        
        system_prompt = """You are a senior macroeconomic analyst for a hedge fund.
Analyze the provided news and economic data to determine the overall market environment.

Your analysis must include:
1. Overall market sentiment (bullish, bearish, or neutral)
2. Risk level assessment (low, medium, or high)
3. Key factors driving the market (top 5)
4. Sector-specific outlooks
5. Potential risks and opportunities

Be specific, data-driven, and consider both short-term and medium-term implications.
Output your analysis in JSON format."""

        user_message = f"""Analyze the following economic and market news from the past week:

{news_summary}

Additional context:
- Current Date: {datetime.now().strftime('%Y-%m-%d')}
- Focus sectors: {inputs.get('sectors', 'all sectors')}
- Include global factors: {inputs.get('include_global', True)}

Provide a comprehensive macro analysis in JSON format with this structure:
{{
    "sentiment": "bullish/bearish/neutral",
    "sentiment_score": 0-100,
    "risk_level": "low/medium/high",
    "key_factors": ["factor1", "factor2", "factor3", "factor4", "factor5"],
    "sector_outlook": {{"technology": "positive", "finance": "neutral", "healthcare": "positive"}},
    "opportunities": ["opportunity1", "opportunity2"],
    "risks": ["risk1", "risk2"],
    "summary": "Brief 2-3 sentence summary of the current market environment"
}}"""

        response = self.call_llm(system_prompt, user_message, json_mode=True)
        analysis = self.parse_json_response(response)
        
        return analysis
    
    def _generate_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate formatted report for downstream agents.
        
        Args:
            analysis: Raw analysis from LLM
            
        Returns:
            Formatted report
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "macro_sentiment": analysis.get("sentiment", "neutral"),
            "sentiment_score": analysis.get("sentiment_score", 50),
            "risk_level": analysis.get("risk_level", "medium"),
            "key_factors": analysis.get("key_factors", []),
            "sector_outlook": analysis.get("sector_outlook", {}),
            "opportunities": analysis.get("opportunities", []),
            "risks": analysis.get("risks", []),
            "summary": analysis.get("summary", ""),
            "recommendations_for_traders": {
                "suggested_bias": analysis.get("sentiment", "neutral"),
                "caution_level": analysis.get("risk_level", "medium"),
                "focus_sectors": self._extract_positive_sectors(
                    analysis.get("sector_outlook", {})
                )
            }
        }
    
    def _extract_positive_sectors(self, sector_outlook: Dict[str, str]) -> List[str]:
        """Extract sectors with positive outlook."""
        positive_keywords = ["positive", "bullish", "strong", "favorable"]
        positive_sectors = [
            sector for sector, outlook in sector_outlook.items()
            if any(keyword in outlook.lower() for keyword in positive_keywords)
        ]
        return positive_sectors


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test():
        agent = MacroAnalystAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            newsapi_key=os.getenv("NEWSAPI_KEY")
        )
        
        result = await agent.analyze({
            "sectors": ["technology", "finance"],
            "lookback_days": 7,
            "include_global": True
        })
        
        print("Macro Analysis Result:")
        import json
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())