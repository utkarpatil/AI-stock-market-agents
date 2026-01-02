# agents/base_agent.py
"""
Base Agent class that all specialized agents inherit from.
Provides common functionality like OpenAI client, logging, and error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from openai import OpenAI
from loguru import logger
import json
from datetime import datetime


class BaseAgent(ABC):
    """Abstract base class for all market analysis agents."""
    
    def __init__(
        self,
        name: str,
        openai_api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent identifier
            openai_api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Model temperature (0-1)
        """
        self.name = name
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        
        # Configure logging
        logger.add(
            f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
        logger.info(f"Initialized {name} agent with model {model}")
    
    @abstractmethod
    async def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method - must be implemented by each agent.
        
        Args:
            inputs: Dictionary containing analysis inputs
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        json_mode: bool = False
    ) -> str:
        """
        Call OpenAI LLM with given prompts.
        
        Args:
            system_prompt: System instructions
            user_message: User query
            json_mode: Whether to force JSON output
            
        Returns:
            LLM response as string
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format
            )
            
            content = response.choices[0].message.content
            logger.debug(f"{self.name} LLM response: {content[:200]}...")
            
            return content
            
        except Exception as e:
            logger.error(f"{self.name} LLM call failed: {str(e)}")
            raise
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM.
        
        Args:
            response: JSON string from LLM
            
        Returns:
            Parsed dictionary
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.debug(f"Response was: {response}")
            raise
    
    def log_analysis(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Log analysis inputs and outputs."""
        logger.info(f"{self.name} analysis completed")
        logger.debug(f"Inputs: {json.dumps(inputs, indent=2)}")
        logger.debug(f"Outputs: {json.dumps(outputs, indent=2)}")
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Standard error handling for agents.
        
        Args:
            error: The exception that occurred
            context: Additional context about where error occurred
            
        Returns:
            Error response dictionary
        """
        error_msg = f"{self.name} error in {context}: {str(error)}"
        logger.error(error_msg)
        
        return {
            "success": False,
            "error": error_msg,
            "agent": self.name,
            "timestamp": datetime.now().isoformat()
        }
    
    def create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create standardized success response.
        
        Args:
            data: Analysis results
            
        Returns:
            Formatted response dictionary
        """
        return {
            "success": True,
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }


class AgentMemory:
    """Simple memory interface for agents that need persistence."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.short_term = []  # Recent analyses
        self.max_short_term = 10
    
    def add_to_short_term(self, item: Dict[str, Any]):
        """Add item to short-term memory with size limit."""
        self.short_term.append(item)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)
    
    def get_recent_context(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n most recent items from memory."""
        return self.short_term[-n:]
    
    def clear(self):
        """Clear short-term memory."""
        self.short_term = []
        logger.info(f"Cleared memory for {self.agent_name}")


# Example usage and testing
if __name__ == "__main__":
    # This would be implemented by concrete agent classes
    pass