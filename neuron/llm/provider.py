from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import openai
import anthropic
from neuron.config import settings

class LLMProvider(ABC):
    @abstractmethod
    def completion(self, prompt: str, system_prompt: str = "") -> str:
        pass

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def completion(self, prompt: str, system_prompt: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def completion(self, prompt: str, system_prompt: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

def get_llm_provider(provider_type: str = None) -> LLMProvider:
    p_type = provider_type or settings.llm_provider
    
    if p_type == "openai":
        return OpenAICompatibleProvider(
            base_url=None, # Use default OpenAI URL
            api_key=settings.openai_api_key,
            model=settings.default_model
        )
    elif p_type == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.abstraction_model
        )
    elif p_type == "groq":
        return OpenAICompatibleProvider(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model=settings.local_llm_model
        )
    elif p_type == "ollama":
        return OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=settings.local_llm_model
        )
    else:
        raise ValueError(f"Unknown LLM provider: {p_type}")
