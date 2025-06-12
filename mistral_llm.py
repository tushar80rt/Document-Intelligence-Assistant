from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import os
from mistralai import Mistral
from pydantic import PrivateAttr
from dotenv import load_dotenv

load_dotenv("api.env")

class MistralLLM(LLM):
    _client: Mistral = PrivateAttr()

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.chat.complete(  
            model="mistral-large-latest",      
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content 

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": "mistral-large-latest"}
