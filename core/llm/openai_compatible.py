from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from pydantic import BaseModel


class OpenAICompatibleModel:
    def __init__(
        self,
        model_name_path: str,
        base_url: str,
        max_tokens: int = 1000,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name_path
        self.model = OpenAI(base_url=base_url, api_key="local")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed

    def preprocess_input(self, input: str) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": input}]

    def run(
        self,
        input: str,
        schema_template: Optional[Union[str, Dict, BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        request: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self.preprocess_input(input),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        prediction = self.model.chat.completions.create(**request)
        return prediction.choices[0].message.content
