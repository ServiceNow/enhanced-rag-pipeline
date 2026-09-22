from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
from openai import AzureOpenAI
from openai._exceptions import RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def create_azure_client() -> AzureOpenAI:
    """Create Azure OpenAI client using environment variables"""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


class AzureOpenAIModel:
    def __init__(
        self,
        model_name_path: str,
        max_tokens: int = 1000,
        temperature: float = 0.5,
        enable_guided_decoding: bool = False,
        response_format: Dict = {"type": "text"},
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name_path
        self.model = create_azure_client()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_format = response_format
        self.enable_guided_decoding = enable_guided_decoding
        self.seed = seed

    def preprocess_input(self, input: str) -> List[Dict[str, Any]]:
        """Convert string input to Azure OpenAI message format"""
        messages = [{"role": "user", "content": input}]
        return messages

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        stop=stop_after_attempt(3),
    )
    def run(
        self,
        input: str,
        schema_template: Optional[Union[str, Dict, BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Run Azure OpenAI model"""
        messages = self.preprocess_input(input)
        
        # Build kwargs once; only include seed when set so behavior is unchanged for unseeded runs.
        common_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            common_kwargs["seed"] = self.seed

        try:
            if schema_template is not None and self.enable_guided_decoding:
                prediction = self.model.beta.chat.completions.parse(
                    response_format=schema_template,
                    **common_kwargs,
                )
                return prediction.choices[0].message.parsed.model_dump_json()
            else:
                prediction = self.model.chat.completions.create(
                    response_format=self.response_format,
                    **common_kwargs,
                )
                return prediction.choices[0].message.content

        except Exception as e:
            logging.error(f"Azure OpenAI error: {e}")
            raise e
