from typing import Any, Dict, List, Optional, Union

from pathlib import Path

import torch
import vllm
from transformers import AutoTokenizer
from core.llm.utils import get_compute_dtype, get_model_from_name
from utils.utils import LOGGER
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams


class VLLMModel:
    def __init__(
        self,
        model_name_path: str,
        max_tokens: int = 300,
        temperature: float = 0.5,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 1024,
        enable_guided_decoding: bool = True,
        **kwargs: Any,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.model_name_path = model_name_path
        self.model_name = model_name_path.split("/")[-1]
        model_name_path = get_model_from_name(model_name_path)

        if not Path(model_name_path).exists():
            LOGGER.info(f"Model {model_name_path} not found, will try to load directly from Hugging Face")

        # get device count
        device_count = torch.cuda.device_count()
        LOGGER.info(f"Available GPUs: {device_count}")
        compute_dtype = get_compute_dtype()

        self.enable_guided_decoding = enable_guided_decoding

        # Use V0 engine and tokenizer_mode='slow' to avoid TokenizersBackend compatibility issues
        import os
        os.environ["VLLM_USE_V1"] = "0" if device_count else "1"
        
        self.model = vllm.LLM(
            model=model_name_path,
            tensor_parallel_size=max(1, device_count),
            dtype=compute_dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
            tokenizer_mode="slow",  # Use slow tokenizer to avoid fast tokenizer compatibility issues
            **kwargs,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        
        # Get context size - use max_model_len as default
        self.context_size = max_model_len


    def preprocess_input(self, input: str) -> List[Dict]:
        messages = [{"role": "user", "content": input}]
        return messages

    def run(
        self,
        input: str,
        schema_template: Optional[Union[str, Dict, BaseModel]] = None,
        **kwargs: Any,
    ) -> Any:
        messages = self.preprocess_input(input)

        guided_decoding = None
        if schema_template is not None and self.enable_guided_decoding:
            guided_decoding = GuidedDecodingParams(json=schema_template.model_json_schema())

        # Use chat() method with slow tokenizer
        prediction = self.model.chat(
            messages,
            sampling_params=vllm.SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                guided_decoding=guided_decoding,
                **kwargs,
            ),
            use_tqdm=False,
        )

        return prediction[0].outputs[0].text
