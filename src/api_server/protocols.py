from pydantic import BaseModel
from typing import Tuple


class TestConfig(BaseModel):
    url: str
    model: str
    dataset_name: str
    endpoint_type: str
    key: str = "EMPTY"
    random_seed: int | None = None
    dataset_config: dict = {}
    legacy: bool = False
    workload_range: Tuple[int | None, int | None] = (None, None)
    kwargs: dict = {}

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "http://209.20.156.226:8000/v1",
                    "model": "facebook/opt-125m",
                    "key": "EMPTY",
                    "dataset_name": "synthesizer",
                    "endpoint_type": "vllm",
                    "legacy": False,
                    "dataset_config": {
                        "func": "lambda t: int(t / 0.1 + 1) if t < 20 else None",
                        "prompt_source": "arena"
                    },
                    "kwargs": {
                        "temperature": 0.9,
                        "top_p": 1,
                        "max_tokens": 512
                    }
                }
            ]
        }
    }
    

    def get_model_name(self):
        return self.model.split("/")[-1]
    
    def get_model_full_name(self):
        if self.endpoint_type == "friendliai":
            if self.model == "meta-llama-3-70b-instruct":
                return "meta-llama/Meta-Llama-3-70B-Instruct"
            elif self.model == "llama-2-13b-chat":
                return "meta-llama/Llama-2-13b-chat-hf"
            elif self.model == "llama-2-70b-chat":
                return "meta-llama/Llama-2-70b-chat-hf"
            elif self.model == "mistral-7b-instruct-v0-2":
                return "mistralai/Mistral-7B-Instruct-v0.2"
            elif self.model == "mixtral-8x7b-instruct-v0-1":
                return "mistralai/Mixtral-8x7B-Instruct-v0.1"
            elif self.model == "gemma-7b-it":
                return "google/gemma-1.1-7b-it"
            elif "gpt" in self.model:
                return self.model.split("/")[-1]
            else:
                return self.model
        else:
            return self.model
