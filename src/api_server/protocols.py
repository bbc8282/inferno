from pydantic import BaseModel
from typing import Tuple, Optional, List, Dict


class TestConfig(BaseModel):
    url: str
    model: str
    tokenizer: Optional[str] = None
    dataset_name: str
    endpoint_type: str
    key: str = "EMPTY"
    random_seed: int | None = None
    dataset_config: dict = {}
    max_run_time: float = 600
    legacy: bool = False
    workload_range: Tuple[int | None, int | None] = (None, None)
    kwargs: dict = {}
    test_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "http://209.20.156.226:8000/v1",
                    "model": "facebook/opt-125m",
                    "tokenizer": "facebook/opt-125m",
                    "key": "EMPTY",
                    "dataset_name": "synthesizer",
                    "endpoint_type": "vllm",
                    "legacy": False,
                    "max_run_time": 60,
                    "dataset_config": {
                        "func": "lambda t: int(t / 0.1 + 1) if t < 20 else None",
                        "prompt_source": "arena"
                    },
                    "kwargs": {
                        "temperature": 0.9,
                        "top_p": 1,
                        "max_tokens": 512
                    },
                    "test_id": "custom_test_001"
                }
            ]
        }
    }