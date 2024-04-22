import openai
from typing import Callable, List, Dict
from .api_protocol import ResPiece
import logging

logger = logging.getLogger("openai")
logger.setLevel(logging.WARNING)


def get_streaming_inference(
    endpoint_type: str,
) -> Callable:
    if endpoint_type == "openai":
        from .openai import streaming_inference

        return streaming_inference
    elif endpoint_type == "togetherai":
        from .togetherai import streaming_inference

        return streaming_inference
    elif endpoint_type == "aws":
        from .aws import streaming_inference

        return streaming_inference
    elif endpoint_type == "vllm":
        from .vllm import streaming_inference

        return streaming_inference
    elif endpoint_type == "friendliai":
        from .friendliai import streaming_inference

        return streaming_inference
    else:
        raise NotImplementedError


def get_inference(
    endpoint_type: str,
) -> Callable:
    if endpoint_type == "openai":
        from .openai import inference

        return inference
    elif endpoint_type == "togetherai":
        from .togetherai import inference

        return inference
    elif endpoint_type == "aws":
        from .aws import inference

        return inference
    elif endpoint_type == "vllm":
        from .vllm import inference

        return inference
    elif endpoint_type == "friendliai":
        from .friendliai import inference

        return inference
    else:
        raise NotImplementedError