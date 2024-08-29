from typing import Callable, List, Dict
from .api_protocol import ResPiece
import logging
import importlib

logger = logging.getLogger("interface")
logger.setLevel(logging.WARNING)

# Centralized endpoint to module mapping
endpoint_to_module: Dict[str, str] = {
    "openai": ".openai",
    "vllm": ".vllm",
    "friendliai": ".friendliai",
    "tgi": ".tgi",
    "triton": ".triton",
}

def get_streaming_inference(endpoint_type: str,) -> Callable:
    module_name = endpoint_to_module.get(endpoint_type)
    if not module_name:
        raise NotImplementedError(f"Endpoint '{endpoint_type}' is not implemented.")
    try:
        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, "streaming_inference")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading streaming_inference() for endpoint {endpoint_type}: {e}")
        raise

async def get_friendliai_streaming_inference():
    """
    For friendliai, it is called asynchronously and in a separate method.
    """
    from .friendliai import streaming_inference
    return streaming_inference

def get_inference(endpoint_type: str,) -> Callable:
    module_name = endpoint_to_module.get(endpoint_type)
    if not module_name:
        raise NotImplementedError(f"Endpoint '{endpoint_type}' is not implemented.")
    try:
        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, "inference")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading inference() for endpoint {endpoint_type}: {e}")
        raise
