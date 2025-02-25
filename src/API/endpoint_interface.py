from typing import Callable, Dict
import logging
import importlib

logger = logging.getLogger("interface")
logger.setLevel(logging.WARNING)

# Centralized endpoint to module mapping
endpoint_to_module: Dict[str, str] = {
    "openai": ".openai",
    "vllm": ".vllm",
    "friendli": ".friendli",
    "tgi": ".tgi",
    "triton": ".triton",
}

def get_streaming_inference(endpoint_type: str) -> Callable:
    """
    Get the appropriate streaming inference function for a given endpoint type.
    
    Args:
        endpoint_type (str): The type of endpoint (e.g., 'openai', 'vllm', 'friendli', etc.).

    Returns:
        Callable: The streaming inference function for the specified endpoint.
    """
    module_name = endpoint_to_module.get(endpoint_type)
    if not module_name:
        raise NotImplementedError(f"Endpoint '{endpoint_type}' is not implemented.")
    try:
        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, "streaming_inference")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading streaming_inference() for endpoint {endpoint_type}: {e}")
        raise

async def get_friendli_streaming_inference():
    """
    For friendli, it is called asynchronously and in a separate method.
    """
    from .friendli import streaming_inference
    return streaming_inference

def get_inference(endpoint_type: str) -> Callable:
    """
    Get the appropriate inference function for a given endpoint type.
    
    Args:
        endpoint_type (str): The type of endpoint (e.g., 'openai', 'vllm', 'friendli', etc.).

    Returns:
        Callable: The inference function for the specified endpoint.
    """
    module_name = endpoint_to_module.get(endpoint_type)
    if not module_name:
        raise NotImplementedError(f"Endpoint '{endpoint_type}' is not implemented.")
    try:
        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, "inference")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading inference() for endpoint {endpoint_type}: {e}")
        raise