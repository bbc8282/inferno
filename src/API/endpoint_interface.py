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
    "friendli": ".friendli",
    "tgi": ".tgi",
    "triton": ".triton",
}

def build_api_url(api_base: str, legacy: bool = False) -> str:
    """
    Build the API URL with proper formatting.
    
    Args:
        api_base (str): Base API URL
        legacy (bool): Whether to use legacy completions endpoint
    
    Returns:
        str: Properly formatted API URL
    """
    # Add http:// if no protocol specified
    if not api_base.startswith(('http://', 'https://')):
        api_base = f'http://{api_base}'
    
    # Remove trailing slash if present
    api_base = api_base.rstrip('/')
    
    # Ensure "/v1" is present if missing
    if not api_base.endswith('/v1'):
        api_base = f"{api_base}/v1"
    
    # Add appropriate endpoint
    endpoint = '/completions' if legacy else '/chat/completions'
    
    return f"{api_base}{endpoint}"

def get_streaming_inference(endpoint_type: str, api_base: str, **kwargs) -> Callable:
    """
    Get the appropriate streaming inference function for a given endpoint type.
    
    Args:
        endpoint_type (str): The type of endpoint (e.g., 'openai', 'vllm', 'friendli', etc.).
        api_base (str): The base API URL.

    Returns:
        Callable: The streaming inference function for the specified endpoint.
    """
    module_name = endpoint_to_module.get(endpoint_type)
    if not module_name:
        raise NotImplementedError(f"Endpoint '{endpoint_type}' is not implemented.")
    try:
        module = importlib.import_module(module_name, package=__package__)
        formatted_api_url = build_api_url(api_base, kwargs.pop("legacy", False))
        kwargs["api_base"] = formatted_api_url
        
        return getattr(module, "streaming_inference")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading streaming_inference() for endpoint {endpoint_type}: {e}")
        raise

async def get_friendli_streaming_inference(api_base: str, **kwargs):
    """
    Get Friendli's streaming inference function with proper API URL formatting.
    """
    from .friendli import streaming_inference
    kwargs["api_base"] = build_api_url(api_base, kwargs.pop("legacy", False))
    return streaming_inference

def get_inference(endpoint_type: str, api_base: str, **kwargs) -> Callable:
    """
    Get the appropriate inference function for a given endpoint type.
    
    Args:
        endpoint_type (str): The type of endpoint (e.g., 'openai', 'vllm', 'friendli', etc.).
        api_base (str): The base API URL.

    Returns:
        Callable: The inference function for the specified endpoint.
    """
    module_name = endpoint_to_module.get(endpoint_type)
    if not module_name:
        raise NotImplementedError(f"Endpoint '{endpoint_type}' is not implemented.")
    try:
        module = importlib.import_module(module_name, package=__package__)
        formatted_api_url = build_api_url(api_base, kwargs.pop("legacy", False))
        kwargs["api_base"] = formatted_api_url
        
        return getattr(module, "inference")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading inference() for endpoint {endpoint_type}: {e}")
        raise