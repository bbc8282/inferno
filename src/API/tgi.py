import openai
from typing import List, Dict
from .api_protocol import ResPiece
import logging
from .utils import prepare_inference_payload, handle_inference_response

logger = logging.getLogger("tgi")
logger.setLevel(logging.WARNING)

def prepare_api_base(api_base: str, legacy: bool = False) -> str:
    """
    Prepare the API base URL with proper formatting.
    
    Args:
        api_base (str): Base API URL
        legacy (bool): Whether to use legacy completions endpoint
    
    Returns:
        str: Properly formatted API base URL
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

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    try:
        api_base = prepare_api_base(kwargs.pop("api_base"), kwargs.get("legacy", False))
        openai.api_base = api_base
        openai.api_key = kwargs.pop("api_key", "EMPTY")
        legacy = kwargs.pop('legacy', False)
        kwargs.pop("stream", None)
        kwargs.setdefault("top_p", 0.5)
        
        payload = prepare_inference_payload(dialog, kwargs.pop("model"), True, legacy, **kwargs)
        
        completion = await (openai.Completion.acreate(**payload) if legacy 
                            else openai.ChatCompletion.acreate(**payload))
        
        async for chunk in completion:
            if "choices" in chunk:
                for choice in chunk.choices:
                    yield ResPiece(
                        index=choice.index,
                        role=None if legacy else choice.delta.get("role"),
                        content=choice.text if legacy else choice.delta.get("content"),
                        stop=choice.finish_reason,
                    )
                    
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        yield e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        yield e


def inference(
    dialog: List[Dict[str, str]],
    **kwargs,
) -> List[Dict[str, str]]:
    api_base = prepare_api_base(kwargs.pop("api_base"), kwargs.get("legacy", False))
    openai.api_base = api_base
    openai.api_key = kwargs.pop("api_key", "EMPTY")
    legacy = kwargs.pop('legacy', False)
    kwargs.pop("stream", None)
    kwargs.setdefault("top_p", 0.5)
    
    payload = prepare_inference_payload(dialog, kwargs.pop("model"), False, legacy, **kwargs)
    
    completion = (openai.Completion.create(**payload) if legacy 
                  else openai.ChatCompletion.create(**payload))
    
    return handle_inference_response(completion, legacy)
