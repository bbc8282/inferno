import json
from typing import List, Dict
from .api_protocol import ResPiece
import logging
import aiohttp
import requests
from .utils import prepare_inference_payload, handle_inference_response

logger = logging.getLogger("friendli")

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
    """Perform streaming inference with SSE (Server-Sent Events)."""
    try:
        api_base = prepare_api_base(kwargs.pop("api_base"), kwargs.get("legacy", False))
        api_key = kwargs.pop("api_key", None)
        legacy = kwargs.pop('legacy', False)
        kwargs.pop("stream", None)
        
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = prepare_inference_payload(dialog, kwargs.pop("model"), True, legacy, **kwargs)

        logger.info("=== STREAMING INFERENCE HTTP REQUEST ===")
        logger.info(f"API Base: {api_base}")
        logger.info(f"Headers : {json.dumps(dict(headers), indent=2, ensure_ascii=False)}")
        logger.info(f"Payload : {json.dumps(payload, indent=2, ensure_ascii=False)}")
        logger.info("=========================================")
            
        async with aiohttp.ClientSession() as session:
            async with session.post(api_base, json=payload, headers=headers) as response:
                if response.status == 429:
                    raise Exception('Rate limit exceeded, consider backing off')
                async for chunk in response.content:
                    s = chunk.decode().strip()
                    if s.startswith('data:'):
                        data = s.split(':', 1)[1].strip()
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if legacy:
                                if "event" in json_data and json_data["event"] == "token_sampled":
                                    yield ResPiece(
                                        index=json_data["index"],
                                        role=None,
                                        content=json_data["text"],
                                        stop=json_data.get("finish_reason", None),
                                    )
                            else:
                                for choice in json_data["choices"]:
                                    yield ResPiece(
                                        index=choice["index"],
                                        role=choice["delta"].get("role"),
                                        content=choice["delta"].get("content", ""),
                                        stop=choice.get("finish_reason", None),
                                    )
                        except json.JSONDecodeError:
                           logger.error(f"Failed to parse JSON: {s}")
    except Exception as e:
        yield e

def inference(
    dialog: List[Dict[str, str]],
    **kwargs,
) -> List[Dict[str, str]]:
    api_base = prepare_api_base(kwargs.pop("api_base"), kwargs.get("legacy", False))
    api_key = kwargs.pop("api_key", None)
    legacy = kwargs.pop('legacy', False)
    kwargs.pop("stream", None)
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = prepare_inference_payload(dialog, kwargs.pop("model"), False, legacy, **kwargs)

    response = requests.post(api_base, json=payload, headers=headers)
    response.raise_for_status()
    json_data = response.json()

    return handle_inference_response(json_data, legacy)