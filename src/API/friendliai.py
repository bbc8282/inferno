import json
from typing import List, Dict
from .api_protocol import ResPiece
import logging
import aiohttp
from .utils import prepare_inference_payload, handle_inference_response

logger = logging.getLogger("friendliai")
logger.setLevel(logging.WARNING)

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    """Perform streaming inference with SSE (Server-Sent Events)."""
    try:
        api_base = kwargs.pop("api_base")
        api_key = kwargs.pop("api_key", None)
        legacy = kwargs.pop('legacy', False)
        kwargs.pop("stream", None)
        
        url = f"{api_base}/completions" if legacy else f"{api_base}/chat/completions"
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = prepare_inference_payload(dialog, kwargs.pop("model"), True, legacy, **kwargs)
            
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
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
                            print(f"Failed to parse JSON: {s}")
    except Exception as e:
        yield e

def inference(
    dialog: List[Dict[str, str]],
    **kwargs,
) -> List[Dict[str, str]]:
    import requests
    
    api_base = kwargs.pop("api_base")
    api_key = kwargs.pop("api_key", None)
    legacy = kwargs.pop('legacy', False)
    kwargs.pop("stream", None)
    
    url = f"{api_base}/completions" if legacy else f"{api_base}/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = prepare_inference_payload(dialog, kwargs.pop("model"), False, legacy, **kwargs)

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    json_data = response.json()

    return handle_inference_response(json_data, legacy)