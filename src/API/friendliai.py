import requests
import json
from typing import List, Dict
from .api_protocol import ResPiece
import logging
import aiohttp

logger = logging.getLogger("friendliai")
logger.setLevel(logging.WARNING)

def format_legacy_dialog(dialog: List[Dict[str, str]]) -> str:
    return "\n".join([f"{message['role']}: {message['content']}" for message in dialog])

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    """Perform streaming inference with SSE (Server-Sent Events)."""
    try:
        if "stream" in kwargs:
            kwargs.pop("stream")
        legacy = kwargs.pop('legacy', False)
        
        if legacy:
            url = f"{kwargs.pop('api_base')}/completions"
        else:
            url = f"{kwargs.pop('api_base')}/chat/completions"
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "Authorization": f"Bearer {kwargs.pop('api_key', None)}",
        }
        
        payload = {
            "model": kwargs['model'],
            "stream": True,
            **kwargs,
        }

        if legacy:
            payload["prompt"] = format_legacy_dialog(dialog)
        else:
            payload["messages"] = dialog
            
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    logger.error('Rate limit exceeded, consider backing off')
                    raise Exception('Rate limit exceeded, consider backing off')
                async for chunk in response.content:
                    s = chunk.decode().strip()
                    if s.startswith('data:'):
                        data = s.split(':', 1)[1].strip()
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                        except:
                            print(s)
                        
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
                                role = choice["delta"].get("role", None)
                                content = choice["delta"].get("content", "")
                                yield ResPiece(
                                    index=choice["index"],
                                    role=role,
                                    content=content,
                                    stop=choice.get("finish_reason", None),
                                )
    except Exception as e:
        yield e

def inference(
    dialog: List[Dict[str, str]],
    **kwargs,
) -> List[Dict[str, str]]:
    if "stream" in kwargs:
        kwargs.pop("stream")
    url = kwargs.pop("api_base")
    headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {kwargs.pop('api_key', None)}",
        }
    
    legacy = kwargs.pop('legacy', False)
    payload = {
        "model": kwargs['model'],
        "stream": False,
        **kwargs,
    }
    
    if legacy:
        payload["prompt"] = format_legacy_dialog(dialog)
    else:
        payload["messages"] = dialog

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    json_data = response.json()
    
    if legacy:
        return [c["text"] for c in json_data["choices"]]
    else:
        return [c["message"]["content"] for c in json_data["choices"]]
