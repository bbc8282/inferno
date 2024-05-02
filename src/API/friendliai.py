import requests
import json
import openai
from typing import List, Dict
from .api_protocol import ResPiece
import logging
import aiohttp

logger = logging.getLogger("friendliai")
logger.setLevel(logging.WARNING)

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    """Perform streaming inference with SSE (Server-Sent Events)."""
    try:
        if "stream" in kwargs:
            kwargs.pop("stream")
        url = kwargs.pop("api_base")
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "Authorization": f"Bearer {kwargs.pop('api_key', None)}",
        }
        payload = {
            "model": kwargs['model'],
            "stream": True,
            "messages": dialog,
            **kwargs,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    logger.error('Rate limit exceeded, consider backing off')
                async for chunk in response.content:
                    s = chunk.decode().strip()
                    if s.startswith('data:'):
                        data = s.split(':', 1)[1].strip()
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            for choice in json_data["choices"]:
                                role = choice["delta"].get("role", None)
                                content = choice["delta"].get("content", "")
                                yield ResPiece(
                                    index=choice["index"],
                                    role=role,
                                    content=content,
                                    stop=choice.get("finish_reason", None),
                                )
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON: {data}")
                            continue
    except Exception as e:
        logging.exception(f"An error occurred during streaming inference: {str(e)}")
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
    payload = {
        "model": kwargs['model'],
        "stream": False,
        "messages": dialog,
        **kwargs,
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.text