import logging
from typing import List, Dict, Any

logger = logging.getLogger("api_utils")

def format_dialog(dialog: List[Dict[str, str]], legacy: bool = False) -> str:
    """Format the dialog based on legacy mode."""
    if legacy:
        return "\n".join([f"{message['role']}: {message['content']}" for message in dialog])
    return dialog

async def make_streaming_request(session, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Any:
    """Make a streaming request to the API."""
    async with session.post(url, json=payload, headers=headers) as response:
        if response.status == 429:
            raise Exception('Rate limit exceeded, consider backing off')
        async for chunk in response.content:
            yield chunk

def prepare_inference_payload(dialog: List[Dict[str, str]], model: str, stream: bool, legacy: bool, **kwargs) -> Dict[str, Any]:
    """Prepare the payload for an inference request."""
    payload = {
        "model": model,
        "stream": stream,
        **kwargs
    }
    if legacy:
        payload["prompt"] = format_dialog(dialog, legacy=True)
    else:
        payload["messages"] = dialog
    return payload

def handle_inference_response(response: Any, legacy: bool) -> List[Dict[str, str]]:
    """Handle the response from a non-streaming inference request."""
    if legacy:
        return [c["text"] for c in response["choices"]]
    else:
        return [c["message"]["content"] for c in response["choices"]]