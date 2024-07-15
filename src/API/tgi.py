import openai
from typing import List, Dict
from .api_protocol import ResPiece
import logging
from .utils import prepare_inference_payload, handle_inference_response

logger = logging.getLogger("tgi")
logger.setLevel(logging.WARNING)

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    try:
        openai.api_base = kwargs.pop("api_base")
        openai.api_key = kwargs.pop("api_key", "EMPTY")
        legacy = kwargs.pop('legacy', False)
        kwargs.pop("stream", None)
        
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
    openai.api_base = kwargs.pop("api_base")
    openai.api_key = kwargs.pop("api_key", "EMPTY")
    legacy = kwargs.pop('legacy', False)
    kwargs.pop("stream", None)
    
    payload = prepare_inference_payload(dialog, kwargs.pop("model"), False, legacy, **kwargs)
    
    completion = (openai.Completion.create(**payload) if legacy 
                  else openai.ChatCompletion.create(**payload))
    
    return handle_inference_response(completion, legacy)