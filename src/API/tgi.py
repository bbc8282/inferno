import openai
from typing import List, Dict
from .api_protocol import ResPiece
import logging

logger = logging.getLogger("tgi")
logger.setLevel(logging.WARNING)


async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    try:
        openai.api_base = kwargs.pop("api_base")
        if "stream" in kwargs:
            kwargs.pop("stream")
        
        completion = await openai.ChatCompletion.acreate(
            model=kwargs.pop("model"),
            messages=dialog,
            stream=True,
            **kwargs,
        )
        async for chunk in completion:
            for choice in chunk.choices:
                role, content = None, None
                if "role" in choice.delta:
                    role = choice.delta["role"]
                if "content" in choice.delta:
                    content = choice.delta["content"]
                yield ResPiece(
                    index=choice.index,
                    role=role,
                    content=content,
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
    
    if "stream" in kwargs:
            kwargs.pop("stream")
            
    completion = openai.ChatCompletion.create(
        model=kwargs.pop("model"),
        messages=dialog,
        **kwargs,
    )
    return [c["message"] for c in completion.choices]