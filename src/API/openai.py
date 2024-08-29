import openai
from typing import List, Dict
from .api_protocol import ResPiece
import logging
from .utils import prepare_inference_payload, handle_inference_response

logger = logging.getLogger("openai")
logger.setLevel(logging.WARNING)

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    try:
        openai.api_key = kwargs.pop("api_key")
        model = kwargs.pop("model")
        legacy = kwargs.pop('legacy', False)
        kwargs.pop("stream", None)

        payload = prepare_inference_payload(dialog, model, True, legacy, **kwargs)

        if legacy:
            async for chunk in await openai.Completion.acreate(**payload):
                yield ResPiece(
                    index=chunk.index,
                    role=None,
                    content=chunk.text,
                    stop=chunk.finish_reason,
                )
        else:
            async for chunk in await openai.ChatCompletion.acreate(**payload):
                if "choices" in chunk:
                    for choice in chunk.choices:
                        yield ResPiece(
                            index=choice.index,
                            role=choice.delta.get("role"),
                            content=choice.delta.get("content"),
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
    openai.api_key = kwargs.pop("api_key")
    model = kwargs.pop("model")
    legacy = kwargs.pop('legacy', False)
    kwargs.pop("stream", None)

    payload = prepare_inference_payload(dialog, model, False, legacy, **kwargs)

    try:
        if legacy:
            completion = openai.Completion.create(**payload)
        else:
            completion = openai.ChatCompletion.create(**payload)
        return handle_inference_response(completion, legacy)
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise