import openai
from typing import List, Dict, AsyncGenerator
from .api_protocol import ResPiece
import logging

logger = logging.getLogger("openai")
logger.setLevel(logging.WARNING)

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs
) -> AsyncGenerator[ResPiece, None]:
    try:
        api_key = kwargs.pop("api_key")
        model = kwargs.pop("model")
        legacy = kwargs.pop("legacy", False)
        openai.api_key = api_key

        if legacy:
            # Use Completion API for legacy mode
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in dialog])
            response = await openai.Completion.acreate(
                model=model,
                prompt=prompt,
                stream=True,
                **kwargs
            )
            async for chunk in response:
                if chunk.choices[0].text:
                    yield ResPiece(
                        index=chunk.choices[0].index,
                        role=None,
                        content=chunk.choices[0].text,
                        stop=chunk.choices[0].finish_reason
                    )
        else:
            # Use Chat Completion API for non-legacy mode
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=dialog,
                stream=True,
                **kwargs
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield ResPiece(
                        index=chunk.choices[0].index,
                        role=chunk.choices[0].delta.role,
                        content=chunk.choices[0].delta.content,
                        stop=chunk.choices[0].finish_reason
                    )

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        yield ResPiece(index=0, role=None, content=str(e), stop=None)

def inference(
    dialog: List[Dict[str, str]],
    **kwargs
) -> List[Dict[str, str]]:
    api_key = kwargs.pop("api_key")
    model = kwargs.pop("model")
    legacy = kwargs.pop("legacy", False)
    openai.api_key = api_key

    try:
        if legacy:
            # Use Completion API for legacy mode
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in dialog])
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                **kwargs
            )
            return [{"role": "assistant", "content": response.choices[0].text}]
        else:
            # Use Chat Completion API for non-legacy mode
            response = openai.ChatCompletion.create(
                model=model,
                messages=dialog,
                **kwargs
            )
            return [response.choices[0].message.to_dict()]

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise