import json
import logging
from typing import List, Dict, Any
import aiohttp
import asyncio
from .api_protocol import ResPiece
import tritonclient.http as httpclient

logger = logging.getLogger("triton")
logger.setLevel(logging.WARNING)

def prepare_triton_input(dialog: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    # Prepare the input for Triton Inference Server
    # This may need to be adjusted based on your specific model's requirements
    input_text = dialog[-1]["content"]  # Get the last message as input
    return {
        "inputs": [
            {
                "name": "INPUT_0",
                "shape": [1],
                "datatype": "BYTES",
                "data": [input_text.encode('utf-8')]
            }
        ]
    }

async def streaming_inference(
    dialog: List[Dict[str, str]],
    **kwargs,
):
    try:
        url = kwargs.pop("api_base")
        model_name = kwargs.pop("model")
        kwargs.pop("stream", None)

        triton_input = prepare_triton_input(dialog, **kwargs)

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url}/v2/models/{model_name}/infer", json=triton_input) as response:
                if response.status != 200:
                    raise Exception(f"Triton inference failed with status {response.status}")
                
                result = await response.json()
                
                # Parse the Triton response
                output = result["outputs"][0]["data"][0]
                decoded_output = output.encode('utf-8').decode('utf-8')
                
                # Yield the response as a ResPiece
                yield ResPiece(
                    index=0,
                    role="assistant",
                    content=decoded_output,
                    stop=None
                )

    except Exception as e:
        logger.error(f"Error in Triton streaming inference: {str(e)}")
        yield e

def inference(
    dialog: List[Dict[str, str]],
    **kwargs,
) -> List[Dict[str, str]]:
    url = kwargs.pop("api_base")
    model_name = kwargs.pop("model")
    kwargs.pop("stream", None)

    triton_input = prepare_triton_input(dialog, **kwargs)

    try:
        triton_client = httpclient.InferenceServerClient(url=url)
        response = triton_client.infer(model_name, triton_input)

        # Parse the Triton response
        output = response.as_numpy("OUTPUT_0")[0]
        decoded_output = output.decode('utf-8')

        return [{"role": "assistant", "content": decoded_output}]
    except Exception as e:
        logger.error(f"Error in Triton inference: {str(e)}")
        raise