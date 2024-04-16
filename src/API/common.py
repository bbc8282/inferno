import requests
import json
import aiohttp
import logging
from typing import List, Dict
from .api_protocol import ResPiece

logger = logging.getLogger("common")
logger.setLevel(logging.WARNING)

async def streaming_inference(prompt: str, **kwargs):
    try:
        url = kwargs.pop("api_base")
        kwargs.pop("api_key")
        kwargs.pop("model")
        payload = {"prompt": prompt, "stream": False, **kwargs}
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"Received non-200 response from server: {response.status}")
                    return
                async for chunk in response.content:
                    if chunk == b"\n":
                        continue
                    s = chunk.decode()[6:]
                    if s == "[DONE]\n":
                        break
                    try:
                        json_data = json.loads(s)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}")
                        continue

                    for choice in json_data["choices"]:
                        role = choice["delta"].get("role")
                        content = choice["delta"].get("content")
                        yield ResPiece(
                            index=choice["index"],
                            role=role,
                            content=content,
                            stop=choice.get("finish_reason"),
                        )
    except Exception as e:
        logger.error(f"Error during streaming inference: {e}")
        yield e

def inference(dialog: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
    url = "http://10.0.0.42:8000/generate"
    payload = {
        "stream": False,
        "messages": dialog,
        **kwargs,
    }
    headers = {"accept": "application/json", "content-type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # This will raise an exception for HTTP error codes
        data = response.json()
        return data.get("choices", [])
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []

if __name__ == "__main__":
    import asyncio
    from rich.live import Live
    from rich.table import Table
    from rich import box

    conf = {"temperature": 0.9, "top_p": 0.9}
    n = 1

    def create_table(dlist):
        table = Table(box=box.SIMPLE)
        for i in range(n):
            table.add_column(f"Stream {i}")
        table.add_row(*dlist)
        return table

    async def main():
        streaming = streaming_inference(prompt="당신의 이름이 뭐죠?", **conf)
        data_columns = [""] * n
        with Live(auto_refresh=True) as live:
            async for ctx in streaming:
                if isinstance(ctx, Exception):
                    raise ctx
                s = ""
                if ctx.role is not None:
                    s += f"<{ctx.role}>: "
                if ctx.content is not None:
                    s += ctx.content
                if ctx.stop is not None:
                    s += f" <{ctx.stop}>"
                data_columns[ctx.index] = data_columns[ctx.index] + s
                live.update(create_table(data_columns))

    asyncio.run(main())
