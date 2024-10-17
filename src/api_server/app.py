from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import datetime
import time
import math

from .test_routes import router as test_router
from .group_routes import router as group_router
from .recommendation_routes import router as recommendation_router
from .db import (
    get_last_heartbeat,
    get_all_worker_ids,
)
from ..simulate.log_to_db import cur_requests_status_of_task, past_packs_of_task
from ..workload_datasets.utils import AVAILABLE_DATASETS

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(test_router)
app.include_router(group_router)
app.include_router(recommendation_router)

def parse_prometheus_text(metrics_text: str):
    lines = metrics_text.strip().split("\n")
    metrics = {}
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.split(None, 1)
        if "{" in key:
            key_base, labels_part = key.split("{", 1)
            labels_part = labels_part.rstrip("}")
            labels = {}
            for label in labels_part.split(","):
                label_key, label_value = label.split("=", 1)
                labels[label_key] = label_value.strip('"')
            value = float(value) if not math.isnan(float(value)) else None
            metrics.setdefault(key_base, []).append({"labels": labels, "value": value})
        else:
            value = float(value) if not math.isnan(float(value)) else None
            metrics[key] = {"value": value}
    return metrics
    
@app.get("/workers", tags=['worker'])
def list_workers():
    worker_ids = get_all_worker_ids()
    return {"workers": worker_ids}

@app.get("/workers/{worker_id}", tags=['worker'])
def worker_health_check(worker_id: str):
    last_heartbeat = get_last_heartbeat(worker_id=worker_id)
    if time.time() - last_heartbeat < 10:
        return {"status": "healthy"}
    else:
        return {"status": "unresponsive"}

@app.get("/dataset_list", tags=['dataset'])
def dataset_list():
    return {
        "available_datasets": [
            {
                "id": dataset_id,
                "name": dataset_info["name"],
                "description": dataset_info["description"]
            }
            for dataset_id, dataset_info in AVAILABLE_DATASETS.items()
        ]
    }
"""
@app.get("/get/vllm_metrics/{test_id}", tags=['vLLM'])
def get_vllm_metrics(test_id: str):
    config = query_config(test_id)
    if config.endpoint_type == "vllm":
        try:
            parsed_url = urlparse(config.url)
            metrics_url = urlunparse((parsed_url.scheme, parsed_url.netloc, '/metrics', '', '', ''))
            response = requests.get(metrics_url)
            response.raise_for_status()
            try:
                parsed_metrics = parse_prometheus_text(response.text)
                return parsed_metrics
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse metrics: {str(e)}")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics from vllm server({metrics_url}): {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"The specified server({config.url}) is not a vllm server.")

@app.get("/get/friendli_metrics/{test_id}", tags=['FriendliAI'])
def get_friendli_metrics(test_id: str, port: str):
    config = query_config(test_id)
    if config.endpoint_type == "friendliai":
        try:
            parsed_url = urlparse(config.url)
            netloc = f"{parsed_url.hostname}:{port}"
            metrics_url = urlunparse((parsed_url.scheme, netloc, '/metrics', '', '', ''))
            response = requests.get(metrics_url)
            response.raise_for_status()
            try:
                parsed_metrics = parse_prometheus_text(response.text)
                return parsed_metrics
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse metrics: {str(e)}")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics from friendliai server({config.url}): {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"The specified server({config.url}) is not a friendliai server.")
"""

@app.get("/trace/status/{test_id}", tags=['trace'])
def trace_status(test_id: str):
    return cur_requests_status_of_task(test_id)

@app.get("/trace/tps/{test_id}", tags=['trace'])
def trace_tps(test_id: str, model: str, sample_len: int = 5):
    packs = past_packs_of_task(test_id, past_time=sample_len)
    from ..analysis.generate_report import count_tokens_from_str, load_tokenizer

    try:
        tokenizer = load_tokenizer(model)
        tokens = sum([count_tokens_from_str(p, tokenizer) for p in packs])
        return {
            "tps": tokens / sample_len,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logging.warning(f"<trace_tps {test_id}>: tps failed: {str(e)}")
        return {
            "tps": 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


if __name__ == "__main__":
    import uvicorn
    
    logging.info("Starting FastAPI server")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
#        ssl_keyfile="key.pem",
#        ssl_certfile="fullchain.pem",
    )
