from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import datetime
import zipfile
import glob
import time
import requests
import math
from urllib.parse import urlparse, urlunparse

from .protocols import TestConfig
from .db import (
    save_config,
    set_test_to_pending,
    query_test_status,
    query_error_info,
    query_model,
    query_config,
    get_id_list,
    set_nickname,
    delete_test,
    get_last_heartbeat,
    get_all_worker_ids,
)
from ..simulate.log_to_db import cur_requests_status_of_task, past_packs_of_task

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


@app.post("/register_test")
def register(config: TestConfig):
    return save_config(config)


@app.get("/start_test/{id}")
def start_test(id: str):
    return set_test_to_pending(id)


@app.get("/config/{id}")
def get_config(id: str):
    return query_config(id)


@app.post("/register_and_start_test")
def register_and_start(config: TestConfig):
    id = save_config(config)
    set_test_to_pending(id)
    return id


@app.get("/set_nickname/{id}")
def set_nickname_by_id(id: str, nickname: str):
    set_nickname(id, nickname)
    return "OK"


@app.get("/delete_test/{id}")
def delete_test_by_id(id: str):
    delete_test(id)
    
    file_patterns = [
        f"tmp/*_{id}.*",
        f"tmp/{id}.zip",
        f"tmp/workload_hash_{id}.txt"
    ]
    
    deleted_files = []
    for pattern in file_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                deleted_files.append(file)
            except OSError as e:
                logging.error(f"Error deleting file {file}: {e}")
    
    if deleted_files:
        logging.info(f"Deleted files associated with test {id}: {', '.join(deleted_files)}")
    
    return {"deleted_files": deleted_files, "message": f"Test {id} and associated files deleted successfully"}


@app.get("/delete_all_tests")
def delete_all_tests():
    ids = get_id_list()
    deleted_ids = []
    errors = []
    deleted_files = []
    
    file_patterns = [
        "tmp/*_*.png",
        "tmp/*_*.json",
        "tmp/*.zip",
        "tmp/*_*.pkl",
        "tmp/workload_hash_*.txt"
    ]
    
    for pattern in file_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                deleted_files.append(file)
            except OSError as e:
                logging.error(f"Error deleting file {file}: {e}")
    
    for id_info in ids:
        test_id = id_info[0]
        try:
            delete_test(test_id)
            deleted_ids.append(test_id)
        except Exception as e:
            errors.append((test_id, str(e)))
    
    if deleted_files:
        logging.info(f"Deleted files: {', '.join(deleted_files)}")
    
    if errors:
        return {
            "deleted_ids": deleted_ids, 
            "errors": errors, 
            "deleted_files": deleted_files
        }
    return {
        "deleted_ids": deleted_ids, 
        "message": "All tests and associated files deleted successfully.", 
        "deleted_files": deleted_files
    }


@app.get("/test_model/{id}")
def test_model(id: str):
    return query_model(id)


@app.get("/test_status/{id}")
def test_status(id: str):
    return query_test_status(id)


@app.get("/error_info/{id}")
def error_info(id: str):
    return query_error_info(id)


@app.get("/id_list")
def id_list():
    return get_id_list()


@app.get("/get/workload_hash/{id}")
def get_workload_hash(id: str):
    if not os.path.exists("tmp/workload_hash_" + id + ".txt"):
        raise HTTPException(status_code=404, detail="Workload hash not found")
    else:
        with open("tmp/workload_hash_" + id + ".txt", mode="r") as f:
            return f.read()
        
@app.get("/get/vllm_metrics/{id}", tags=['vLLM'])
def get_vllm_metrics(id: str):
    config = get_config(id)
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
        raise HTTPException(status_code=400, detail=f"The specified server({metrics_url}) is not a vllm server.")


@app.get("/get/frienfli_metrics/{id}", tags=['FriendliAI'])
def get_friendli_metrics(id: str, port: str):
    config = get_config(id)
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


@app.get("/report/throughput/{id}", tags=['report'])
def report_throughput(id: str):
    if not os.path.exists("tmp/tp_" + id + ".png"):
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        file_like = open("tmp/tp_" + id + ".png", mode="rb")
        return StreamingResponse(file_like, media_type="image/png")


@app.get("/report/requests_status/{id}", tags=['report'])
def report_requests_status(id: str):
    if not os.path.exists("tmp/rs_" + id + ".png"):
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        file_like = open("tmp/rs_" + id + ".png", mode="rb")
        return StreamingResponse(file_like, media_type="image/png")


@app.get("/report/json/{id}", tags=['report'])
def report_json(id: str):
    if not os.path.exists("tmp/report_" + id + ".json"):
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        file_like = open("tmp/report_" + id + ".json", mode="r")
        return StreamingResponse(file_like, media_type="application/json")


@app.get("/report/download/{id}", tags=['report'])
def download_report(id: str):
    file_paths = glob.glob("tmp/*_" + id + ".*")
    if len(file_paths) == 0:
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        zip_filename = f"tmp/{id}.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for file in file_paths:
                zipf.write(file, arcname=os.path.basename(file))
        file_like = open(zip_filename, mode="rb")
        return StreamingResponse(
            file_like,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{id}.zip"'},
        )


@app.get("/trace/status/{id}", tags=['trace'])
def trace_status(id: str):
    return cur_requests_status_of_task(id)


@app.get("/trace/tps/{id}", tags=['trace'])
def trace_tps(id: str, model: str, sample_len: int = 5):
    packs = past_packs_of_task(id, past_time=sample_len)
    from ..analysis.generate_report import count_tokens_from_str, load_tokenizer

    try:
        tokenizer = load_tokenizer(model)
        tokens = sum([count_tokens_from_str(p, tokenizer) for p in packs])
        return {
            "tps": tokens / sample_len,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logging.warning(f"<trace_tps {id}>: tps failed: {str(e)}")
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
