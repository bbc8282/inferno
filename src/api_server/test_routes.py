from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from typing import Literal, List, Dict
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
    add_hardware_info,
    get_hardware_info_with_cost
)
import os
import glob
import zipfile
import logging

class HardwareInfo(BaseModel):
    gpu_model: Literal['A100', 'A10', 'A30', 'A40', 'H100', 'H200']
    gpu_count: int

    @validator('gpu_count')
    def validate_gpu_count(cls, v):
        if v <= 0:
            raise ValueError('GPU count must be positive')
        return v

router = APIRouter(prefix="/tests", tags=["tests"])

@router.post("/register")
def register(config: TestConfig):
    """
    Register a new test configuration.

    - **config**: The test configuration details

    Returns:
    - The ID of the registered test
    """
    is_valid, error_message = verify_config(config)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    return save_config(config)

@router.get("/start/{test_id}")
def start_test(test_id: str):
    """
    Start a specific test.

    - **test_id**: The ID of the test to start

    Returns:
    - A confirmation message

    Example:
    ```
    GET /tests/start/test_001
    ```
    """
    return set_test_to_pending(test_id)

@router.get("/config/{test_id}")
def get_config(test_id: str):
    """
    Get the configuration of a specific test.

    - **test_id**: The ID of the test

    Returns:
    - The test configuration

    Example:
    ```
    GET /tests/config/test_001
    ```
    """
    return query_config(test_id)

@router.post("/register_and_start")
def register_and_start(config: TestConfig):
    """
    Register a new test configuration and start the test immediately.

    - **config**: The test configuration details

    Returns:
    - The ID of the registered and started test
    """
    is_valid, error_message = verify_config(config)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    test_id = save_config(config)
    set_test_to_pending(test_id)
    return test_id

@router.get("/set_nickname/{test_id}")
def set_nickname_by_id(test_id: str, nickname: str):
    """
    Set a nickname for a specific test.

    - **test_id**: The ID of the test
    - **nickname**: The nickname to set for the test

    Returns:
    - A confirmation message

    Example:
    ```
    GET /tests/set_nickname/test_001?nickname=performance_test_1
    ```
    """
    set_nickname(test_id, nickname)
    return "OK"

@router.get("/delete/{test_id}")
def delete_test_by_test_id(test_id: str):
    """
    Delete a specific test and its associated files.

    - **test_id**: The ID of the test to delete

    Returns:
    - A list of deleted files and a confirmation message

    Example:
    ```
    GET /tests/delete/test_001
    ```
    """
    delete_test(test_id)
    
    file_patterns = [
        f"tmp/*_{test_id}.*",
        f"tmp/{test_id}.zip",
        f"tmp/workload_hash_{test_id}.txt"
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
        logging.info(f"Deleted files associated with test {test_id}: {', '.join(deleted_files)}")
    
    return {"deleted_files": deleted_files, "message": f"Test {test_id} and associated files deleted successfully"}

@router.get("/delete_all")
def delete_all_tests():
    """
    Delete all tests and their associated files.

    Returns:
    - A list of deleted test IDs, any errors encountered, and a list of deleted files

    Example:
    ```
    GET /tests/delete_all
    ```
    """
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

@router.get("/model/{test_id}")
def test_model(test_id: str):
    """
    Get the model information for a specific test.

    - **test_id**: The ID of the test

    Returns:
    - The model information

    Example:
    ```
    GET /tests/model/test_001
    ```
    """
    return query_model(test_id)

@router.get("/status/{test_id}")
def test_status(test_id: str):
    """
    Get the status of a specific test.

    - **test_id**: The ID of the test

    Returns:
    - The current status of the test

    Example:
    ```
    GET /tests/status/test_001
    ```
    """
    return query_test_status(test_id)

@router.get("/error_info/{test_id}")
def error_info(test_id: str):
    """
    Get error information for a specific test, if any.

    - **test_id**: The ID of the test

    Returns:
    - Error information, if any

    Example:
    ```
    GET /tests/error_info/test_001
    ```
    """
    return query_error_info(test_id)

@router.get("/id_list")
def id_list():
    """
    Get a list of all test IDs.

    Returns:
    - A list of all test IDs

    Example:
    ```
    GET /tests/id_list
    ```
    """
    return get_id_list()

@router.get("/workload_hash/{test_id}")
def get_workload_hash(test_id: str):
    """
    Get the workload hash for a specific test.

    - **test_id**: The ID of the test

    Returns:
    - The workload hash

    Example:
    ```
    GET /tests/workload_hash/test_001
    ```
    """
    if not os.path.exists("tmp/workload_hash_" + test_id + ".txt"):
        raise HTTPException(status_code=404, detail="Workload hash not found")
    else:
        with open("tmp/workload_hash_" + test_id + ".txt", mode="r") as f:
            return f.read()

@router.get("/report/throughput/{test_id}")
def report_throughput(test_id: str):
    """
    Get the throughput report for a specific test.

    - **test_id**: The ID of the test

    Returns:
    - A PNG image of the throughput report

    Example:
    ```
    GET /tests/report/throughput/test_001
    ```
    """
    if not os.path.exists("tmp/tp_" + test_id + ".png"):
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        file_like = open("tmp/tp_" + test_id + ".png", mode="rb")
        return StreamingResponse(file_like, media_type="image/png")

@router.get("/report/requests_status/{test_id}")
def report_requests_status(test_id: str):
    """
    Get the requests status report for a specific test.

    - **test_id**: The ID of the test

    Returns:
    - A PNG image of the requests status report

    Example:
    ```
    GET /tests/report/requests_status/test_001
    ```
    """
    if not os.path.exists("tmp/rs_" + test_id + ".png"):
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        file_like = open("tmp/rs_" + test_id + ".png", mode="rb")
        return StreamingResponse(file_like, media_type="image/png")

@router.get("/report/json/{test_id}")
def report_json(test_id: str):
    """
    Get the JSON report for a specific test.

    - **test_id**: The ID of the test

    Returns:
    - A JSON file containing the test report

    Example:
    ```
    GET /tests/report/json/test_001
    ```
    """
    if not os.path.exists("tmp/report_" + test_id + ".json"):
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        file_like = open("tmp/report_" + test_id + ".json", mode="r")
        return StreamingResponse(file_like, media_type="application/json")

@router.get("/report/download/{test_id}")
def download_report(test_id: str):
    """
    Download all reports for a specific test as a zip file.

    - **test_id**: The ID of the test

    Returns:
    - A zip file containing all reports for the test

    Example:
    ```
    GET /tests/report/download/test_001
    ```
    """
    file_paths = glob.glob("tmp/*_" + test_id + ".*")
    if len(file_paths) == 0:
        raise HTTPException(status_code=404, detail="Report not found")
    else:
        zip_filename = f"tmp/{test_id}.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for file in file_paths:
                zipf.write(file, arcname=os.path.basename(file))
        file_like = open(zip_filename, mode="rb")
        return StreamingResponse(
            file_like,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{test_id}.zip"'},
        )

@router.post("/hardware/{test_id}")
def add_test_hardware_info(test_id: str, hardware_info: HardwareInfo = Body(..., example={
    "gpu_model": "A100",
    "gpu_count": 4
})):
    """
    Add hardware information for a specific test.

    - **test_id**: The ID of the test
    - **hardware_info**: The hardware information to add

    Returns:
    - A confirmation message

    Example:
    ```
    POST /tests/hardware/test_001
    {
        "gpu_model": "A100",
        "gpu_count": 4
    }
    ```
    """
    try:
        add_hardware_info(
            test_id,
            hardware_info.gpu_model,
            hardware_info.gpu_count
        )
        return {"message": "Hardware info added successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/hardware/{test_id}")
def get_test_hardware_info(test_id: str):
    """
    Get hardware information and associated cost for a specific test.

    - **test_id**: The ID of the test

    Returns:
    - Hardware information and associated cost

    Example:
    ```
    GET /tests/hardware/test_001
    ```
    """
    hardware_info = get_hardware_info_with_cost(test_id)
    if hardware_info is None:
        raise HTTPException(status_code=404, detail=f"No hardware info found for test ID: {test_id}")
    return hardware_info

# Helper function (move this to a separate utility file if needed)
def verify_config(config: TestConfig) -> tuple[bool, str]:
    """
    Verify the test configuration.

    This function checks various aspects of the test configuration to ensure it's valid.

    Parameters:
    - config: TestConfig object containing the test configuration

    Returns:
    - A tuple (is_valid, error_message), where:
        - is_valid: A boolean indicating whether the configuration is valid
        - error_message: A string describing the error if the configuration is invalid, or an empty string if it's valid
    """
    if config.max_run_time is not None and config.max_run_time <= 0:
        return False, "max_run_time must be positive"
    
    if not config.url:
        return False, "URL must be provided"
    
    if not config.model:
        return False, "Model must be specified"
    
    if config.endpoint_type not in ["tgi", "vllm", "friendliai", "triton", "openai"]:
        return False, f"Unsupported endpoint type: {config.endpoint_type}"
    
    if config.dataset_name not in ["arena", "oasst1", "synthesizer", "dolly", "openorca"]:
        return False, f"Unsupported dataset name: {config.dataset_name}"
    
    return True, ""