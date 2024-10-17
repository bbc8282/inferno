from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from .db import (
    db_create_group,
    db_add_tests_to_group,
    db_get_all_groups,
    db_remove_group,
    db_remove_all_groups,
    db_check_group_status,
    db_get_group_test_results,
    db_get_group_tests,
    db_remove_test_from_group
)

router = APIRouter(prefix="/group", tags=["group"])

class TestIdsModel(BaseModel):
    test_ids: List[str]

    class Config:
        schema_extra = {
            "example": {
                "test_ids": ["test_001", "test_002", "test_003"]
            }
        }

@router.post("/create")
def create_group(group_id: str):
    group = db_create_group(group_id)
    return {"group_id": group}

@router.post("/register/{group_id}")
def register_tests_to_group(
    group_id: str, 
    test_ids: TestIdsModel
):
    """
    Register tests to a group.

    - **group_id**: The ID of the group to register tests to
    - **test_ids**: A list of test IDs to be added to the group
    """
    db_add_tests_to_group(group_id, test_ids.test_ids)
    return {"message": f"Successfully added {len(test_ids.test_ids)} tests to group '{group_id}'"}

@router.delete("/tests/{group_id}/{test_id}")
def remove_test_from_group(group_id: str, test_id: str):
    success = db_remove_test_from_group(group_id, test_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Test '{test_id}' not found in group '{group_id}'")
    return {"message": f"Test '{test_id}' removed from group '{group_id}'"}

@router.get("/tests/{group_id}")
def get_group_tests(group_id: str):
    tests = db_get_group_tests(group_id)
    if not tests:
        raise HTTPException(status_code=404, detail=f"No tests found for group '{group_id}'")
    return {"tests": tests}

@router.get("/list")
def get_all_groups():
    groups = db_get_all_groups()
    return {"groups": groups}

@router.delete("/delete/{group_id}")
def delete_group(group_id: str):
    db_remove_group(group_id)
    return {"message": f"Group {group_id} deleted successfully"}

@router.delete("/delete_all")
def delete_all_groups():
    db_remove_all_groups()
    return {"message": "All groups deleted successfully"}

@router.get("/status/{group_id}")
def get_group_status(group_id: str):
    status = db_check_group_status(group_id)
    return {"status": status}

@router.get("/results/{group_id}")
def get_group_test_results(group_id: str) -> Dict[str, List[Dict]]:
    results = db_get_group_test_results(group_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"No results found for group '{group_id}'")
    return {"results": results}