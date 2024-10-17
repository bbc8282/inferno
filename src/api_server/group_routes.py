from fastapi import APIRouter, HTTPException, Body
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
def create_group(group_id: str = Body(..., example="performance_test_group")):
    """
    Create a new group with the given group_id.

    - **group_id**: A unique identifier for the new group
    
    Returns:
    - A dictionary containing the created group_id
    
    Example:
    ```
    POST /group/create
    {
        "group_id": "performance_test_group"
    }
    ```
    """
    group = db_create_group(group_id)
    return {"group_id": group}

@router.post("/register/{group_id}")
def register_tests_to_group(
    group_id: str, 
    test_ids: TestIdsModel = Body(..., example={"test_ids": ["test_001", "test_002", "test_003"]})
):
    """
    Register tests to a group.

    - **group_id**: The ID of the group to register tests to
    - **test_ids**: A list of test IDs to be added to the group
    
    Returns:
    - A message confirming the number of tests added to the group
    
    Example:
    ```
    POST /group/register/performance_test_group
    {
        "test_ids": ["test_001", "test_002", "test_003"]
    }
    ```
    """
    db_add_tests_to_group(group_id, test_ids.test_ids)
    return {"message": f"Successfully added {len(test_ids.test_ids)} tests to group '{group_id}'"}

@router.delete("/tests/{group_id}/{test_id}")
def remove_test_from_group(group_id: str, test_id: str):
    """
    Remove a specific test from a group.

    - **group_id**: The ID of the group
    - **test_id**: The ID of the test to be removed
    
    Returns:
    - A message confirming the removal of the test from the group
    
    Example:
    ```
    DELETE /group/tests/performance_test_group/test_001
    ```
    """
    success = db_remove_test_from_group(group_id, test_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Test '{test_id}' not found in group '{group_id}'")
    return {"message": f"Test '{test_id}' removed from group '{group_id}'"}

@router.get("/tests/{group_id}")
def get_group_tests(group_id: str):
    """
    Get all tests associated with a specific group.

    - **group_id**: The ID of the group

    Returns:
    - A list of tests in the group

    Example:
    ```
    GET /group/tests/performance_test_group
    ```
    """
    tests = db_get_group_tests(group_id)
    if not tests:
        raise HTTPException(status_code=404, detail=f"No tests found for group '{group_id}'")
    return {"tests": tests}

@router.get("/list")
def get_all_groups():
    """
    Get a list of all groups.

    Returns:
    - A list of all group IDs

    Example:
    ```
    GET /group/list
    ```
    """
    groups = db_get_all_groups()
    return {"groups": groups}

@router.delete("/delete/{group_id}")
def delete_group(group_id: str):
    """
    Delete a specific group.

    - **group_id**: The ID of the group to be deleted

    Returns:
    - A message confirming the deletion of the group

    Example:
    ```
    DELETE /group/delete/performance_test_group
    ```
    """
    db_remove_group(group_id)
    return {"message": f"Group {group_id} deleted successfully"}

@router.delete("/delete_all")
def delete_all_groups():
    """
    Delete all groups.

    Returns:
    - A message confirming the deletion of all groups

    Example:
    ```
    DELETE /group/delete_all
    ```
    """
    db_remove_all_groups()
    return {"message": "All groups deleted successfully"}

@router.get("/status/{group_id}")
def get_group_status(group_id: str):
    """
    Get the status of a specific group.

    - **group_id**: The ID of the group

    Returns:
    - The status of the group

    Example:
    ```
    GET /group/status/performance_test_group
    ```
    """
    status = db_check_group_status(group_id)
    return {"status": status}

@router.get("/results/{group_id}")
def get_group_test_results(group_id: str) -> Dict[str, List[Dict]]:
    """
    Get the test results for a specific group.

    - **group_id**: The ID of the group

    Returns:
    - A dictionary containing the test results for the group

    Example:
    ```
    GET /group/results/performance_test_group
    ```
    """
    results = db_get_group_test_results(group_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"No results found for group '{group_id}'")
    return {"results": results}