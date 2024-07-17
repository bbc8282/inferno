from .sim_visit import sim_visit
from ..workload_datasets.protocol import Workload
from .protocol import VisitResponse
import asyncio
from typing import List, Tuple, Dict, Any
import logging
import time
from uuid import uuid4
from .log_to_db import init_task, mark_finish_for_task

async def sim_workload_in_single_thread(
    workload: List[Tuple[float, List[Any]]],
    endpoint_type: str,
    task_id: str = "",
    **kwargs: Dict[str, Any],
) -> List[VisitResponse]:
    """
       Simulate a workload in a single thread.

       This function processes a list of visits, each containing multiple requests,
       and simulates their execution against a specified endpoint.

       Args:
           workload (List[Tuple[float, List[Any]]]): A list of tuples, each containing
               a timestamp and a list of requests to be simulated.
           endpoint_type (str): The type of endpoint to simulate against.
           task_id (str, optional): A unique identifier for this simulation task.
               If not provided, a UUID will be generated.
           **kwargs: Additional keyword arguments to be passed to the simulation.

       Returns:
           List[VisitResponse]: A list of VisitResponse objects, each containing
           the results of simulating a single visit.

       Raises:
           ValueError: If the workload is empty or invalid.
           RuntimeError: If there's an error during the simulation process.

       Note:
           This function uses asyncio for concurrent processing of requests within
           each visit, but processes visits sequentially.
       """
    task_id = task_id or str(uuid4())
    logging.info(f"<{task_id[:4]}>: start simulating.")

    tasks: List[Tuple[int, asyncio.Task]] = []
    responses: List[Tuple[int, VisitResponse]] = []
    
    # TIME_TOLERANCE: Maximum allowed time difference between scheduled and actual execution time.
    # - Smaller values increase timing accuracy but may cause more CPU usage.
    # - Larger values decrease accuracy but are more forgiving on system resources.
    # - Default is 0.05 seconds (50 milliseconds).
    TIME_TOLERANCE = kwargs.get("time_tolerance", 0.05)
    # TIME_STEP: Duration to sleep between each iteration of the main simulation loop.
    # - Smaller values provide finer granularity but increase CPU usage.
    # - Larger values are more CPU-friendly but may reduce simulation precision.
    # - Setting to 0 removes sleep, potentially maximizing CPU usage.
    # - Default is min(0.05, TIME_TOLERANCE) to balance precision and resource use.
    TIME_STEP = kwargs.pop("time_step", min(0.05, TIME_TOLERANCE))
    # CHECK_SIZE: Maximum number of tasks to check for completion in each iteration.
    # - Smaller values may increase responsiveness to individual task completion.
    # - Larger values may improve overall efficiency but could delay detection of completed tasks.
    # - Should be tuned based on expected workload and system capabilities.
    # - Default is 10, striking a balance between responsiveness and efficiency.
    CHECK_SIZE = kwargs.pop("check_size", 10)
    
    next_index = 0
    total_visit_num = len(workload)
    total_req_num = sum(len(v) for _, v in workload)
    finish_num = 0
    start_timestamp = time.time()
    
    init_task(task_id, total_req_num, start_timestamp)
    
    while True:
        # Launch new tasks
        cur_offset = time.time() - start_timestamp
        logging.debug(f"current offset {cur_offset}")

        if next_index < total_visit_num:
            assert workload[next_index][0] is not None
            logging.debug(f"next visit {next_index} scheduled at {workload[next_index][0]}")
            
            if cur_offset >= workload[next_index][0]:
                logging.debug(f"launch visit {next_index}")
                tasks.append(
                    (
                        next_index,
                        asyncio.create_task(
                            sim_visit(
                                workload[next_index][1],
                                next_index,
                                task_id,
                                endpoint_type,
                                **kwargs,
                            )
                        ),
                    )
                )
                next_index += 1
                
                next_visit_time = workload[next_index][0] if next_index < total_visit_num else None
                time_until_next = (next_visit_time - cur_offset) if next_visit_time else None
                logging.info(
                    f"<{task_id[:4]}>: launch visit {next_index - 1} finished. next visit {next_index} scheduled at {next_visit_time} after {time_until_next}"
                )

        # Recycle finished tasks & check if all visits are done
        if tasks:
            to_remove = []
            
            for i in range(min(CHECK_SIZE, len(tasks))):
                if tasks[i][1].done():
                    finish_num += 1
                    logging.info(
                        f"visit <{task_id[:4]}:{tasks[i][0]}> done. Total {finish_num}/{total_visit_num} visits done."
                    )
                    responses.append((tasks[i][0], tasks[i][1].result()))
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                tasks.pop(i)
            
            logging.debug(f"finished {len(to_remove)} tasks, {len(tasks)} tasks not finished.")
        else:
            if next_index == total_visit_num:
                break

        if TIME_STEP > 0:
            await asyncio.sleep(TIME_STEP)

    mark_finish_for_task(task_id, time.time())
    
    return [response for _, response in sorted(responses, key=lambda x: x[0])]
