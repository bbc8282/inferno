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
    sim_start_time: float | None,
    endpoint_type: str,
    task_id: str = "",
    **kwargs: Dict[str, Any],
) -> List[VisitResponse]:
    """
    Simulate a workload and return the responses.

    Parameters:
    - workload: List of tuples, each containing a float and a list.
    - sim_start_time: The start time for the simulation.
    - endpoint_type: The type of endpoint.
    - task_id: The ID of the task.
    - kwargs: Additional keyword arguments.

    Returns:
    - List of VisitResponse objects.
    """
    
    if not task_id:
        task_id = str(uuid4())
        logging.info(f"sim_workload_in_single_thread: task_id is not set, set to {task_id}")
    
    TIME_TOLERANCE = kwargs.get("time_tolerance", 0.1)
    SKIP_IDLE_MIN: float | None = kwargs.pop("skip_idle_min", None)
    
    if sim_start_time is None:
        logging.info(f"<{task_id[:4]}>: sim_start_time is not set, simulating immediately.")
    elif sim_start_time - time.time() < TIME_TOLERANCE:
        logging.warning(
            f"sim_start_time is set to {sim_start_time}, but current time is {time.time()}, simulating run immediately."
        )
    else:
        logging.info(
            f"sim_start_time is set to {sim_start_time}, simulating after {sim_start_time - time.time()} seconds."
        )
        await asyncio.sleep(sim_start_time - time.time())
    
    logging.info(f"<{task_id[:4]}>:: start simulating.")

    tasks: List[Tuple[int, asyncio.Task]] = []
    responses: List[Tuple[int, VisitResponse]] = []

    TIME_STEP = kwargs.pop("time_step", min(0.1, TIME_TOLERANCE))
    CHECK_SIZE = kwargs.pop("check_size", 10)
    
    next_index = 0
    total_visit_num = len(workload)
    total_req_num = sum(len(v) for _, v in workload)
    skip_offset = 0
    finish_num = 0
    start_timestamp = time.time()
    
    init_task(task_id, total_req_num, start_timestamp)
    
    while True:
        # Launch new tasks
        cur_offset = time.time() - start_timestamp + skip_offset
        logging.debug(f"current offset {cur_offset}")

        if next_index < total_visit_num:
            assert workload[next_index][0] is not None
            logging.debug(f"next visit {next_index} scheduled at {workload[next_index][0]}")
            launch_immediately = False
            
            if cur_offset - workload[next_index][0] > TIME_TOLERANCE:
                logging.warning(
                    f"<{task_id[:4]}>: visit {next_index} cannot be executed in time, late {cur_offset - workload[next_index][0]}."
                )
                launch_immediately = True
            
            if abs(cur_offset - workload[next_index][0]) < TIME_TOLERANCE or launch_immediately:
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

        # Recycle finished tasks & check if system is idle & check if all visits are done
        if tasks:
            to_remove = []
            unfinished_count = 0
            
            for i in range(min(CHECK_SIZE, len(tasks))):
                if tasks[i][1].done():
                    finish_num += 1
                    logging.info(
                        f"visit <{task_id[:4]}:{tasks[i][0]}> done. Total {finish_num}/{total_visit_num} visits done."
                    )
                    responses.append((tasks[i][0], tasks[i][1].result()))
                    to_remove.append(i)
                else:
                    unfinished_count += 1
            
            for i in reversed(to_remove):
                tasks.pop(i)
            
            logging.debug(f"finished {len(to_remove)} tasks, {unfinished_count} tasks not finished.")
        else:
            next_time = workload[next_index][0] if next_index < total_visit_num else None
            
            if SKIP_IDLE_MIN and next_time:
                skip = max(0, next_time - cur_offset - 10 * TIME_STEP - SKIP_IDLE_MIN)
                if skip > 0:
                    skip_offset += skip
                    logging.info(f"<{task_id[:4]}>: skip idle time {skip}, skip_offset now {skip_offset}")
            
            if next_index == total_visit_num:
                break

        if TIME_STEP != 0:
            await asyncio.sleep(TIME_STEP)

    # Sort responses
    responses.sort(key=lambda x: x[0])
    mark_finish_for_task(task_id, time.time())
    
    return [response[1] for response in responses]

if __name__ == "__main__":
    from ..workload_datasets.arena import ArenaDataset
    from ..workload_datasets.oasst1 import Oasst1Dataset
    from ..setup_logger import setup_logger
    from rich import print as rprint

    setup_logger(level=logging.DEBUG)

    conf = {
        "api_base": "http://3.14.115.113:8000/v1",
        "api_key": "EMPTY",
        "model": "vicuna-7b-v1.3",
    }
    # dataset = ArenaDataset()
    dataset = Oasst1Dataset()
    workloads = dataset.to_workload()[:100]
    # offest = [0]+[w[0] for w in workloads]
    # offests_delta = [offest[i+1]-offest[i] for i in range(len(offest)-1)]
    # rprint(offests_delta)
    responses = asyncio.run(
        sim_workload_in_single_thread(workloads, None, skip_idle_min=0.5, **conf)
    )

    # rprint(sample_visit)
    # rprint(responses)
    result = [(w, r) for w, r in zip(workloads, responses)]
    rprint(result, file=open("tmp_100.log", "w"))
