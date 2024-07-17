from ..workload_datasets.protocol import Visit, VisitCtx
from .protocol import ReqResponse, VisitResponse
from typing import List, Tuple
from ..API.endpoint_interface import get_streaming_inference, get_friendliai_streaming_inference
from ..API.api_protocol import ResPiece
import time
import asyncio
import logging
from .log_to_db import (
    init_request,
    mark_success_for_request,
    mark_error_for_request,
    log_new_pack,
)

async def sim_visit(
    visit: Visit, visit_index: int, task_id: str, endpoint_type: str, **kwargs
) -> VisitResponse:
    """
    Simulate a visit and return the responses.

    This function processes a series of requests within a visit, handling scheduling,
    inference, and error cases. It maintains a context for inter-request dependencies.

    Args:
        visit (Visit): A list of tuples, each containing a scheduled offset and a SimReq object.
        visit_index (int): The index of the current visit in the workload.
        task_id (str): A unique identifier for the current task.
        endpoint_type (str): The type of inference endpoint to use (e.g., "friendliai", "openai").
        **kwargs: Additional keyword arguments to pass to the inference function.

    Returns:
        VisitResponse: An object containing the start and end timestamps of the visit,
                       a list of responses for each request, and a flag indicating if the visit failed.
    """
    visit_start_time = time.time()
    ctx: VisitCtx = dict()
    responses: List[ReqResponse] = []
    
    logging.debug(f"<sim_visit {visit_index}>: launch visit sim, size of dialog {len(visit)}.")
    
    for scheduled_offset, sim_req in visit:
        try:
            response = await process_request(sim_req, visit_start_time, scheduled_offset, ctx, task_id, visit_index, endpoint_type, **kwargs)
            responses.append(response)
            
            if response.error_info:
                break
        except Exception as e:
            logging.error(f"<sim_visit {visit_index}>: Unexpected error: {str(e)}")
            break
    
    return VisitResponse(
        start_timestamp=visit_start_time,
        end_timestamp=time.time(),
        responses=responses,
        failed=any(r.error_info is not None for r in responses),
    )

async def process_request(sim_req, visit_start_time, scheduled_offset, ctx, task_id, visit_index, endpoint_type, **kwargs):
    """
    Process a single request within a visit.

    This function handles the scheduling, inference, and response creation for a single request.

    Args:
        sim_req (SimReq): The request to be processed.
        visit_start_time (float): The start time of the current visit.
        scheduled_offset (float): The scheduled time offset for this request.
        ctx (VisitCtx): The context dictionary for inter-request dependencies.
        task_id (str): The unique identifier for the current task.
        visit_index (int): The index of the current visit.
        endpoint_type (str): The type of inference endpoint to use.
        **kwargs: Additional keyword arguments for the inference function.

    Returns:
        ReqResponse: An object containing the response details and any error information.
    """
    await schedule_request(visit_start_time, scheduled_offset)
    
    req_start_time = time.time()
    launch_latency = calculate_launch_latency(visit_start_time, scheduled_offset, req_start_time)
    
    dialog = sim_req.messages(ctx)
    inference_conf = sim_req.shadow_params(**kwargs)
    
    logging.debug(f"<{sim_req.id}>: start inference.")
    init_request(task_id, visit_index, sim_req.id, req_start_time, launch_latency)
    
    res_loggings: List[Tuple[float, ResPiece]] = []
    
    try:
        ret_str, res_loggings = await perform_inference(sim_req, dialog, inference_conf, task_id, visit_index, endpoint_type)
        
        end_time = time.time()
        ctx[sim_req.id] = ret_str
        
        response = create_success_response(sim_req, req_start_time, end_time, dialog, res_loggings, launch_latency, ret_str)
        mark_success_for_request(task_id, visit_index, sim_req.id, end_time)
        
        return response
    except Exception as e:
        return handle_inference_error(e, sim_req, req_start_time, dialog, res_loggings, launch_latency, task_id, visit_index)

async def schedule_request(visit_start_time, scheduled_offset):
    """
    Schedule the request based on its offset from the visit start time.

    If the current time is before the scheduled time, this function will wait
    until the scheduled time before proceeding.

    Args:
        visit_start_time (float): The start time of the current visit.
        scheduled_offset (float): The scheduled time offset for this request.
    """
    if scheduled_offset is not None:
        scheduled_time = visit_start_time + scheduled_offset
        current_time = time.time()
        if current_time < scheduled_time:
            await asyncio.sleep(scheduled_time - current_time)

def calculate_launch_latency(visit_start_time, scheduled_offset, req_start_time):
    """
    Calculate the launch latency for a request.

    Launch latency is the difference between when a request was scheduled to start
    and when it actually started.

    Args:
        visit_start_time (float): The start time of the current visit.
        scheduled_offset (float): The scheduled time offset for this request.
        req_start_time (float): The actual start time of the request.

    Returns:
        float: The calculated launch latency.
    """
    return max(0, req_start_time - (visit_start_time + scheduled_offset)) if scheduled_offset is not None else 0

async def perform_inference(sim_req, dialog, inference_conf, task_id, visit_index, endpoint_type):
    """
    Perform the inference for a request.

    This function handles both streaming and non-streaming inference,
    logging each piece of the response as it's received.

    Args:
        sim_req (SimReq): The request to be processed.
        dialog (List[Dict]): The dialog context for this request.
        inference_conf (Dict): Configuration for the inference.
        task_id (str): The unique identifier for the current task.
        visit_index (int): The index of the current visit.
        endpoint_type (str): The type of inference endpoint to use.

    Returns:
        Tuple[str, List]: A tuple containing the full response string and a list of response pieces.

    Raises:
        ValueError: If the model is not specified in the inference configuration.
        NotImplementedError: If non-streaming inference is requested.
        Exception: If any error occurs during inference.
    """
    if "model" not in inference_conf or inference_conf["model"] is None:
        raise ValueError("Model must be specified in the inference configuration")
    
    res_loggings: List[Tuple[float, ResPiece]] = []
    ret_str = ""
    
    if sim_req.stream:
        if endpoint_type == "friendliai":
            streaming_func = await get_friendliai_streaming_inference()
        else:
            streaming_func = get_streaming_inference(endpoint_type)
        
        async for res_piece in streaming_func(dialog, **inference_conf):
            if isinstance(res_piece, Exception):
                raise res_piece
            res_loggings.append((time.time(), res_piece))
            if res_piece.content:
                log_new_pack(task_id, visit_index, sim_req.id, time.time(), res_piece.content)
        
        ret_str = "".join([p[1].content for p in res_loggings if p[1].content is not None and p[1].index == 0])
    else:
        raise NotImplementedError("Non-streaming inference is not implemented")
    
    logging.debug(f"<{sim_req.id}>: finish inference.")
    return ret_str, res_loggings

def create_success_response(sim_req, req_start_time, end_time, dialog, res_loggings, launch_latency, ret_str):
    return ReqResponse(
        req_id=sim_req.id,
        start_timestamp=req_start_time,
        end_timestamp=end_time,
        dialog=dialog + [{"role": "assistant", "content": ret_str}],
        loggings=res_loggings,
        launch_latency=launch_latency,
    )

def handle_inference_error(e, sim_req, req_start_time, dialog, res_loggings, launch_latency, task_id, visit_index):
    import traceback
    
    infos = traceback.format_exc()
    logging.warning(f"<{sim_req.id}>: exception caught, request failed: {str(e)}: {infos}")
    
    exit_time = time.time()
    
    error_response = ReqResponse(
        req_id=sim_req.id,
        start_timestamp=req_start_time,
        end_timestamp=exit_time,
        dialog=dialog + [{"role": "assistant", "content": ""}],
        loggings=res_loggings,
        launch_latency=launch_latency,
        error_info=(str(e), infos),
    )
    
    mark_error_for_request(task_id, visit_index, sim_req.id, exit_time, str(e))
    return error_response