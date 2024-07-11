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
    """
    visit_start_time = time.time()
    ctx: VisitCtx = dict()
    responses: List[ReqResponse] = []
    logging.debug(
        f"<sim_visit {visit_index}>: launch visit sim, size of dialog {len(visit)}."
    )
    for scheduled_offset, sim_req in visit:
        dialog = sim_req.messages(ctx)
        res_loggings: List[Tuple[float, ResPiece]] = []
        inference_conf = sim_req.shadow_params(**kwargs)
        
        if scheduled_offset is not None:
            scheduled_time = visit_start_time + scheduled_offset
            current_time = time.time()
            if current_time < scheduled_time:
                await asyncio.sleep(scheduled_time - current_time)
            
        req_start_time = time.time()
        launch_latency = max(0, req_start_time - (visit_start_time + scheduled_offset)) if scheduled_offset is not None else 0
        
        logging.debug(f"<{sim_req.id}>: start inference.")
        init_request(task_id, visit_index, sim_req.id, req_start_time, launch_latency)
        ret_str = ""
        try:
            assert (
                inference_conf["model"] is not None
            ), f"<sim_visit {visit_index}>: model must be specified"
            if sim_req.stream:
                if endpoint_type == "friendliai":
                    # Use SSE (Use Function Call)
                    streaming_func = await get_friendliai_streaming_inference()
                    async for res_piece in streaming_func(dialog, **inference_conf):
                        if isinstance(res_piece, Exception):
                            raise res_piece
                        res_loggings.append((time.time(), res_piece))
                        if res_piece.content:
                            log_new_pack(
                                task_id,
                                visit_index,
                                sim_req.id,
                                time.time(),
                                res_piece.content,
                            )
                else:
                    # Do not use SSE (Use Coroutine)
                    async for res_piece in get_streaming_inference(endpoint_type)(
                        dialog, **inference_conf
                    ):
                        if isinstance(res_piece, Exception):
                            raise res_piece
                        res_loggings.append((time.time(), res_piece))
                        if res_piece.content:
                            log_new_pack(
                                task_id,
                                visit_index,
                                sim_req.id,
                                time.time(),
                                res_piece.content,
                            )
                ret_str = "".join(
                    [
                        p[1].content
                        for p in res_loggings
                        if p[1].content is not None and p[1].index == 0
                    ]
                )
            else:
                raise NotImplementedError
            logging.debug(f"<{sim_req.id}>: finish inference.")
            end_time = time.time()
            ctx[sim_req.id] = ret_str
            responses.append(
                ReqResponse(
                    req_id=sim_req.id,
                    start_timestamp=req_start_time,
                    end_timestamp=end_time,
                    dialog=dialog + [{"role": "assistant", "content": ret_str}],
                    loggings=res_loggings,
                    launch_latency=launch_latency,
                )
            )
            mark_success_for_request(task_id, visit_index, sim_req.id, end_time)
        except Exception as e:
            import traceback

            infos = traceback.format_exc()
            logging.warning(
                f"<sim_visit {visit_index}>: exception caught, visit failed: {str(e)}: {infos}"
            )

            exit_time = time.time()

            responses.append(
                ReqResponse(
                    req_id=sim_req.id,
                    start_timestamp=req_start_time,
                    end_timestamp=exit_time,
                    dialog=dialog + [{"role": "assistant", "content": ret_str}],
                    loggings=res_loggings,
                    launch_latency=launch_latency,
                    error_info=(str(e), infos),
                )
            )
            mark_error_for_request(task_id, visit_index, sim_req.id, exit_time, str(e))
            break
    return VisitResponse(
        start_timestamp=visit_start_time,
        end_timestamp=time.time(),
        responses=responses,
        failed=responses[-1].error_info is not None,
    )

if __name__ == "__main__":
    from ..workload_datasets.protocol import Visit, VisitCtx
    from ..workload_datasets.arena import ArenaDataset

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    conf = {
        "api_base": "http://3.14.115.113:8000/v1",
        "api_key": "EMPTY",
        "model": "vicuna-7b-v1.3",
    }
    dataset = ArenaDataset()
    workloads = dataset.to_workload()
    for w in workloads:
        if len(w[1]) > 1:
            sample_visit = w[1]
            break
    responses = asyncio.run(sim_visit(sample_visit, 0, "example", **conf))
    from rich import print as rprint

    rprint(sample_visit)
    rprint(responses)
