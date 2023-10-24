from ..Datasets.protocol import Visit, VisitCtx
from .protocol import ReqResponse, VisitResponse
from typing import List, Tuple
from ..API.openai import streaming_inference
from ..API.api_protocol import ResPiece
import time
import asyncio
import logging
import time


async def sim_visit(visit: Visit, **kwargs) -> VisitResponse:
    """
    Simulate a visit and return the responses.
    """
    visit_start_time = time.time()
    ctx: VisitCtx = dict()
    responses: List[ReqResponse] = []
    last_finish = None
    HUMAN_INPUT_WPS = kwargs.pop("human_input_wps", 1.0)
    TIME_TOLERANCE = kwargs.pop("time_tolerance", 0.1)
    logging.debug(f"<sim_visit>: launch visit sim, size of dialog {len(visit)}.")
    for scheduled_offest, sim_req in visit:
        launch_latency = 0.0
        dialog = sim_req.messages(ctx)
        res_loggings: List[Tuple[float, ResPiece]] = []
        ret_str = None
        inference_conf = sim_req.shadow_params(**kwargs)
        if scheduled_offest is not None:
            scheduled_time = visit_start_time + scheduled_offest
            if scheduled_time - time.time() > -TIME_TOLERANCE:
                logging.debug(
                    f"<{sim_req.id}>: wait for scheduled time, after {scheduled_time - time.time()}."
                )
                await asyncio.sleep(scheduled_time - time.time())
            else:
                logging.warning(
                    f"<{sim_req.id}>: Request cannot be executed in time, {time.time() - scheduled_time} s late, launch immediately."
                )
        elif last_finish is not None:
            sim_time = len(dialog[-1]["content"].split()) / HUMAN_INPUT_WPS
            logging.debug(f"<{sim_req.id}>: simulate human input, after {sim_time} s.")
            await asyncio.sleep(sim_time)
        else:
            logging.debug(f"<{sim_req.id}>: launch immediately.")
        req_start_time = time.time()
        logging.debug(f"<{sim_req.id}>: start inference.")
        try:
            assert (
                inference_conf["model"] is not None
            ), "<sim_visit>: model must be specified"
            if sim_req.stream:
                async for res_piece in streaming_inference(dialog, **inference_conf):
                    if isinstance(res_piece, Exception):
                        raise res_piece
                    res_loggings.append((time.time(), res_piece))
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
            last_finish = time.time()
            ctx[sim_req.id] = ret_str
            responses.append(
                ReqResponse(
                    req_id=sim_req.id,
                    start_timestamp=req_start_time,
                    end_timestamp=last_finish,
                    dialog=dialog + [{"role": "assitant", "content": ret_str}],
                    loggings=res_loggings,
                    launch_latency=launch_latency,
                )
            )
        except Exception as e:
            logging.warning(f"<sim_visit>: exception caught, visit failed: {str(e)}")
            import traceback

            responses.append(
                ReqResponse(
                    req_id=sim_req.id,
                    start_timestamp=req_start_time,
                    end_timestamp=time.time(),
                    launch_latency=launch_latency,
                    error_info=(str(e), traceback.format_exc()),
                )
            )
            break
    return VisitResponse(
        start_timestamp=visit_start_time,
        end_timestamp=time.time(),
        responses=responses,
        failed=responses[-1].error_info is not None,
    )


if __name__ == "__main__":
    from ..Datasets.protocol import Visit, VisitCtx
    from ..Datasets.arena import ArenaDataset

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
    responses = asyncio.run(sim_visit(sample_visit, **conf))
    from rich import print as rprint

    rprint(sample_visit)
    rprint(responses)
