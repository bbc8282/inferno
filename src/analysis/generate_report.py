from ..simulate.protocol import VisitResponse, ReqResponse
from .report import VisitLevelReport, RequestLevelReport
from typing import List
import numpy as np
import bisect

def load_tokenizer(tokenizer_name: str, hf_auth_key: str = None):
    from transformers import AutoTokenizer
    print(f"load tokenizer {tokenizer_name}")
    if hf_auth_key:
        return AutoTokenizer.from_pretrained(tokenizer_name, token=hf_auth_key)
    else:
        return AutoTokenizer.from_pretrained(tokenizer_name)

def count_tokens_from_str(s: str, tokenizer, tokenizer_name: str) -> int:
    return len(tokenizer(s, return_tensors="np")["input_ids"][0])

def generate_request_level_report(
    ress: List[ReqResponse], tokenizer_name: str, **kwargs) -> RequestLevelReport:
    hf_auth_key = kwargs.pop("hf_auth_key", None)
    tokenizer = load_tokenizer(tokenizer_name, hf_auth_key)
    
    success = [res for res in ress if res.error_info is None]
    assert len(success) > 0, "all requests failed, cannot generate report."
    
    TTFT = [res.loggings[0][0] - res.start_timestamp for res in success if res.loggings]
    start_on_time = [res.launch_latency == 0.0 for res in success]
    time_per_request = [res.end_timestamp - res.start_timestamp for res in success]
    token_per_request = []
    token_timestamp = []
    
    for c in ress:
        count = 0
        for pack in c.loggings:
            if len(pack) > 1 and pack[1].content:
                num = count_tokens_from_str(pack[1].content, tokenizer, tokenizer_name)
                count += num
                token_timestamp.append((pack[0], num))
        if c.error_info is None:
            token_per_request.append(count)
    
    assert token_timestamp, "all requests failed, cannot generate report."
    
    token_timestamp.sort(key=lambda x: x[0])
    throughput_windows = kwargs.get("throughput_windows", 5)
    throughput_step = kwargs.get("throughput_step", 0.5)
    count_list = np.zeros(
        int((token_timestamp[-1][0] - token_timestamp[0][0]) / throughput_step) + 1
    )
    
    for t, c in token_timestamp:
        count_list[int((t - token_timestamp[0][0]) / throughput_step)] += c
    
    sample_list = np.zeros(len(count_list))
    for i in range(len(sample_list)):
        ty = token_timestamp[0][0] + i * throughput_step
        sample_list[i] = bisect.bisect_right(
            token_timestamp, ty + throughput_windows / 2, key=lambda x: x[0]
        ) - bisect.bisect_left(
            token_timestamp, ty - throughput_windows / 2, key=lambda x: x[0]
        )
    
    sample_list = sample_list / throughput_windows
    TPOT: List[float] = []
    
    if len(time_per_request) != len(token_per_request):
        raise ValueError("Time per request and token per request lists are of different lengths.")
    
    for ti, to in zip(time_per_request, token_per_request):
        if to == 0:
            TPOT.append(0)
        else:
            TPOT.append(ti / to)
    
    if not ress:
        raise ValueError("The list of responses is empty.")
    
    total_duration = max(res.end_timestamp for res in ress) - min(res.start_timestamp for res in ress)
    rps = len(ress) / total_duration if total_duration > 0 else 0
    
    return RequestLevelReport(
        request_num=len(ress),
        fail_rate=1 - len(success) / len(ress),
        TTFT=TTFT,
        latency=[res.end_timestamp - res.start_timestamp for res in success],
        SLO=len(start_on_time) / len(ress),
        time_per_request=time_per_request,
        token_per_request=token_per_request,
        token_timestamp=token_timestamp,
        TPOT=TPOT,
        total_tps_list=sample_list,
        Throughput=np.max(sample_list),
        tokenizer_name=tokenizer_name,
        total_duration=total_duration,
        rps=rps,
    )


def generate_visit_level_report(
    ress: List[VisitResponse], tokenizer_name: str
) -> VisitLevelReport:
    return VisitLevelReport(
        visit_num=len(ress),
        fail_rate=1 - len([res for res in ress if res.failed is False]) / len(ress),
        time_usage_per_visit=[res.end_timestamp - res.start_timestamp for res in ress],
        request_level_report=generate_request_level_report(
            sum([v.responses for v in ress], []),
            tokenizer_name,
        ),
    )


def generate(
    ress: List[VisitResponse], tokenizer_name: str, report_level: str
) -> VisitLevelReport | RequestLevelReport:
    assert report_level in ["visit", "request"]
    if report_level == "visit":
        return generate_visit_level_report(ress, tokenizer_name)
    else:
        for res in ress:
            assert len(res.responses) == 1
        return generate_request_level_report(
            [v.responses[0] for v in ress], tokenizer_name
        )


if __name__ == "__main__":
    import pickle
    from rich import print as rprint

    path = "tmp/responses_single_request.pkl"
    data: List[VisitResponse] = pickle.load(open(path, "rb"))
    rprint(
        generate_visit_level_report(
            data, "meta-llama/Llama-2-7b-chat-hf"
        ).show_as_dict()
    )
