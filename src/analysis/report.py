from attrs import define
from typing import List, Tuple
import numpy as np


@define
class RequestLevelReport:
    request_num: int
    fail_rate: float

    TTFT: List[float]  # Time To First Token (TTFT)
    latency: List[float]  # Time cost from request to last response
    SLO: float  # Service Level Objective
    time_per_request: List[float]
    token_per_request: List[int]
    token_timestamp: List[Tuple[float, int]]
    TPOT: List[float]  # Time Per Output Token (avg for each request)
    Throughput: float
    tokenizer_name: str
    total_tps_list: List[float]
    total_duration: float
    rps: float

    def show_as_dict(self):
        return {
            "Total_request_num": self.request_num,
            "Fail_rate": self.fail_rate,
            "TTFT": {
                "min": np.min(self.TTFT),
                "max": np.max(self.TTFT),
                "avg": np.mean(self.TTFT),
                "std": np.std(self.TTFT),
            },
            "SLO": self.SLO,
            "TPOT": {
                "min": np.min(self.TPOT),
                "max": np.max(self.TPOT),
                "avg": np.mean(self.TPOT),
                "std": np.std(self.TPOT),
            },
            "Throughput_token_per_sec": self.Throughput,
            "Total_duration_sec": self.total_duration,
            "RPS": self.rps,
        }

    def visualize(self):
        raise NotImplementedError


@define
class VisitLevelReport:
    visit_num: int
    fail_rate: float
    time_usage_per_visit: List[float]
    request_level_report: RequestLevelReport

    def show_as_dict(self):
        return {
            "visit_num": self.visit_num,
            "fail_rate": self.fail_rate,
            "time_usage_per_visit": self.time_usage_per_visit,
            "request_level_report": self.request_level_report.show_as_dict(),
        }

    def visualize(self):
        raise NotImplementedError
