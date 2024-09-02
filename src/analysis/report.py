from attrs import define
from typing import List, Tuple
import numpy as np


@define
class RequestLevelReport:
    request_num: int
    fail_rate: float
    SLO: float  # Service Level Objective
    TTFT: List[float]  # Time To First Token (TTFT)
    latency: List[float]  # Time cost from request to last response
    time_per_request: List[float]
    token_per_request: List[int]
    token_timestamp: List[Tuple[float, int]]
    TPOT: List[float]  # Time Per Output Token (avg for each request)
    Throughput: float
    tokenizer_name: str
    total_tps_list: List[float]
    total_duration: float
    rps: float
    
    def calculate_stable_average_throughput(self, trim_percent=5):
        """
        Calculate the stable average throughput, excluding the initial and final periods.
        
        :param trim_percent: Percentage of values to trim from each end (default 5%)
        :return: Stable average of the throughput values
        """
        values = np.array(self.total_tps_list)
        trim_size = int(len(values) * (trim_percent / 100))
        trimmed_values = values[trim_size:-trim_size]
        return np.mean(trimmed_values)

    def show_as_dict(self):
        return {
            "Total_request_num": self.request_num,
            "Fail_rate": self.fail_rate,
            "SLO": self.SLO,
            "latency": {
                "min": np.min(self.latency),
                "max": np.max(self.latency),
                "avg": np.mean(self.latency),
                "std": np.std(self.latency),
                "95_percentile": np.percentile(self.latency, 95),
            },
            "TTFT": {
                "min": np.min(self.TTFT),
                "max": np.max(self.TTFT),
                "avg": np.mean(self.TTFT),
                "std": np.std(self.TTFT),
                "95_percentile": np.percentile(self.TTFT, 95),
            },
            "TPOT": {
                "min": np.min(self.TPOT),
                "max": np.max(self.TPOT),
                "avg": np.mean(self.TPOT),
                "std": np.std(self.TPOT),
                "95_percentile": np.percentile(self.TPOT, 95),
            },
            "Throughput": {
                "max": self.Throughput,
                "avg": self.calculate_stable_average_throughput(),
                "95_percentile": np.percentile(self.total_tps_list, 95),
            },
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
