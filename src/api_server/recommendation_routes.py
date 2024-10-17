from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Literal, Optional
from .db import db_get_group_test_results, get_hardware_info_with_cost
import logging

router = APIRouter(prefix="/recommendation", tags=["recommendation"])

class PerformanceMetric(BaseModel):
    metric: Literal["ttft", "tpot", "latency", "rps", "throughput"]
    target: float

class RecommendationRequest(BaseModel):
    group_id: str
    performance_metric: PerformanceMetric

class RecommendationResponse(BaseModel):
    most_recommended: Optional[str]
    resource_efficient: Optional[str]
    performance_priority: Optional[str]
    cost_efficient: Optional[str]

def calculate_score(target: float, actual: float, gpu_cost: int, is_paid_engine: bool, metric: str) -> float:
    if metric in ["ttft", "tpot", "latency"]:
        # For these metrics, lower is better
        diff = abs(target - actual)
        performance_score = max(0, 1 - (diff / target))
    else:  # "rps", "throughput"
        # For these metrics, higher is better
        performance_score = min(1, actual / target)

    cost_score = 1 / (gpu_cost + 1)  # +1 to avoid division by zero
    engine_score = 0 if is_paid_engine else 0.2
    return performance_score * 0.5 + cost_score * 0.3 + engine_score * 0.2

def is_paid_engine(endpoint_type: str) -> bool:
    return endpoint_type in ["openai", "friendliai"]  # Add other paid engines as needed

def get_metric_value(result: Dict, metric: str) -> float:
    if metric == "ttft":
        return result['TTFT']['avg']
    elif metric == "tpot":
        return result['TPOT']['avg']
    elif metric == "latency":
        return result['latency']['avg']
    elif metric == "rps":
        return result['RPS']
    elif metric == "throughput":
        return result['Throughput']['avg']
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_servers(request: RecommendationRequest):
    logging.info(f"Received recommendation request for group: {request.group_id}")
    group_results = db_get_group_test_results(request.group_id)
    
    if not group_results:
        logging.warning(f"No results found for group '{request.group_id}'")
        raise HTTPException(status_code=404, detail=f"No results found for group '{request.group_id}'")

    logging.info(f"Found {len(group_results)} tests in the group")

    recommendations = {
        "most_recommended": None,
        "resource_efficient": None,
        "performance_priority": None,
        "cost_efficient": None
    }

    best_score = float('-inf')
    best_performance = float('-inf') if request.performance_metric.metric in ["rps", "throughput"] else float('inf')
    best_resource_efficiency = float('-inf')
    best_cost_efficiency = float('-inf')

    valid_tests = []

    for test in group_results:
        test_id = test['id']
        config = test['config']
        result = test['result']
        
        logging.info(f"Processing test: {test_id}")
        
        if not result:
            logging.warning(f"No result found for test: {test_id}")
            continue

        hardware_info = get_hardware_info_with_cost(test_id)
        if not hardware_info:
            logging.warning(f"No hardware info found for test: {test_id}")
            continue

        try:
            actual_value = get_metric_value(result, request.performance_metric.metric)
        except KeyError as e:
            logging.warning(f"Missing metric {request.performance_metric.metric} for test {test_id}: {e}")
            continue

        logging.info(f"Test {test_id} is valid. Metric value: {actual_value}")
        valid_tests.append(test_id)

        gpu_cost = hardware_info['gpu_cost']
        is_paid = is_paid_engine(config['endpoint_type'])

        # Most recommended
        score = calculate_score(request.performance_metric.target, actual_value, gpu_cost, is_paid, request.performance_metric.metric)
        if score > best_score:
            best_score = score
            recommendations["most_recommended"] = test_id

        # Performance priority
        if request.performance_metric.metric in ["rps", "throughput"]:
            if actual_value > best_performance:
                best_performance = actual_value
                recommendations["performance_priority"] = test_id
        else:
            if actual_value < best_performance:
                best_performance = actual_value
                recommendations["performance_priority"] = test_id

        # Resource efficiency
        resource_efficiency = calculate_score(request.performance_metric.target, actual_value, gpu_cost, False, request.performance_metric.metric)
        if resource_efficiency > best_resource_efficiency:
            best_resource_efficiency = resource_efficiency
            recommendations["resource_efficient"] = test_id

        # Cost efficiency (only for vllm and TGI)
        if not is_paid:
            cost_efficiency = calculate_score(request.performance_metric.target, actual_value, gpu_cost, False, request.performance_metric.metric)
            if cost_efficiency > best_cost_efficiency:
                best_cost_efficiency = cost_efficiency
                recommendations["cost_efficient"] = test_id

    if not valid_tests:
        logging.error("No valid tests found in the group")
        raise HTTPException(status_code=404, detail="No valid tests found in the group")

    logging.info(f"Valid tests: {valid_tests}")
    logging.info(f"Recommendations: {recommendations}")

    # If there's only one test, use it for all recommendations
    if len(valid_tests) == 1:
        single_test_id = valid_tests[0]
        for key in recommendations:
            recommendations[key] = single_test_id

    return RecommendationResponse(**recommendations)

def get_metric_value(result: Dict, metric: str) -> float:
    try:
        if metric == "ttft":
            return result['TTFT']['avg']
        elif metric == "tpot":
            return result['TPOT']['avg']
        elif metric == "latency":
            return result['latency']['avg']
        elif metric == "rps":
            return result['RPS']
        elif metric == "throughput":
            return result['Throughput']['avg']
        else:
            raise ValueError(f"Unknown metric: {metric}")
    except KeyError as e:
        logging.error(f"Failed to get metric value: {e}")
        raise