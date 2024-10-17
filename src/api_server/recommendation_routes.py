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
        relative_diff = (actual - target) / target
        if relative_diff <= 0:
            # Exceeded target (actual is lower), higher score for being closer to target
            performance_score = 1 + abs(relative_diff)
        else:
            # Did not meet target, lower score
            performance_score = 1 / (1 + relative_diff)
    else:  # "rps", "throughput"
        # For these metrics, higher is better
        relative_diff = (actual - target) / target
        if relative_diff >= 0:
            # Exceeded target, higher score for being further above target
            performance_score = 1 + relative_diff
        else:
            # Did not meet target, lower score
            performance_score = 1 / (1 - relative_diff)

    # Normalize performance_score to be between 0 and 1
    performance_score = min(max(performance_score / 2, 0), 1)

    cost_score = 1 / (gpu_cost + 1)  # +1 to avoid division by zero
    engine_score = 0 if is_paid_engine else 0.2
    
    # Weighted sum of scores
    total_score = performance_score * 0.5 + cost_score * 0.3 + engine_score * 0.2
    
    logging.info(f"Scores - Performance: {performance_score:.2f}, Cost: {cost_score:.2f}, Engine: {engine_score:.2f}, Total: {total_score:.2f}")
    
    return total_score

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

    valid_tests = []
    for test in group_results:
        test_id = test['id']
        if not test['result']:
            logging.warning(f"No result found for test: {test_id}")
            continue
        if not get_hardware_info_with_cost(test_id):
            logging.warning(f"No hardware info found for test: {test_id}")
            continue
        valid_tests.append(test_id)

    if not valid_tests:
        logging.error("No valid tests found in the group")
        raise HTTPException(status_code=404, detail="No valid tests found in the group")

    if len(valid_tests) == 1:
        single_test_id = valid_tests[0]
        logging.info(f"Only one valid test found: {single_test_id}. Using it for all recommendations.")
        return RecommendationResponse(
            most_recommended=single_test_id,
            resource_efficient=single_test_id,
            performance_priority=single_test_id,
            cost_efficient=single_test_id
        )

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

    for test in group_results:
        test_id = test['id']
        if test_id not in valid_tests:
            continue

        config = test['config']
        result = test['result']
        hardware_info = get_hardware_info_with_cost(test_id)
        
        try:
            actual_value = get_metric_value(result, request.performance_metric.metric)
        except KeyError as e:
            logging.warning(f"Missing metric {request.performance_metric.metric} for test {test_id}: {e}")
            continue

        logging.info(f"Processing test: {test_id}. Metric value: {actual_value}")

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

        # Cost efficiency (only for non-paid engines)
        if not is_paid:
            cost_efficiency = calculate_score(request.performance_metric.target, actual_value, gpu_cost, False, request.performance_metric.metric)
            if cost_efficiency > best_cost_efficiency:
                best_cost_efficiency = cost_efficiency
                recommendations["cost_efficient"] = test_id

    logging.info(f"Recommendations: {recommendations}")

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