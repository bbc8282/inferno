import sqlite3
import os
from .protocols import TestConfig
from uuid import uuid4
from typing import List, Tuple, Dict, Optional
import time
import datetime
import json
import logging

db_path = "tmp/api_server.db"

GPU_COST_RATIO = {
    'A100': 10,
    'A10': 2,
    'A30': 3,
    'A40': 4,
    'H100': 25,
    'H200': 35
}

if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE test (id text, config text, status text, model text, start_timestamp text, nickname text)"
    )
    cursor.execute("CREATE TABLE error (id text, error_info text)")
    cursor.execute("""
        CREATE TABLE heartbeat (
            worker_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL
        );
        """)
    cursor.execute("""
        CREATE TABLE groups (
            id TEXT PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE group_tests (
            group_id TEXT,
            test_id TEXT,
            FOREIGN KEY (group_id) REFERENCES groups (id),
            FOREIGN KEY (test_id) REFERENCES test (id)
        )
    """)
    cursor.execute("""
        CREATE TABLE hardware_info (
            test_id TEXT PRIMARY KEY,
            gpu_model TEXT CHECK( gpu_model IN ('A100', 'A10', 'A30', 'A40', 'H100', 'H200') ),
            gpu_count INTEGER NOT NULL CHECK( gpu_count > 0 ),
            FOREIGN KEY (test_id) REFERENCES test (id) ON DELETE CASCADE
        )
    """)
    conn.commit()

def report_error(id: str, error_info: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO error VALUES (?, ?)", (id, error_info))
    cursor.execute("UPDATE test SET status=? WHERE id=?", ("error", id))
    conn.commit()


# return a list of (id, nickname, timestamp) from latest to oldest, timestamp is a string in format %Y-%m-%d %H:%M:%S
def get_id_list() -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, nickname, start_timestamp FROM test ORDER BY start_timestamp DESC"
    )
    return [
        (
            id,
            nickname,
            datetime.datetime.fromtimestamp(int(timestamp)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )
        for id, nickname, timestamp in cursor.fetchall()
    ]


def query_error_info(id: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT error_info FROM error WHERE id=?", (id,))
    error_info = cursor.fetchone()
    if error_info is None:
        return f"{id} has no error"
    return error_info[0]


def query_model(id: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT model FROM test WHERE id=?", (id,))
    model = cursor.fetchone()
    if model is None:
        return ""
    return model[0]


def query_config(id: str) -> TestConfig:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT config FROM test WHERE id=?", (id,))
    config = cursor.fetchone()
    if config is None:
        return None
    return TestConfig.model_validate_json(config[0])


def save_config(config: TestConfig) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT id FROM test")
    existing_ids = [row[0] for row in cursor.fetchall()]
    
    if config.test_id in existing_ids:
        id = str(uuid4())
    else:
        id = config.test_id or str(uuid4())
    
    model = config.model
    config_str = config.model_dump_json()
    cursor.execute(
        "INSERT INTO test VALUES (?, ?, ?, ?, ?, ?)",
        (id, config_str, "init", model, str(int(time.time())), ""),
    )
    conn.commit()
    return id

def delete_test(id: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM test WHERE id=?", (id,))
    cursor.execute("DELETE FROM error WHERE id=?", (id,))
    conn.commit()


def set_nickname(id: str, nickname: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE test SET nickname=? WHERE id=?", (nickname, id))
    conn.commit()


def query_nickname(id: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT nickname FROM test WHERE id=?", (id,))
    nickname = cursor.fetchone()
    if nickname is None:
        return f"Cannot find test {id}"
    return nickname[0]


def query_test_status(id: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM test WHERE id=?", (id,))
    status = cursor.fetchone()
    if status is None:
        return f"Cannot find test {id}"
    return status[0]


def set_status(id: str, st: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE test SET status=? WHERE id=?", (st, id))
    conn.commit()
    return "OK"


def set_test_to_pending(id: str) -> str:
    status = query_test_status(id)
    if status is None:
        return f"Cannot find test {id}"
    if status == "running":
        return f"Test {id} is already running"
    return set_status(id, "pending")


def get_all_pending_tests() -> List[Tuple[str, TestConfig]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, config FROM test WHERE status=?", ("pending",))
    return [
        (id, TestConfig.model_validate_json(config_str))
        for id, config_str in cursor.fetchall()
    ]
    
def get_all_worker_ids():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT worker_id FROM heartbeat")
    worker_ids = [row[0] for row in cursor.fetchall()]
    return worker_ids

def update_worker_heartbeat(worker_id: str, timestamp: float):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO heartbeat (worker_id, timestamp) VALUES (?, ?)", (worker_id, timestamp))
    conn.commit()

def get_last_heartbeat(worker_id: str) -> float:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT MAX(timestamp) FROM heartbeat WHERE worker_id=?", (worker_id,))
    row = cur.fetchone()
    return row[0] if row[0] else 0.0

def db_create_group(group_id: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO groups (id) VALUES (?)", (group_id,))
        conn.commit()
        return group_id
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"Group '{group_id}' already exists")

def db_add_tests_to_group(group_id: str, test_ids: List[str]):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        for test_id in test_ids:
            cursor.execute("INSERT INTO group_tests (group_id, test_id) VALUES (?, ?)", (group_id, test_id))
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"An error occurred: {e.args[0]}")
    finally:
        conn.close()

def db_get_all_groups() -> List[Tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM groups")
    return cursor.fetchall()

def db_remove_group(group_id: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM groups WHERE id = ?", (group_id,))
    cursor.execute("DELETE FROM group_tests WHERE group_id = ?", (group_id,))
    conn.commit()

def db_remove_all_groups() -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM groups")
    cursor.execute("DELETE FROM group_tests")
    conn.commit()

def db_check_group_status(group_id: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.status
        FROM group_tests gt
        JOIN test t ON gt.test_id = t.id
        WHERE gt.group_id = ?
    """, (group_id,))
    statuses = [row[0] for row in cursor.fetchall()]
    
    if not statuses:
        return "empty"
    if all(status == "init" for status in statuses):
        return "init"
    if any(status == "running" for status in statuses):
        return "running"
    if all(status == "finish" for status in statuses):
        return "finish"
    return "mixed"

def db_get_group_tests(group_id: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT test_id FROM group_tests WHERE group_id=?", (group_id,))
    tests = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tests

def db_remove_test_from_group(group_id: str, test_id: str) -> bool:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM group_tests WHERE group_id=? AND test_id=?", (group_id, test_id))
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success

def db_get_group_test_results(group_id: str) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.id, t.config, t.status, t.model, t.start_timestamp, t.nickname
        FROM test t
        JOIN group_tests gt ON t.id = gt.test_id
        WHERE gt.group_id = ?
    """, (group_id,))
    tests = cursor.fetchall()
    conn.close()

    results = []
    for test in tests:
        test_id, config, status, model, start_timestamp, nickname = test
        hardware_info = get_hardware_info_with_cost(test_id)
        result = {
            "id": test_id,
            "config": json.loads(config),
            "status": status,
            "model": model,
            "start_timestamp": start_timestamp,
            "nickname": nickname,
            "result": read_test_result(test_id),
            "hardware_info": hardware_info
        }
        logging.debug(f"Test result for {test_id}: {result}")
        results.append(result)

    return results

def read_test_result(test_id: str) -> Optional[Dict]:
    result_file = f"tmp/report_{test_id}.json"
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {result_file}")
            return None
    else:
        print(f"Result file not found: {result_file}")
        return None
    
def calculate_gpu_cost(gpu_model: str, gpu_count: int) -> int:
    return GPU_COST_RATIO.get(gpu_model.upper(), 0) * gpu_count

def get_hardware_info_with_cost(test_id: str) -> Dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT gpu_model, gpu_count
        FROM hardware_info
        WHERE test_id = ?
    """, (test_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        gpu_model, gpu_count = result
        cost = calculate_gpu_cost(gpu_model, gpu_count)
        return {
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "gpu_cost": cost
        }
    else:
        return None

def add_hardware_info(test_id: str, gpu_model: str, gpu_count: int):
    valid_gpu_models = list(GPU_COST_RATIO.keys())
    
    if gpu_model.upper() not in valid_gpu_models:
        raise ValueError(f"Invalid GPU model. Must be one of {valid_gpu_models}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO hardware_info (test_id, gpu_model, gpu_count)
        VALUES (?, ?, ?)
    """, (test_id, gpu_model.upper(), gpu_count))
    
    conn.commit()
    conn.close()