from typing import List
from .protocol import Workload, SimReq, OpenAIMessage
from .utils import key_timestamp_to_offset, cache, compress_workload, load_local_dataset
from datetime import datetime
import logging


class Oasst1Dataset:
    @cache()
    def _load(self, hf_auth_key: str = None):
        from datasets import load_dataset

        #raw = load_dataset("OpenAssistant/oasst1", use_auth_token=hf_auth_key)
        raw = load_local_dataset('oasst1')
        merged_raw = list(raw["train"]) + list(raw["validation"])
        dicted_data = {
            v["message_id"]: {
                "timestamp": datetime.fromisoformat(v["created_date"]).timestamp(),
                **v,
            }
            for v in merged_raw
        }
        grouped_data = {}
        for v in dicted_data.values():
            grouped_data[v["message_tree_id"]] = [
                v,
                *grouped_data.get(v["message_tree_id"], []),
            ]
        return dicted_data, grouped_data

    def __init__(self, hf_auth_key: str = None):
        try:
            self.dicted_data, self.grouped_data = self._load(hf_auth_key)
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")

    @cache()
    def to_workload(self, separate_req_in_one_visit=False, **kwargs) -> Workload:
        def get_prompter_id(cur_id):
            if cur_id == None:
                return None
            cur_req = self.dicted_data[cur_id]
            if cur_req["role"] == "prompter":
                return "oasst1-" + cur_id
            else:
                return get_prompter_id(cur_req["parent_id"])

        def parse_simreq(cur_id) -> SimReq:
            def get_messages(cur_id) -> List[OpenAIMessage]:
                if cur_id == None:
                    return []
                cur_req = self.dicted_data[cur_id]
                return get_messages(cur_req["parent_id"]) + [
                    OpenAIMessage(
                        role="user" if cur_req["role"] == "prompter" else "assistant",
                        content=cur_req["text"]
                        if cur_req["role"] == "prompter" or separate_req_in_one_visit
                        else None,
                        dep_id=None
                        if cur_req["role"] == "prompter" or separate_req_in_one_visit
                        else get_prompter_id(cur_req["parent_id"]),
                    )
                ]

            cur_req = self.dicted_data[cur_id]
            return SimReq(
                id=f"oasst1-{cur_id}",
                dep_id=get_prompter_id(cur_req["parent_id"])
                if not separate_req_in_one_visit
                else None,
                messages_with_dep=get_messages(cur_id),
                stream=True,
                model=None,
                n=1,
                temperature=kwargs.get("temperature", 1),
                top_p=kwargs.get("top_p", 1),
                max_tokens=kwargs.get("max_tokens", None),
            )

        compression_ratio = kwargs.pop("compression_ratio", 1.0)

        if not separate_req_in_one_visit:

            def get_visit_start_time(group):
                for v in group:
                    if v["parent_id"] == None:
                        return v["timestamp"]

            unordered_workloads = []
            for group in self.grouped_data.values():
                start_time = get_visit_start_time(group)
                unordered_workloads.append(
                    (
                        start_time,
                        sorted(
                            [
                                (
                                    v["timestamp"] - start_time,
                                    parse_simreq(v["message_id"]),
                                )
                                for v in group
                                if v["role"] == "prompter"
                            ],
                            key=lambda v: v[0],
                        ),
                    )
                )
            return compress_workload(
                key_timestamp_to_offset(unordered_workloads), compression_ratio
            )
        else:
            return compress_workload(
                key_timestamp_to_offset(
                    [
                        (v["timestamp"], [(None, parse_simreq(v["message_id"]))])
                        for v in self.dicted_data.values()
                        if v["role"] == "prompter"
                    ]
                ),
                compression_ratio,
            )

    def dialogs(self) -> List[str]:
        return [v["text"] for v in self.dicted_data.values() if v["role"] == "prompter"]


if __name__ == "__main__":
    from rich import print as rprint
    from .utils import assert_visit_is_legal
    import time

    start_time = time.time()
    ds = Oasst1Dataset()
    ds_workload = ds.to_workload()
    compression_ratio = 0.2
    ds_workload_compressed = ds.to_workload(compression_ratio=compression_ratio)
    assert len(ds_workload_compressed) == len(ds_workload)
    for i in range(len(ds_workload)):
        assert ds_workload_compressed[i][0] == ds_workload[i][0] / compression_ratio
    print([ds_workload[i][0] for i in range(10)])
    print([ds_workload_compressed[i][0] for i in range(10)])
    print(len(ds_workload))
    # rprint(ds.grouped_data['bebaaa50-9c9e-4fee-be15-1ff4d7efea8a'])
    rprint(ds_workload[1])
    for d in ds_workload:
        assert_visit_is_legal(d[1])
    ds_workload_sep = ds.to_workload(separate_req_in_one_visit=True)
    rprint(ds_workload_sep[0])
    for d in ds_workload_sep:
        if len(d[1][0][1].messages_with_dep) > 1:
            rprint(d)
            break
    for d in ds_workload_sep:
        assert_visit_is_legal(d[1])
    print(time.time() - start_time)
