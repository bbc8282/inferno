from typing import List, Tuple
from .protocol import Workload, Visit, SimReq, OpenAIMessage
from .utils import key_timestamp_to_offset, cache, compress_workload
import logging

class DollyDataset:
    def __init__(self, hf_auth_key: str = None):
        from datasets import load_dataset
        try:
            self.raw = load_dataset("databricks/databricks-dolly-15k", use_auth_token=hf_auth_key)
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")

    @cache()
    def to_workload(self, separate_req_in_one_visit_with_interval=None, **kwargs) -> Workload:
        def parse_simreq(d, i) -> SimReq:
            separate_req_in_one_visit = (separate_req_in_one_visit_with_interval is not None)
            return SimReq(
                id=f'dolly-{i}',
                dep_id=None,
                messages_with_dep=[
                    OpenAIMessage(
                        role="user",
                        content=f"{d['instruction']}\n{d['context']}",
                        dep_id=None,
                    ),
                    OpenAIMessage(
                        role="assistant",
                        content=d["response"] if separate_req_in_one_visit else None,
                        dep_id=None if separate_req_in_one_visit else f'dolly-{i}',
                    ),
                ],
                stream=True,
                model=None,
                n=1,
                temperature=kwargs.get("temperature", 1),
                top_p=kwargs.get("top_p", 1),
                max_tokens=kwargs.get("max_tokens", None),
            )

        compression_ratio = kwargs.pop("compression_ratio", 1.0)

        if separate_req_in_one_visit_with_interval is None:
            def parse_visit(d) -> Visit:
                return [(None, parse_simreq(d, i)) for i in range(len(self.raw["train"]))]

            return compress_workload(
                key_timestamp_to_offset([(None, parse_visit(d)) for d in self.raw["train"]]),
                compression_ratio,
            )
        else:
            def parse_timestamped_visits(d) -> List[Tuple[float, Visit]]:
                return [
                    (separate_req_in_one_visit_with_interval * i, [(None, parse_simreq(d, i))])
                    for i in range(len(self.raw["train"]))
                ]

            return compress_workload(
                key_timestamp_to_offset(
                    sum([parse_timestamped_visits(d) for d in self.raw["train"]], [])
                ),
                compression_ratio,
            )

    def dialogs(self) -> List[str]:
        return [f"{d['instruction']}\n{d['context']}" for d in self.raw["train"]]

if __name__ == "__main__":
    from rich import print as rprint
    from .utils import assert_visit_is_legal
    import time

    start_time = time.time()
    ds = DollyDataset()
    ds_workload = ds.to_workload()
    end_time = time.time()
    rprint(ds_workload[0])
    for d in ds_workload:
        if len(d[1]) > 1:
            rprint(d)
            break
    for d in ds_workload:
        assert_visit_is_legal(d[1])
    ds_workload_sep = ds.to_workload(separate_req_in_one_visit_with_interval=60)
    rprint(ds_workload_sep[0])
    for d in ds_workload_sep:
        if len(d[1][0][1].messages_with_dep) > 1:
            rprint(d)
            break
    for d in ds_workload_sep:
        assert_visit_is_legal(d[1])
    print(f"load time: {end_time - start_time}")
    print(f"Time used: {time.time() - start_time}")