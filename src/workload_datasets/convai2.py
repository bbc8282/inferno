from typing import List, Tuple
from .protocol import Workload, Visit, SimReq, OpenAIMessage
from .utils import key_timestamp_to_offset, cache, compress_workload
import logging

class ConvAI2Dataset:
    def __init__(self, hf_auth_key: str = None):
        from datasets import load_dataset
        try:
            self.raw = load_dataset("conv_ai_2", use_auth_token=hf_auth_key)
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")

    @cache()
    def to_workload(self, separate_req_in_one_visit_with_interval=None, **kwargs) -> Workload:
        def parse_simreq(d, i) -> SimReq:
            separate_req_in_one_visit = (separate_req_in_one_visit_with_interval is not None)
            messages = []
            for j, utterance in enumerate(d['dialog']):
                role = "user" if j % 2 == 0 else "assistant"
                messages.append(OpenAIMessage(
                    role=role,
                    content=utterance if role == "user" or separate_req_in_one_visit else None,
                    dep_id=None if role == "user" or separate_req_in_one_visit else f'convai2-{i}-{j-1}',
                ))

            return SimReq(
                id=f'convai2-{i}-{len(d["dialog"])-1}',
                dep_id=f'convai2-{i}-{len(d["dialog"])-3}' if len(d["dialog"]) > 1 and not separate_req_in_one_visit else None,
                messages_with_dep=messages,
                stream=True,
                model=None,
                n=1,
                temperature=kwargs.get("temperature", 1),
                top_p=kwargs.get("top_p", 1),
                max_tokens=kwargs.get("max_tokens", None),
            )

        compression_ratio = kwargs.pop("compression_ratio", 1.0)

        if separate_req_in_one_visit_with_interval is None:
            def parse_visit(d, i) -> Visit:
                return [(None, parse_simreq(d, i))]

            return compress_workload(
                key_timestamp_to_offset([(i, parse_visit(d, i)) for i, d in enumerate(self.raw["train"])]),
                compression_ratio,
            )
        else:
            def parse_timestamped_visits(d, i) -> List[Tuple[float, Visit]]:
                return [
                    (separate_req_in_one_visit_with_interval * j, [(None, parse_simreq(d, i))])
                    for j in range(len(d['dialog']) // 2)
                ]

            return compress_workload(
                key_timestamp_to_offset(
                    sum([parse_timestamped_visits(d, i) for i, d in enumerate(self.raw["train"])], [])
                ),
                compression_ratio,
            )

    def dialogs(self) -> List[str]:
        return [" ".join(d['dialog'][::2]) for d in self.raw["train"]]

if __name__ == "__main__":
    from rich import print as rprint
    from .utils import assert_visit_is_legal
    import time

    start_time = time.time()
    ds = ConvAI2Dataset()
    ds_workload = ds.to_workload()
    end_time = time.time()
    print(len(ds_workload))
    rprint(ds_workload[0])
    for d in ds_workload:
        assert_visit_is_legal(d[1])
    ds_workload_sep = ds.to_workload(separate_req_in_one_visit_with_interval=60)
    rprint(ds_workload_sep[0])
    for d in ds_workload_sep:
        assert_visit_is_legal(d[1])
    print(f"load time: {end_time - start_time}")
    print(f"Time used: {time.time() - start_time}")