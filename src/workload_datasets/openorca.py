from typing import List, Tuple
from .protocol import Workload, Visit, SimReq, OpenAIMessage
from .utils import key_timestamp_to_offset, cache, compress_workload, load_local_dataset
import logging
import random

class OpenOrcaDataset:
    def __init__(self, hf_auth_key: str = None):
        from datasets import load_dataset
        try:
            #self.raw = load_dataset("Open-Orca/OpenOrca", use_auth_token=hf_auth_key)
            self.raw = load_local_dataset('openorca')
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")

    @cache()
    def to_workload(self, separate_req_in_one_visit_with_interval=None, **kwargs) -> Workload:
        def parse_simreq(d, i) -> SimReq:
            separate_req_in_one_visit = (separate_req_in_one_visit_with_interval is not None)
            messages = []
            
            # Add system message if available
            if 'system_prompt' in d and d['system_prompt']:
                messages.append(OpenAIMessage(
                    role="system",
                    content=d['system_prompt'],
                    dep_id=None
                ))
            
            # Add user message
            messages.append(OpenAIMessage(
                role="user",
                content=d['question'],
                dep_id=None
            ))
            
            # Add assistant message
            messages.append(OpenAIMessage(
                role="assistant",
                content=d['response'] if separate_req_in_one_visit else None,
                dep_id=None if separate_req_in_one_visit else f'openorca-{i}',
            ))

            return SimReq(
                id=f'openorca-{i}',
                dep_id=None,
                messages_with_dep=messages,
                stream=True,
                model=None,
                n=1,
                temperature=kwargs.get("temperature", 1),
                top_p=kwargs.get("top_p", 1),
                max_tokens=kwargs.get("max_tokens", None),
            )

        compression_ratio = kwargs.pop("compression_ratio", 1.0)
        sample_size = kwargs.pop("sample_size", None)  # Add this line to allow sampling

        # Sample the dataset if sample_size is specified
        if sample_size is not None:
            data = random.sample(list(enumerate(self.raw["train"])), min(sample_size, len(self.raw["train"])))
        else:
            data = enumerate(self.raw["train"])

        if separate_req_in_one_visit_with_interval is None:
            def parse_visit(d, i) -> Visit:
                return [(None, parse_simreq(d, i))]

            return compress_workload(
                key_timestamp_to_offset([(i, parse_visit(d, i)) for i, d in data]),
                compression_ratio,
            )
        else:
            def parse_timestamped_visits(d, i) -> List[Tuple[float, Visit]]:
                return [(separate_req_in_one_visit_with_interval * i, [(None, parse_simreq(d, i))])]

            return compress_workload(
                key_timestamp_to_offset(
                    sum([parse_timestamped_visits(d, i) for i, d in data], [])
                ),
                compression_ratio,
            )

    @cache()
    def dialogs(self) -> List[str]:
        return [d['question'] for d in self.raw["train"]]