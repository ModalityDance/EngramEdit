import json
import random
import typing
from pathlib import Path

from torch.utils.data import Dataset


class MQUAKEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = None,
        shuffle_seed: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / "MQuAKE-CF-3k-v2.json"

        with open(cf_loc, "r") as f:
            raw = json.load(f)
        data = []
        for i, record in enumerate(raw):
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": record["requested_rewrite"],
                    "paraphrase_prompts": record["questions"],
                    "new_answer": record["new_answer"],
                    "new_answer_alias": record["new_answer_alias"],
                    "answer": record["answer"],
                    "answer_alias": record["answer_alias"],
                    "single_hops": record["single_hops"],
                    "new_single_hops": record["new_single_hops"],
                    "neighborhood_prompts": [],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(data)

        self._data = data[:size]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]
