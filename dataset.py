import contextlib
import os
from typing import Callable, IO, Optional, Union
from datetime import datetime

import torch
from bs4 import BeautifulSoup
import msg_parser
from torch.utils.data import Dataset
from pathlib import Path
from transformers import GPT2Tokenizer
import lxml

import args


class Pipeline:
    def __init__(self, *steps: Callable[[str], str]):
        self.steps = steps

    def __call__(self, start: str) -> str:
        current = start
        for step in self.steps:
            current = step(current)
        return current


def _cutout(raw: str) -> str:
    start = raw.find(",") + 1
    end = raw.find("be kind!") + len("be kind!")
    return raw[start:end]


_clean = Pipeline(
    lambda r: r.strip(),  # remove weirdness
    lambda r: ''.join([i if ord(i) < 128 else '' for i in r]),  # remove non unicode characters
    _cutout
)

Openable = Union[IO, os.PathLike]


class YLTDataset(Dataset):
    """
    I want to throw myself into a fire
    """
    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"
    PAD = "<|pad|>"

    def __init__(self, raw_emails: list[Openable], *,
                 limit: Optional[tuple[datetime, datetime]] = None):
        super(YLTDataset, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2',
            bos_token=self.SOT,
            eos_token=self.EOT,
            pad_token=self.PAD,
            return_token_type_ids=False,
        )

        if limit:
            assert limit[0] < limit[1]
        self.limit = limit

        self._preprocess(raw_emails)
        self._tokenize()

    @classmethod
    def from_dir(cls, root: str, *args, **kwargs):
        p = Path(root)
        assert p.is_dir()
        return cls(list(p.iterdir()), *args, **kwargs)

    def to_cache(self):
        with open(".ylt_cache.txt", "w") as f:
            for ylt in self.ylts:
                f.write(ylt + "\n" + "=" * 10 + "\n")

    def _preprocess(self, raws: list[Openable]):
        self.ylts: list[str] = []
        for raw in raws:
            try:
                msg = msg_parser.MsOxMessage(raw)
            except AttributeError:
                continue
            sent = datetime.strptime(msg.sent_date, "%a, %d %b %Y %H:%M:%S %z")

            if self.limit and not (self.limit[0] < sent < self.limit[1]):
                continue

            body = BeautifulSoup(msg.body, "lxml").text
            cleaned = _clean(body)

            if len(cleaned) < 10: continue

            self.ylts.append(cleaned)

    def _tokenize(self):
        self.encodings = []
        for ylt in self.ylts:
            out = self.tokenizer(
                self.SOT + ylt + self.EOT,
                truncation=True,
                max_length=768,
                padding="max_length"
            )
            self.encodings.append((
                torch.unsqueeze(torch.tensor(out["input_ids"]), 0),
                torch.unsqueeze(torch.tensor(out["attention_mask"]), 0)
            ))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

    @property
    def train_len(self) -> int:
        return int(len(self) * args.TRAIN_PORTION)

    @property
    def eval_len(self):
        return len(self) - self.train_len
