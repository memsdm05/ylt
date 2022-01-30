import contextlib

import torch
import os
from typing import Callable, IO, Optional, Union
from datetime import datetime
from bs4 import BeautifulSoup
import msg_parser
from torch.utils.data import Dataset
from pathlib import Path
import lxml


class Pipeline:
    def __init__(self, *steps: Callable[[str], str]):
        self.steps = steps

    def __call__(self, start: str) -> str:
        current = start
        for step in self.steps:
            current = step(current)
        return current

Openable = Union[IO, os.PathLike]

class YLTDataset(Dataset):
    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"

    def __init__(self, raw_emails: list[Openable], *,
                 limit: Optional[tuple[datetime, datetime]] = None):
        super(YLTDataset, self).__init__()

        if limit:
            assert limit[0] < limit[1]
        self.limit = limit

        self.clean = Pipeline(
            lambda r: r.strip(),  # remove weirdness
            lambda r: ''.join([i if ord(i) < 128 else '' for i in r]),  # remove non unicode characters
            self._cutout
        )

        self.ylts: list[str] = []
        self._parse(raw_emails)

        self.average_len = self._calc_average_length()

    def _calc_average_length(self) -> int:
        return int(
            sum([len(ylt) for ylt in self.ylts]) / len(self.ylts)
        )

    def _cutout(self, raw: str) -> str:
        start = raw.find(",") + 1
        end = raw.find("be kind!") + len("be kind!")
        return raw[start:end]

    @classmethod
    def from_dir(cls, root: str, *args, **kwargs):
        p = Path(root)
        assert p.is_dir()
        return cls(list(p.iterdir()), *args, **kwargs)

    def _parse(self, raws: list[Openable]):
        for raw in raws:
            try:
                msg = msg_parser.MsOxMessage(raw)
            except AttributeError:
                continue
            sent = datetime.strptime(msg.sent_date, "%a, %d %b %Y %H:%M:%S %z")

            if self.limit and not (self.limit[0] < sent < self.limit[1]):
                continue

            body = BeautifulSoup(msg.body, "lxml").text
            cleaned = self.clean(body)

            if len(cleaned) < 10: continue

            self.ylts.append(self.SOT + cleaned + self.EOT)

    def __len__(self):
        return len(self.ylts)

    def __getitem__(self, idx):
        return self.ylts[idx]

    @property
    def train_len(self) -> int:
        return int(len(self) * 0.9)

    @property
    def eval_len(self):
        return len(self) - self.train_len
