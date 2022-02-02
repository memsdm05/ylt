import argparse
import os.path
import pickle
import random
from datetime import datetime
from typing import Callable, IO, Optional, Union, Any
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import random_split, Dataset
from pathlib import Path
from bs4 import BeautifulSoup
import sys
import msg_parser
import numpy as np
import gc

# todo finish cleanup
# todo eliminate pipeline

# CONSTANTS
EPOCHS = 6
BASE = "gpt2"
WARMUP_STEPS = 1e2
BATCH_SIZE = 1
WEIGHT_DECAY = 0.01
TRAIN_PORTION = 0.1

parser = argparse.ArgumentParser(description="Fine-tunes GPT2 to generate Yo La Tengo's")
parser.add_argument("-s", "--seed")
parser.add_argument("--save-to")
args = parser.parse_args()


def _cutout(raw: str) -> str:
    start = raw.find(",") + 1
    end = raw.find("be kind!") - len("Laugh hardrun fast")
    return raw[start:end]


Openable = Union[IO, os.PathLike]


class YLTDataset(Dataset):
    """
    I want to throw myself into a fire
    """
    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"
    PAD = "<|pad|>"

    _clean_pipline = [
        lambda r: r.strip(),  # remove weirdness
        lambda r: ''.join([i if ord(i) < 128 else '' for i in r]),  # remove non unicode characters
        _cutout,
        lambda r: r + " Laugh Hard... Run Fast... Be Kind!"
    ]

    def __init__(self, path: os.PathLike):
        super(YLTDataset, self).__init__()

        path = Path(path)
        assert path.exists() and path.is_dir()

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            BASE,
            bos_token=self.SOT,
            eos_token=self.EOT,
            pad_token=self.PAD,
            return_token_type_ids=False,
        )

        self.ylts = []
        self._process(path)

    def clean(self, raw: str) -> str:
        current = raw
        for step in self._clean_pipeline:
            current = step(current)

    def _process(self, path: Path):
        for file in path.iterdir():
            try:
                msg = msg_parser.MsOxMessage(file)
            except AttributeError:
                continue

            body = BeautifulSoup(msg.body, "lxml").text
            cleaned = self.clean(body)

            if len(cleaned) < 10:
                continue

            out = self.tokenizer(
                self.SOT + cleaned + self.EOT,
                truncation=True,
                max_length=768,
                padding="max_length"
            )

            self.ylts.append((
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


# random seeds
s = args.SEED
if s:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)

dataset = YLTDataset("TODO")
dataset.tokenizer.save_pretrained(sys.argv[3])

config = GPT2Config.from_pretrained(args.BASE_MODEL, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(args.BASE_MODEL, config=config)
model.resize_token_embeddings(len(dataset.tokenizer))

train_set, eval_set = random_split(dataset, [dataset.train_len, dataset.eval_len])


def data_collector(features):
    stack = torch.stack([f[0] for f in features])

    return {
        "input_ids": stack,
        "labels": stack,
        "attention_mask": torch.stack([f[1] for f in features])
    }


training_args = TrainingArguments(
    output_dir="../data/results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=int(WARMUP_STEPS), # ??
    weight_decay=WEIGHT_DECAY,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=data_collector
)

gc.collect()
torch.cuda.empty_cache()
# torch.cuda.set_per_process_memory_fraction(0.5)
trainer.train()
trainer.save_model(sys.argv[2])
