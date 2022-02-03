"""
The trainer script for Yo La Tengo Generation. Should be run on beefy machines
"""

__author__ = "Ben Browner"
__license__ = "MIT"

import argparse
import logging
import os.path
import sys
from pathlib import Path
import warnings

# this only needs to work one time on 2/3/2022
warnings.simplefilter("ignore")

# todo finish cleanup
# todo eliminate pipeline

# CONSTANTS
EPOCHS = 6
BASE = "gpt2"
WARMUP_STEPS = 1e2
BATCH_SIZE = 1
WEIGHT_DECAY = 0.01
TRAIN_PORTION = 0.9


class Args:
    seed: int
    output: Path
    dataset: Path


parser = argparse.ArgumentParser(
    description="Fine-tunes a GPT2 model for Yo La Tengo generation"
)
parser.add_argument(
    "dataset",
    type=Path,
    help="A directory containing Outlook Messages (.msg) "
         "to be used for training the model",
)
parser.add_argument("-s", "--seed", type=int, default=42, help="The seed")
parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=Path("../output"),
    help="The directory that the model " "and the tokenizer will save to",
)

args = parser.parse_args(namespace=Args)

print("(1/4) importing packages...")
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import random_split, Dataset
from bs4 import BeautifulSoup
import msg_parser
import gc


def _cutout(raw: str) -> str:
    start = raw.find(",") + 1
    end = raw.find("be kind!") - len("Laugh hardrun fast")
    return raw[start:end]


class YLTDataset(Dataset):
    """
    I want to throw myself into a fire
    """

    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"
    PAD = "<|pad|>"

    _clean_pipeline = [
        lambda r: r.strip(),  # remove weirdness
        lambda r: "".join([i if ord(i) < 128 else "" for i in r]),  # remove non unicode characters
        _cutout,
        lambda r: r + " Laugh Hard... Run Fast... Be Kind!",
    ]

    def __init__(self, path: Path):
        super(YLTDataset, self).__init__()

        assert path.is_dir()

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            BASE,
            bos_token=self.SOT,
            eos_token=self.EOT,
            pad_token=self.PAD,
        )

        self.ylts = []
        self._process(path)

    def _clean(self, raw: str) -> str:
        current = raw
        for step in self._clean_pipeline:
            current = step(current)
        return current

    def _process(self, path: Path):
        for file in path.iterdir():
            try:
                msg = msg_parser.MsOxMessage(file)
            except AttributeError:
                continue

            body = BeautifulSoup(msg.body, "lxml").text
            cleaned = self._clean(body)

            if len(cleaned) < 10:
                continue

            out = self.tokenizer(
                self.SOT + cleaned + self.EOT,
                truncation=True,
                padding="max_length",
            )

            self.ylts.append(
                (
                    torch.unsqueeze(torch.tensor(out["input_ids"]), 0),
                    torch.unsqueeze(torch.tensor(out["attention_mask"]), 0),
                )
            )

    def __len__(self):
        return len(self.ylts)

    def __getitem__(self, idx):
        return self.ylts[idx]

    @property
    def train_len(self) -> int:
        return int(len(self) * TRAIN_PORTION)

    @property
    def eval_len(self):
        return len(self) - self.train_len


print("(2/4) loading dataset...")
dataset = YLTDataset(args.dataset)
dataset.tokenizer.save_pretrained(args.output / "ylt_tokenizer")

config = GPT2Config.from_pretrained(BASE, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(BASE, config=config)
model.resize_token_embeddings(len(dataset.tokenizer))

train_set, eval_set = random_split(dataset, [dataset.train_len, dataset.eval_len])


def data_collector(features):
    stack = torch.stack([f[0] for f in features])

    return {
        "input_ids": stack,
        "labels": stack,
        "attention_mask": torch.stack([f[1] for f in features]),
    }


print("(3/4) preparing trainer...")
training_args = TrainingArguments(
    output_dir=str(args.output / "ylt_tokenizer"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=int(WARMUP_STEPS),  # ??
    weight_decay=WEIGHT_DECAY,
    overwrite_output_dir=True,
    seed=args.seed,
    do_train=True,
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=data_collector,
)

gc.collect()
torch.cuda.empty_cache()

print("(4/4) training...")
trainer.train()
trainer.save_model(str(args.output / "ylt_model"))

print("Done!")
