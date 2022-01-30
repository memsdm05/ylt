from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import get_linear_schedule_with_warmup
from dataset import YLTDataset
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import sys
import time
import args

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2',
    bos_token=YLTDataset.SOT,
    eos_token=YLTDataset.EOT
)
dataset = YLTDataset.from_dir(sys.argv[1])
config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)


def create_dataloader(ds):
    return DataLoader(
        ds,
        sampler=RandomSampler(ds),
        batch_size=args.BATCH_SIZE
    )


train_set, eval_set = random_split(dataset, [dataset.train_len, dataset.eval_len])
train_dataloader = create_dataloader(train_set)
eval_dataloader = create_dataloader(eval_set)

optimizer = AdamW(
    model.parameters(),
    lr=args.LRU,  # learning rate
    eps=args.EPSILON,  # epsilon
)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.WARMUP_STEPS,
    num_training_steps=len(train_dataloader) * args.WARMUP_STEPS
)

for epoch in range(args.EPOCHS):
    model.train()

    for step, batch in enumerate(train_dataloader):
        tokens = tokenizer(batch, truncation=True, max_length=dataset.average_len + 20)