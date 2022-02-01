import os.path
import pickle
import random

from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from dataset import YLTDataset
import torch
from torch.utils.data import random_split
import sys
import args
import numpy as np
import gc

# random seeds
s = args.SEED
if s:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)

dataset = None
if not os.path.exists("dataset.pickle"):
    dataset = YLTDataset.from_dir(sys.argv[1])
    if args.DATASET_CACHE:
        with open("dataset.pickle", "wb") as f:
            pickle.dump(dataset, f)
else:
    with open("dataset.pickle", "rb") as f:
        dataset = pickle.load(f)

dataset.tokenizer.save_pretrained(sys.argv[3])
# dataset.to_cache()
config = GPT2Config.from_pretrained(args.BASE_MODEL, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(args.BASE_MODEL, config=config)
model.resize_token_embeddings(len(dataset.tokenizer))
# def create_dataloader(ds):
#     return DataLoader(
#         ds,
#         sampler=RandomSampler(ds),
#         batch_size=args.BATCH_SIZE
#     )

train_set, eval_set = random_split(dataset, [dataset.train_len, dataset.eval_len])


# train_dataloader = create_dataloader(train_set)
# eval_dataloader = create_dataloader(eval_set)
#
# optimizer = AdamW(
#     model.parameters(),
#     lr=args.LRU,  # learning rate
#     eps=args.EPSILON,  # epsilon
# )
# scheduler = get_linear_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=args.WARMUP_STEPS,
#     num_training_steps=len(train_dataloader) * args.WARMUP_STEPS
# )


# for epoch in range(args.EPOCHS):
#     print(f"=== EPOCH {epoch + 1} ===")
#
#     start = time.perf_counter()
#     total_loss = 0
#     model.train()
#
#     for step, batch in enumerate(train_dataloader):
#         ids = labels = batch[0].to(device)
#         masks = batch[1].to(device)
#
#         model.zero_grad()
#
#         outputs = model(
#             ids,
#             labels=labels,
#             attention_mask=masks,
#             token_type_ids=None
#         )
#
#         loss = outputs[0]
#         batch_loss = loss.item()
#         total_loss += batch_loss
#
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

def dummy_data_collector(features):
    stack = torch.stack([f[0] for f in features])

    return {
        "input_ids": stack,
        "labels": stack,
        "attention_mask": torch.stack([f[1] for f in features])
    }


training_args = TrainingArguments(
    output_dir="../data/results",
    num_train_epochs=args.EPOCHS,
    per_device_train_batch_size=args.BATCH_SIZE,
    per_device_eval_batch_size=args.BATCH_SIZE,
    warmup_steps=args.WARMUP_STEPS,
    weight_decay=args.WEIGHT_DECAY,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True

    # fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=dummy_data_collector
)

gc.collect()
torch.cuda.empty_cache()
# torch.cuda.set_per_process_memory_fraction(0.5)
trainer.train()
trainer.save_model(sys.argv[2])