from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from dataset import YLTDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

dataset = YLTDataset.from_dir("E:/docs/yo_la_tengo")
for idx in range(len(dataset)):
    print(dataset[idx])
config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

# optimizer = Adam(model, lr=0.0005)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

def train(data):
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        for idx, ylt in enumerate(loader):
            ylt = torch.unsqueeze(ylt, 0)
            tensor = torch.tensor(tokenizer.encode(ylt[0]))