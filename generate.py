from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

model = GPT2LMHeadModel.from_pretrained("./data/model")
tokenizer = GPT2Tokenizer.from_pretrained("./data/tokenizer")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, early_stopping=True)

print(generator("", num_workers=8, max_length=200)[0]["generated_text"])
