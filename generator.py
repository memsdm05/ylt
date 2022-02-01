from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed

model = GPT2LMHeadModel.from_pretrained("E:/docs/ylt_model")
tokenizer = GPT2Tokenizer.from_pretrained("./data/tokenizer")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, early_stopping=True)
set_seed(42)
print(generator("", num_workers=8, max_length=300)[0]["generated_text"])
