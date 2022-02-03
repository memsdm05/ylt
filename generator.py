import os.path

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


class YLTGenerator:
    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"
    PAD = "<|pad|>"

    def __init__(self, tokenizer_path: str, model_path: str, min_length=100, max_length=300):
        assert os.path.exists(tokenizer_path) and os.path.exists(model_path)

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer_path,
            bos_token=self.SOT,
            eos_token=self.EOT,
            pad_token=self.PAD,
        )
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

        self.min_length = min_length
        self.max_length = max_length

        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, use_fast=True)

    def __call__(self, start: str) -> str:
        output = self.generator(start, min_length=self.min_length, max_length=self.max_length, num_return_sequences=1)
        return output[0]["generated_text"]


if __name__ == '__main__':
    pass
