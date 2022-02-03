import os.path

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


class YLTGenerator:
    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"
    PAD = "<|pad|>"

    def __init__(self, tokenizer_path: str, model_path: str):
        assert os.path.exists(tokenizer_path) and os.path.exists(model_path)

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer_path,
            bos_token=self.SOT,
            eos_token=self.EOT,
            pad_token=self.PAD,
        )
        self.model = GPT2LMHeadModel.from_pretrained(model_path)


    def __call__(self, start: str, min_length: int = 100, max_length: int = 300) -> str:
        output = self.model.generate(
            self.tokenizer.encode(start, return_tensors="pt"),
            min_length=min_length,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )

        return self.tokenizer.decode(output)


if __name__ == '__main__':
    pass
