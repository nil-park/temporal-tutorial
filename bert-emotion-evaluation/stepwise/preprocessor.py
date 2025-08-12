from transformers import AutoTokenizer


class Preprocessor:
    def __init__(self):
        model_name = "boltuix/bert-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, input_text: str):  # TODO: type hint + serialization
        return self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
