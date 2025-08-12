from transformers import AutoTokenizer

from bert_emotion_types import Tokenized


class Preprocessor:
    def __init__(self):
        model_name = "boltuix/bert-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, x: str) -> Tokenized:
        y1 = self.tokenizer(x, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return Tokenized.from_tokenizer_output(y1)
