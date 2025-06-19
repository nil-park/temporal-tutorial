# from transformers import AutoTokenizer

# MODEL_NAME: str = "boltuix/bert-emotion"


# class BertEmotionTokenizer:
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     def tokenize(self, text):
#         return self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

#     def decode(self, token_ids):
#         return self.tokenizer.decode(token_ids, skip_special_tokens=True)
