import torch
from transformers import AutoModelForSequenceClassification

from bert_emotion_types import OutputLogits, Tokenized


class Predictor:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, x: Tokenized) -> OutputLogits:
        with torch.no_grad():
            outputs = self.model(
                input_ids=torch.tensor(x.input_ids, device=self.device),
                attention_mask=torch.tensor(x.attention_mask, device=self.device),
                token_type_ids=torch.tensor(x.token_type_ids, device=self.device),
            )
        return OutputLogits.from_model_output(outputs)
