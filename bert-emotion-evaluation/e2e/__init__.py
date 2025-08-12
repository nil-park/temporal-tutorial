from transformers import pipeline

from bert_emotion_types import InferenceResult


class EndToEndModel:
    def __init__(self):
        self.model = pipeline("text-classification", model="boltuix/bert-emotion")

    def __call__(self, input_text: str) -> InferenceResult:
        result: list[dict] = self.model(input_text)
        return InferenceResult(**result[0])


__all__ = ["EndToEndModel"]
