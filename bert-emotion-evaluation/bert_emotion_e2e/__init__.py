from transformers import pipeline

from bert_emotion_types import InferenceResult, InferenceResults


class EndToEndModel:
    def __init__(self):
        self.model = pipeline("text-classification", model="boltuix/bert-emotion")

    def __call__(self, x: list[str]) -> InferenceResults:
        result: list[dict] = self.model(x)

        return InferenceResults(
            inference_results=[
                InferenceResult(
                    label=y["label"],
                    score=y["score"],
                )
                for y in result
            ]
        )


__all__ = ["EndToEndModel"]
