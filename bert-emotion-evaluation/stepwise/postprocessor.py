from pathlib import Path

import torch
import yaml

from bert_emotion_types import InferenceResult


class Postprocessor:
    def __init__(self):
        id2label_yaml = (Path(__file__).parent / "id2label.yaml").read_text()
        self.id2label = yaml.safe_load(id2label_yaml)

    def __call__(self, x):  # TODO: type hint + serialization
        probs = torch.softmax(x.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        return InferenceResult(
            label=self.id2label[pred_id],
            score=probs[0][pred_id].item(),  # type: ignore
        )
