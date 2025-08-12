from pathlib import Path

import torch
import yaml

from bert_emotion_types import InferenceResult, InferenceResults, OutputLogits


class Postprocessor:
    def __init__(self):
        id2label_yaml = (Path(__file__).parent / "id2label.yaml").read_text()
        self.id2label = yaml.safe_load(id2label_yaml)

    def __call__(self, x: OutputLogits) -> InferenceResults:
        logits = torch.tensor(x.logits, dtype=torch.float32, device="cpu")
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        return InferenceResults(
            inference_results=[
                InferenceResult(
                    label=self.id2label[pred_id],
                    score=probs[i][pred_id].item(),
                )
                for i, pred_id in enumerate(pred_ids.tolist())
            ]
        )
