from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .types import InferenceResult, TestCases
from .utils import print_cyan, print_green, print_magenta, print_red, print_yellow

FILE_DIR = Path(__file__).parent
REPOSITORY_ROOT = FILE_DIR.parent.parent
TEST_CASES_FILE = REPOSITORY_ROOT / "artifacts" / "test_cases.yaml"
ID2LABEL_FILE = REPOSITORY_ROOT / "artifacts" / "id2label.yaml"


def main():
    print_cyan("Loading bert-emotion model for sentiment analysis...")
    model_name = "boltuix/bert-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = model.config.id2label
    print_magenta("Model loaded successfully!")

    print_magenta("id2label mapping:")
    print_yellow(yaml.dump(id2label, sort_keys=True))
    with open(ID2LABEL_FILE, "r") as file:
        _id2label = yaml.safe_load(file)
        assert id2label == _id2label, "id2label mapping does not match the saved file."

    print_cyan("Preparing test cases...")
    with open(TEST_CASES_FILE, "r") as file:
        test_cases = TestCases(**yaml.safe_load(file))
    print_magenta(f"Loaded {len(test_cases.tests)} test cases.")

    print_cyan("Running inference on test cases...")
    for case in test_cases.tests:
        # preprocess
        p1 = tokenizer(
            case.input,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # predict
        p2 = model(**p1)

        # postprocess
        probs = torch.softmax(p2.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        typed_result = InferenceResult(
            label=id2label[pred_id],
            score=probs[0][pred_id].item(),  # type: ignore
        )

        if typed_result.label == case.output:
            print_green(f"Test passed for input: {case.input}.")
            print_yellow(f"  - {typed_result}")
        else:
            print_red(f"Test failed for input: {case.input}. Expected: {case.output}.")
            print_yellow(f"  - {typed_result}")


if __name__ == "__main__":
    main()
