from pathlib import Path

import yaml

from bert_emotion_types import TestCases
from bert_emotion_utils import print_cyan, print_green, print_magenta, print_red, print_yellow

from . import Postprocessor, Predictor, Preprocessor

TEST_CASES_FILE = Path(__file__).parent.parent / "artifacts" / "test_cases.yaml"


def main():
    print_cyan("Loading bert-emotion model for sentiment analysis...")
    preprocessor = Preprocessor()
    predictor = Predictor()
    postprocessor = Postprocessor()
    print_magenta("Model loaded successfully!")

    print_cyan("Preparing test cases...")
    with open(TEST_CASES_FILE, "r") as file:
        test_cases = TestCases(**yaml.safe_load(file))
    print_magenta(f"Loaded {len(test_cases.tests)} test cases.")

    print_cyan("Running text-classification pipeline on test cases...")
    inputs = [case.input for case in test_cases.tests]
    expected_labels = [case.output for case in test_cases.tests]
    p1 = preprocessor(inputs)
    p2 = predictor(p1)
    got_outputs = postprocessor(p2).inference_results
    for x, y, expected in zip(inputs, got_outputs, expected_labels):
        if y.label == expected:
            print_green(f"Test passed for input: {x}.")
            print_yellow(f"  - {y}")
        else:
            print_red(f"Test failed for input: {x}. Expected: {expected}.")
            print_yellow(f"  - {y}")
