from pathlib import Path

import yaml

from bert_emotion_types import InferenceResult, TestCases
from utils import print_cyan, print_green, print_magenta, print_red, print_yellow

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
    for case in test_cases.tests:
        p1 = preprocessor(case.input)
        p2 = predictor(p1)
        result: InferenceResult = postprocessor(p2)
        if result.label == case.output:
            print_green(f"Test passed for input: {case.input}.")
            print_yellow(f"  - {result}")
        else:
            print_red(f"Test failed for input: {case.input}. Expected: {case.output}.")
            print_yellow(f"  - {result}")
