from pathlib import Path

import yaml

from bert_emotion_stepwise import Preprocessor
from bert_emotion_types import Tokenized
from bert_emotion_utils import print_cyan, print_yellow

TEST_CASES_FILE = Path(__file__).parent.parent / "artifacts" / "test_pipeline.yaml"


def test_serialization_preprocessor():
    print()
    test_cases = yaml.safe_load(TEST_CASES_FILE.read_text())
    preprocessor = Preprocessor()

    print_cyan("Testing serialization of Preprocessor...")
    input_text = test_cases["input"]

    encoded_output = preprocessor(input_text, format="yaml")
    print_cyan("Tokenized output (encoded):")
    print_yellow(encoded_output)

    assert yaml.safe_load(encoded_output) == {
        "tokenized": test_cases["tokenized"]
    }, "Serialized output does not match expected output."


def test_serialization_preprocessor_round_trip():
    print()
    text = TEST_CASES_FILE.read_text()
    test_cases = yaml.safe_load(text)
    print_cyan("Testing round-trip serialization of Preprocessor...")
    decoded = Tokenized.decode(text, format="yaml")
    print_cyan("Deserialized Tokenized object:")
    print_yellow(str(decoded))

    encoded = decoded.encode(format="yaml")
    assert yaml.safe_load(encoded) == {
        "tokenized": test_cases["tokenized"]
    }, "Serialized output does not match expected output."
