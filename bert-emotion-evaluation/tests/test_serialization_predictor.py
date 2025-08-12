from pathlib import Path

import yaml

from bert_emotion_stepwise import Predictor
from bert_emotion_types import OutputLogits, Tokenized
from bert_emotion_utils import print_cyan, print_yellow

TEST_CASES_FILE = Path(__file__).parent.parent / "artifacts" / "test_pipeline.yaml"


def test_serialization_predictor():
    print()
    text = TEST_CASES_FILE.read_text()
    test_cases = yaml.safe_load(text)
    predictor = Predictor() 

    print_cyan("Testing serialization of Predictor...")
    tokenized = Tokenized.decode(text, format="yaml")
    output_logits = predictor(tokenized)
    encoded_output = output_logits.encode(format="yaml")
    print_cyan("Infered output (encoded):")
    print_yellow(encoded_output)

    assert yaml.safe_load(encoded_output) == {
        "output_logits": test_cases["output_logits"]
    }, "Serialized output does not match expected output."


def test_serialization_predictor_round_trip():
    print()
    text = TEST_CASES_FILE.read_text()
    test_cases = yaml.safe_load(text)
    print_cyan("Testing round-trip serialization of Predictor...")
    decoded = OutputLogits.decode(text, format="yaml")
    print_cyan("Deserialized OutputLogits object:")
    print_yellow(str(decoded))

    encoded = decoded.encode(format="yaml")
    assert yaml.safe_load(encoded) == {
        "output_logits": test_cases["output_logits"]
    }, "Serialized output does not match expected output."
