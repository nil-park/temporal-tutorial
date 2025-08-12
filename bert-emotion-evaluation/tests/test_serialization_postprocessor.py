from pathlib import Path

import yaml

from bert_emotion_stepwise import Postprocessor
from bert_emotion_types import InferenceResults, OutputLogits
from bert_emotion_utils import print_cyan, print_yellow

TEST_CASES_FILE = Path(__file__).parent.parent / "artifacts" / "test_pipeline.yaml"


def test_serialization_postprocessor():
    print()
    text = TEST_CASES_FILE.read_text()
    test_cases = yaml.safe_load(text)
    postprocessor = Postprocessor()

    print_cyan("Testing serialization of Postprocessor...")
    output_logits = OutputLogits.decode(text, format="yaml")
    inference_results = postprocessor(output_logits)
    encoded_output = inference_results.encode(format="yaml")
    print_cyan("Postprocessed output (encoded):")
    print_yellow(encoded_output)

    assert yaml.safe_load(encoded_output) == {
        "inference_results": test_cases["inference_results"]
    }, "Serialized output does not match expected output."


def test_serialization_postprocessor_round_trip():
    print()
    text = TEST_CASES_FILE.read_text()
    test_cases = yaml.safe_load(text)
    print_cyan("Testing round-trip serialization of Postprocessor...")
    decoded = InferenceResults.decode(text, format="yaml")
    print_cyan("Deserialized InferenceResults object:")
    print_yellow(str(decoded))

    encoded = decoded.encode(format="yaml")
    assert yaml.safe_load(encoded) == {
        "inference_results": test_cases["inference_results"]
    }, "Serialized output does not match expected output."
