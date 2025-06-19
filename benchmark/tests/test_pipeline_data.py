from pathlib import Path

import yaml

from benchmark.types import Tokenized
from benchmark.utils import print_cyan, print_yellow

FILE_DIR = Path(__file__).parent
REPOSITORY_ROOT = FILE_DIR.parent.parent
TEST_CASES_FILE = REPOSITORY_ROOT / "artifacts" / "test_pipeline.yaml"
MODEL_NAME = "boltuix/bert-emotion"


def test_tokenizer_output():
    print()
    from transformers import AutoTokenizer

    test_cases = yaml.safe_load(TEST_CASES_FILE.read_text())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_text = test_cases["input"]
    output = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    typed_output = Tokenized.from_tokenizer_output(output)
    print_cyan("Tokenized output (encoded):")
    encoded_output = typed_output.to_base64_yaml()
    print_yellow(encoded_output)
    decoded = Tokenized.from_base64_yaml(encoded_output)
    assert decoded == typed_output, "Decoded output does not match the original tokenized output."
    assert encoded_output == yaml.safe_dump(
        {"tokenized": test_cases["tokenized"]},
        sort_keys=True,
        indent=2,
    )
