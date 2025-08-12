import base64
import json
from typing import Literal

import blosc
import numpy as np
import yaml
from pydantic import BaseModel


class TestCase(BaseModel):
    input: str
    output: str


class TestCases(BaseModel):
    tests: list[TestCase]


class InferenceResult(BaseModel):
    label: str
    score: float


class Tokenized(BaseModel):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    token_type_ids: list[list[int]]
    shape: list[int]

    @staticmethod
    def from_tokenizer_output(tokenizer_output: dict) -> "Tokenized":
        shape = list(tokenizer_output["input_ids"].shape)
        assert shape == list(
            tokenizer_output["attention_mask"].shape
        ), "Input IDs and attention mask must have the same shape."
        assert shape == list(
            tokenizer_output["token_type_ids"].shape
        ), "Input IDs and token type IDs must have the same shape."
        return Tokenized(
            input_ids=tokenizer_output["input_ids"].tolist(),
            attention_mask=tokenizer_output["attention_mask"].tolist(),
            token_type_ids=tokenizer_output["token_type_ids"].tolist(),
            shape=shape,
        )

    def encode(self, format: Literal["json", "yaml"]) -> str:
        def _encode(arr):
            arr_np = np.array(arr, dtype=np.int32)
            compressed = blosc.compress(arr_np.tobytes(), typesize=4)

            return base64.b64encode(compressed).decode("utf-8")

        data = {
            "tokenized": {
                "input_ids": _encode(self.input_ids),
                "attention_mask": _encode(self.attention_mask),
                "token_type_ids": _encode(self.token_type_ids),
                "shape": self.shape,
            }
        }
        if format == "json":
            return json.dumps(data, sort_keys=True)
        else:
            return yaml.safe_dump(data, sort_keys=True, indent=2)

    @staticmethod
    def decode(text: str, format: Literal["json", "yaml"]) -> "Tokenized":
        if format == "json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
        shape = data["tokenized"]["shape"]

        def decode(encoded):
            arr_bytes = blosc.decompress(base64.b64decode(encoded))
            return np.frombuffer(arr_bytes, dtype=np.int32).reshape(shape).tolist()

        return Tokenized(
            input_ids=decode(data["tokenized"]["input_ids"]),
            attention_mask=decode(data["tokenized"]["attention_mask"]),
            token_type_ids=decode(data["tokenized"]["token_type_ids"]),
            shape=shape,
        )
