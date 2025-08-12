import base64
import json
from typing import Literal

import blosc
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict


class TestCase(BaseModel):
    input: str
    output: str


class TestCases(BaseModel):
    tests: list[TestCase]


class InferenceResult(BaseModel):
    label: str
    score: float


class InferenceResults(BaseModel):
    model_config = ConfigDict(extra="ignore")
    inference_results: list[InferenceResult]

    def encode(self, format: Literal["json", "yaml"]) -> str:
        data = {
            "inference_results": [
                {
                    "label": result.label,
                    "score": result.score,
                }
                for result in self.inference_results
            ]
        }
        if format == "json":
            return json.dumps(data, sort_keys=True)
        else:
            return yaml.safe_dump(data, sort_keys=True, indent=2)

    @staticmethod
    def decode(text: str, format: Literal["json", "yaml"]) -> "InferenceResults":
        if format == "json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)

        return InferenceResults(**data)


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
        data = {
            "tokenized": {
                "input_ids": _encode_int32(self.input_ids),
                "attention_mask": _encode_int32(self.attention_mask),
                "token_type_ids": _encode_int32(self.token_type_ids),
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

        return Tokenized(
            input_ids=_decode_int32(data["tokenized"]["input_ids"], shape),
            attention_mask=_decode_int32(data["tokenized"]["attention_mask"], shape),
            token_type_ids=_decode_int32(data["tokenized"]["token_type_ids"], shape),
            shape=shape,
        )


class OutputLogits(BaseModel):
    logits: list[list[float]]
    shape: list[int]

    @staticmethod
    def from_model_output(model_output) -> "OutputLogits":
        return OutputLogits(
            logits=model_output.logits.tolist(),
            shape=list(model_output.logits.shape),
        )

    def encode(self, format: Literal["json", "yaml"]) -> str:
        data = {
            "output_logits": {
                "logits": _encode_float32(self.logits),
                "shape": self.shape,
            }
        }
        if format == "json":
            return json.dumps(data, sort_keys=True)
        else:
            return yaml.safe_dump(data, sort_keys=True, indent=2)

    @staticmethod
    def decode(text: str, format: Literal["json", "yaml"]) -> "OutputLogits":
        if format == "json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
        shape = data["output_logits"]["shape"]

        return OutputLogits(
            logits=_decode_float32(data["output_logits"]["logits"], shape),
            shape=shape,
        )


def _encode_int32(arr):
    arr_np = np.array(arr, dtype=np.int32)
    compressed = blosc.compress(arr_np.tobytes(), typesize=4)

    return base64.b64encode(compressed).decode("utf-8")


def _decode_int32(encoded, shape):
    arr_bytes = blosc.decompress(base64.b64decode(encoded))
    return np.frombuffer(arr_bytes, dtype=np.int32).reshape(shape).tolist()


def _encode_float32(arr):
    arr_np = np.array(arr, dtype=np.float32)
    compressed = blosc.compress(arr_np.tobytes(), typesize=4)

    return base64.b64encode(compressed).decode("utf-8")


def _decode_float32(encoded, shape):
    arr_bytes = blosc.decompress(base64.b64decode(encoded))
    return np.frombuffer(arr_bytes, dtype=np.float32).reshape(shape).tolist()
