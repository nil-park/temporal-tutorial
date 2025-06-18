from pydantic import BaseModel


class TestCase(BaseModel):
    input: str
    output: str


class TestCases(BaseModel):
    tests: list[TestCase]


class InferenceResult(BaseModel):
    label: str
    score: float


class TokenizedInput(BaseModel):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    token_type_ids: list[list[int]]
    shape: list[int]

    @staticmethod
    def from_tokenizer_output(tokenizer_output: dict) -> "TokenizedInput":
        shape = list(tokenizer_output["input_ids"].shape)
        assert shape == list(
            tokenizer_output["attention_mask"].shape
        ), "Input IDs and attention mask must have the same shape."
        assert shape == list(
            tokenizer_output["token_type_ids"].shape
        ), "Input IDs and token type IDs must have the same shape."
        return TokenizedInput(
            input_ids=tokenizer_output["input_ids"].tolist(),
            attention_mask=tokenizer_output["attention_mask"].tolist(),
            token_type_ids=tokenizer_output["token_type_ids"].tolist(),
            shape=shape,
        )
