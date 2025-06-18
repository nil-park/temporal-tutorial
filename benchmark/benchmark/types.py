from pydantic import BaseModel


class TestCase(BaseModel):
    input: str
    output: str


class TestCases(BaseModel):
    tests: list[TestCase]


class InferenceResult(BaseModel):
    label: str
    score: float
