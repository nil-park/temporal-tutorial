from pathlib import Path
from random import choices
from time import time

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm, trange
from transformers.pipelines import pipeline

from .utils import print_cyan, print_green, print_magenta, print_red, print_yellow

FILE_DIR = Path(__file__).parent
REPOSITORY_ROOT = FILE_DIR.parent.parent
TEST_CASES_FILE = REPOSITORY_ROOT / "cases" / "test_cases.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
    )
    batch_size: int = Field(
        default=16,
        alias="b",
        description="Batch size for the model inference.",
        ge=1,
        le=64,
    )
    iterations: int = Field(
        default=100,
        alias="i",
        description=(
            "Number of iterations to run the benchmark. "
            "If set to 0, print the results without running the benchmark."
        ),
    )


class TestCase(BaseModel):
    input: str
    output: str


class TestCases(BaseModel):
    tests: list[TestCase]


class InferenceResult(BaseModel):
    label: str
    score: float


def main():
    print_cyan("Loading settings...")
    settings = Settings()
    print_magenta(f"Settings loaded: {settings}")

    print_cyan("Loading bert-emotion model for sentiment analysis...")
    sentiment_analysis = pipeline("text-classification", model="boltuix/bert-emotion")
    print_magenta("Model loaded successfully!")

    print_cyan("Preparing test cases...")
    with open(TEST_CASES_FILE, "r") as file:
        test_cases = TestCases(**yaml.safe_load(file))
    print_magenta(f"Loaded {len(test_cases.tests)} test cases.")

    if settings.iterations:
        inputs = [case.input for case in test_cases.tests]
        print_cyan(f"Running benchmark for {settings.iterations} iterations...")
        start_time = time()
        for i in trange(settings.iterations, desc="Benchmarking iterations"):
            case = choices(inputs, k=settings.batch_size)
            _ = sentiment_analysis(case)
        end_time = time()
        elapsed_time = end_time - start_time
        milliseconds_per_iteration = (elapsed_time * 1000) / settings.iterations
        print_magenta(
            f"Benchmark completed. Average inference time: {milliseconds_per_iteration:.2f} ms per iteration."
        )
    else:
        print_cyan("No iterations specified. Running tests without benchmarking.")
        for case in tqdm(test_cases.tests, desc="Running test cases"):
            result: list[dict] = sentiment_analysis(case.input)  # type: ignore
            typed_result = InferenceResult(**result[0])
            if typed_result.label == case.output:
                print_green(f"Test passed for input: {case.input}.")
                print_yellow(f"  - {typed_result}")
            else:
                print_red(f"Test failed for input: {case.input}. Expected: {case.output}.")
                print_yellow(f"  - {typed_result}")


if __name__ == "__main__":
    main()
