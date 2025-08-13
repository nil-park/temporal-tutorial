import asyncio
import os

from bert_emotion_stepwise import Preprocessor
from bert_emotion_types import Tokenized
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

preprocessor = Preprocessor()


@activity.defn(name="preprocess")
async def preprocess(texts: list[str]) -> Tokenized:
    delay = float(os.environ.get("SIMULATE_PREPROCESS_DELAY", "0"))
    if delay:
        await asyncio.sleep(delay)
    return preprocessor(texts)


async def main():
    client = await Client.connect(
        os.environ["TEMPORAL_TARGET"],
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue="preprocess-q",
        activities=[preprocess],
        max_concurrent_activities=1,
    )
    await worker.run()


def async_run():
    asyncio.run(main())
