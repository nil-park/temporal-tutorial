import asyncio
import os

from bert_emotion_stepwise import Preprocessor
from bert_emotion_types import Tokenized
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

preprocessor = Preprocessor()


@activity.defn(name="preprocess")
async def preprocess(texts: list[str]) -> str:
    tokenized: Tokenized = preprocessor(texts)
    encoded = tokenized.encode(format="json")
    return encoded


async def main():
    client = await Client.connect(os.environ["TEMPORAL_TARGET"])
    worker = Worker(
        client,
        task_queue="preprocess-q",
        activities=[preprocess],
    )
    await worker.run()


def async_run():
    asyncio.run(main())
