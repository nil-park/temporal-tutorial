import asyncio
import os

import torch
from bert_emotion_stepwise import Predictor
from bert_emotion_types import OutputLogits, Tokenized
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

predictor = Predictor(device="cuda" if torch.cuda.is_available() else "cpu")


@activity.defn(name="predict")
async def predict(tokenized: Tokenized) -> OutputLogits:
    delay = float(os.environ.get("SIMULATE_PREDICTOR_DELAY", "0"))
    if delay:
        await asyncio.sleep(delay)
    return predictor(tokenized)


async def main():
    client = await Client.connect(
        os.environ["TEMPORAL_TARGET"],
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue="predict-q",
        activities=[predict],
        max_concurrent_activities=1,
    )
    await worker.run()


def async_run():
    asyncio.run(main())
