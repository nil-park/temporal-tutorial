import asyncio
import os

import torch
from bert_emotion_stepwise import Predictor
from bert_emotion_types import OutputLogits, Tokenized
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

predictor = Predictor(device="cuda" if torch.cuda.is_available() else "cpu")


@activity.defn(name="predict")
def predict(tokenized: str) -> str:
    decoded = Tokenized.decode(tokenized, format="json")
    output_logits = predictor(decoded)
    encoded = OutputLogits.encode(output_logits, format="json")
    return encoded


async def main():
    client = await Client.connect(os.environ["TEMPORAL_TARGET"])
    worker = Worker(
        client,
        task_queue="predict-q",
        activities=[predict],
    )
    await worker.run()


def async_run():
    asyncio.run(main())
