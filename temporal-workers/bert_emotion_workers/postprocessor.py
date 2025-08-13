import asyncio
import os

from bert_emotion_stepwise import Postprocessor
from bert_emotion_types import InferenceResults, OutputLogits
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

postprocessor = Postprocessor()


@activity.defn(name="postprocess")
async def postprocess(logits: OutputLogits) -> InferenceResults:
    return postprocessor(logits)


async def main():
    client = await Client.connect(
        os.environ["TEMPORAL_TARGET"],
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue="postprocess-q",
        activities=[postprocess],
    )
    await worker.run()


def async_run():
    asyncio.run(main())
