import asyncio
import os

from bert_emotion_stepwise import Postprocessor
from bert_emotion_types import InferenceResults, OutputLogits
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

postprocessor = Postprocessor()


@activity.defn(name="postprocess")
def postprocess(logits: str) -> str:
    decoded = OutputLogits.decode(logits, format="json")
    results = postprocessor(decoded)
    encoded = InferenceResults.encode(results, format="json")
    return encoded


async def main():
    client = await Client.connect(os.environ["TEMPORAL_TARGET"])
    worker = Worker(
        client,
        task_queue="postprocess-q",
        activities=[postprocess],
    )
    await worker.run()


def async_run():
    asyncio.run(main())
