import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from .workflow_def import BertEmotionWorkflow


async def main():
    client = await Client.connect(os.environ["TEMPORAL_TARGET"])
    worker = Worker(
        client,
        task_queue="workflow-q",
        workflows=[BertEmotionWorkflow],
    )
    await worker.run()


def async_run():
    asyncio.run(main())
