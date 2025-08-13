import os
import uuid
from datetime import timedelta

import pytest
from bert_emotion_utils import print_yellow
from temporalio.client import Client

from bert_emotion_workers.workflow_def import BertEmotionWorkflow


@pytest.mark.asyncio
async def test_bert_emotion_workflow():
    texts = [
        "I love programming in Python! It's so much fun.",
        "I hate waiting in long lines. It's so frustrating.",
    ]

    client = await Client.connect(os.environ["TEMPORAL_TARGET"])

    wid = str(uuid.uuid4())
    handle = await client.start_workflow(
        BertEmotionWorkflow.run,
        texts,
        id=wid,
        task_queue="workflow-q",
        execution_timeout=timedelta(seconds=10),
    )

    result = await handle.result()
    print()
    print_yellow(f"Workflow {wid} completed with result: {result}")
