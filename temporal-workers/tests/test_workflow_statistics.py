import os

import pytest
from bert_emotion_utils import print_cyan, print_yellow
from temporalio.api.enums.v1 import TaskQueueType
from temporalio.api.taskqueue.v1 import TaskQueue
from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest
from temporalio.client import Client


@pytest.mark.asyncio
async def test_bert_emotion_count_workflows():
    client = await Client.connect(os.environ["TEMPORAL_TARGET"])

    # Only count running workflows (not closed)
    y = await client.count_workflows(
        query="WorkflowType='BertEmotionWorkflow' AND CloseTime IS NULL",
    )
    print()
    print_cyan("Running workflow count:")
    print_yellow(str(y))


@pytest.mark.asyncio
async def test_bert_emotion_describe_workflow_task_queue():
    client = await Client.connect(os.environ["TEMPORAL_TARGET"])

    y = await client.workflow_service.describe_task_queue(
        DescribeTaskQueueRequest(
            namespace="default",
            task_queue=TaskQueue(name="preprocess-q"),
            task_queue_type=TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY,
        )
    )
    print()
    print_cyan("Describing task queue for preprocess-q:")
    print_yellow(str(y))
