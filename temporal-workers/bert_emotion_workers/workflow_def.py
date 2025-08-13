from datetime import timedelta

from bert_emotion_types import InferenceResults
from temporalio import workflow


@workflow.defn
class BertEmotionWorkflow:
    @workflow.run
    async def run(self, texts: list[str]) -> InferenceResults:
        tokenized = await workflow.execute_activity(
            "preprocess",
            texts,
            start_to_close_timeout=timedelta(seconds=30),
            task_queue="preprocess-q",
        )
        logits = await workflow.execute_activity(
            "predict",
            tokenized,
            start_to_close_timeout=timedelta(seconds=60),
            task_queue="predict-q",
        )
        results = await workflow.execute_activity(
            "postprocess",
            logits,
            start_to_close_timeout=timedelta(seconds=10),
            task_queue="postprocess-q",
        )
        return results
