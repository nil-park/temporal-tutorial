from transformers import AutoModelForSequenceClassification


class Predictor:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion")

    def __call__(self, x):  # TODO: type hind + serialization
        return self.model(**x)
