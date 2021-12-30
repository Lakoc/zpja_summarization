from datasets import load_dataset, load_metric
import time


class CnnDm:
    def __init__(self, split):
        self.dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        self.rouge = load_metric("rouge")

    @staticmethod
    def process_single_summary(element, **kwargs):
        start = time.time()
        element['summary'] = kwargs['function'](element['article'], **kwargs['f_args'])
        end = time.time()
        element['time'] = end - start
        return element

    def process_summarization(self, function, **kwargs):
        self.dataset = self.dataset.map(self.process_single_summary, fn_kwargs={'function': function, **kwargs})

    def calculate_score(self):
        rouge = self.rouge.compute(predictions=self.dataset.data['summary'], references=self.dataset.data['highlights'])
        return rouge
