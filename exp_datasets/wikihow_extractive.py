from datasets import load_dataset, load_metric
import time


class WikiHow:
    """
    Dataset Size                230,843
    Average Article Length      579.8
    Average Summary Length      62.1
    Vocabulary Size             556,461
    """

    def __init__(self, split):
        self.dataset = load_dataset("wikihow", "all", data_dir="data", split=split)
        self.rouge = load_metric("rouge")
        self.create_extractive_text()

    @staticmethod
    def process_single_summary(element, **kwargs):
        start = time.time()
        element['summary'] = kwargs['function'](element['text'], **kwargs['f_args'])
        end = time.time()
        element['time'] = end - start
        return element

    @staticmethod
    def concat_summary(element):
        element['text'] = element['text'] + element['headline']
        return element

    def create_extractive_text(self):
        self.dataset = self.dataset.map(self.concat_summary)

    def process_summarization(self, function, **kwargs):
        self.dataset = self.dataset.map(self.process_single_summary, fn_kwargs={'function': function, **kwargs})

    def calculate_score(self):
        rouge = self.rouge.compute(predictions=self.dataset.data['summary'], references=self.dataset.data['headline'])
        return rouge
