from datasets import load_dataset
from metrics.rouge import Rouge
import time


class WikiHow:
    def __init__(self):
        self.dataset = load_dataset("wikihow", "all", data_dir="data", split='test[:10]')
        self.curr_ind = 0
        self.rouge = Rouge()

    @staticmethod
    def process_single_summary(element, **kwargs):
        start = time.time()
        element[f'summary_{kwargs["name"]}'] = kwargs['function'](element['text'])
        end = time.time()
        element[f'time_{kwargs["name"]}'] = end - start
        return element

    def process_summarization(self, function, name):
        self.dataset = self.dataset.map(self.process_single_summary, fn_kwargs={'function': function, 'name': name})


if __name__ == '__main__':
    data = WikiHow()

    data.process_summarization(lambda x: x, 'rand')
