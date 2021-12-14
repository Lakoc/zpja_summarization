from datasets import load_metric


class Rouge:
    def __init__(self):
        self.metric = load_metric('rouge')

    def score(self, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)


if __name__ == '__main__':
    pred = ["a b c d"]
    ref = ["a 1 b 2 c"]
    metric = Rouge()
    results = metric.score(pred, ref)
    print(results["rouge1"])
    print(results["rouge1"])
