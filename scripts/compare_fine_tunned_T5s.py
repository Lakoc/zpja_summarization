from exp_datasets.wikihow import WikiHow
from exp_datasets.cnn_daily_mail import CnnDm
from abstractive.T5 import T5
from abstractive.T5Wiki import T5Wiki
import pandas as pd
import numpy as np

if __name__ == '__main__':
    part_of_dataset = f'test[:50%]'
    datasets = [WikiHow(part_of_dataset), CnnDm(part_of_dataset)]
    methods = [T5, T5Wiki]
    rouge_keys = ['rouge1', 'rouge2', 'rougeL']
    for dataset in datasets:
        metrics = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'time': [], 'method': []}
        for method in methods:
            m = method()
            dataset.process_summarization(m.summarize, f_args={})
            score = dataset.calculate_score()
            for metric in rouge_keys:
                metrics[metric].append(f'{score[metric].mid.recall:.2f} {score[metric].mid.precision:.2f}')
            metrics['time'].append(np.mean(dataset.dataset.data['time']))
            metrics['method'].append(type(m).__name__)

        df = pd.DataFrame(metrics)
        df.to_csv(f'outputs/compare_fine_tunned_{type(dataset).__name__}.csv', sep=';')
