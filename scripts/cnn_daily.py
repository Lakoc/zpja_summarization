from scripts.methods import methods
from exp_datasets.cnn_daily_mail import CnnDm
import pandas as pd
import numpy as np

if __name__ == '__main__':
    part_of_dataset = f'test[:50%]'
    cnn = CnnDm(part_of_dataset)
    metrics = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'time': [], 'method': []}
    rouge_keys = ['rouge1', 'rouge2', 'rougeL']

    for method in methods:
        m = method()
        cnn.process_summarization(m.summarize, f_args={})
        score = cnn.calculate_score()
        for metric in rouge_keys:
            metrics[metric].append(f'{score[metric].mid.recall:.2f} {score[metric].mid.precision:.2f}')
        metrics['time'].append(np.mean(cnn.dataset.data['time']))
        metrics['method'].append(type(m).__name__)

    df = pd.DataFrame(metrics)
    df.to_csv('outputs/cnn_daily_mail.csv', sep=';')
