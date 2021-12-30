from scripts.methods import methods
from exp_datasets.wikihow_extractive import WikiHow
import pandas as pd
import numpy as np

if __name__ == '__main__':
    part_of_dataset = f'test[:50%]'
    wiki_how = WikiHow(part_of_dataset)
    metrics = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'time': [], 'method': []}
    rouge_keys = ['rouge1', 'rouge2', 'rougeL']

    for method in methods:
        m = method()
        wiki_how.process_summarization(m.summarize, f_args={})
        score = wiki_how.calculate_score()
        for metric in rouge_keys:
            metrics[metric].append(f'{score[metric].mid.recall:.2f} {score[metric].mid.precision:.2f}')
        metrics['time'].append(np.mean(wiki_how.dataset.data['time']))
        metrics['method'].append(type(m).__name__)

    df = pd.DataFrame(metrics)
    df.to_csv('outputs/wikihow_ex.csv', sep=';')
