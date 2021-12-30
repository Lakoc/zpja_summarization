from scripts.methods import methods
from exp_datasets.cnn_daily_mail import CnnDm
import pandas as pd

if __name__ == '__main__':
    n_articles = 10
    cnn = CnnDm(f'test[:{n_articles}]')
    articles = {'article': cnn.dataset['article'], 'highlight': cnn.dataset['highlights']}

    for method in methods:
        m = method()
        cnn.process_summarization(m.summarize, f_args={})
        articles[type(m).__name__] = cnn.dataset['summary']

    df = pd.DataFrame(articles)
    df.to_csv('outputs/qualitative_analysis_data.csv', sep=';')
