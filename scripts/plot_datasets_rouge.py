import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for dataset in ['cnn_daily_mail', 'wikihow']:
        df = pd.read_csv(f'outputs/{dataset}.csv', sep=';')

        new_df = pd.concat([df[column].str.split(expand=True).rename(
            columns={0: f"{column} recall", 1: f"{column} precision"}) for column in ['rouge1', 'rouge2', 'rougeL']],
            axis=1).astype('float')

        for metric in ['recall', 'precision']:
            new_df_split = new_df[[f"{column} {metric}" for column in ['rouge1', 'rouge2', 'rougeL']]]
            new_df_split['method'] = df['method']

            new_df_split = new_df_split.set_index('method').transpose()

            ax = new_df_split.plot.line(figsize=(10, 4))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.tight_layout()
            plt.tick_params(bottom=False)
            plt.savefig(f'outputs/{dataset}_{metric}.pdf')
