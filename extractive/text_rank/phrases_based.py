from extractive.base import Extractive
import pandas as pd
# noinspection PyUnresolvedReferences
import pytextrank


class PhrasesBasedTextRank(Extractive):
    def __init__(self, lang):
        super().__init__(lang)
        self.en_nlp.add_pipe("textrank", config={"stopwords": {"word": ["NOUN"]}})

    def summarize(self, document, num_sentences):
        doc = self.en_nlp(document)
        tr = doc._.textrank
        return tr.summary(limit_sentences=num_sentences)


if __name__ == '__main__':
    summarizer = PhrasesBasedTextRank('en_core_web_sm')
    df = pd.read_csv("tennis_articles.csv")
    text = df['article_text'][0]

    summarization = summarizer.summarize(text, 2)
    for out_sentence in summarization:
        print(out_sentence)
