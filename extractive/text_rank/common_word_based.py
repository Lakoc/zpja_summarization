import pandas as pd
import numpy as np

from extractive.text_rank.text_rank_base import TextRank


class CommonBasedTextRank(TextRank):
    def __init__(self, lang):
        super().__init__(lang)

    @staticmethod
    def calculate_sentence_sim(s1, s2):
        if len(s1) > 1 and len(s2) > 1:
            common_words = [w for w in s1 if w in s2]
            return len(common_words) / (np.log(len(s1)) + np.log(len(s2)))
        else:
            return 0

    @staticmethod
    def create_sim_matrix(sentences):
        n_sentences = len(sentences)
        sim_mat = np.array(
            [[CommonBasedTextRank.calculate_sentence_sim(sentences[i], sentences[j]) if i != j else 0 for j in
              range(n_sentences)]
             for i in range(n_sentences)])
        return sim_mat

    def summarize(self, document, num_sentences):
        document = self.en_nlp(document)
        sentences = list(document.sents)
        cleaned_sentences = [[token.lemma_.lower() for token in sentence if
                              not token.is_stop and not token.is_punct] for sentence in sentences]
        sim_mat = self.create_sim_matrix(cleaned_sentences)
        ranked_sentences = self.process_text_ranking(sim_mat, sentences, num_sentences)
        return ranked_sentences


if __name__ == '__main__':
    summarizer = CommonBasedTextRank('en_core_web_sm')
    df = pd.read_csv("tennis_articles.csv")
    text = df['article_text'][0]

    summarization = summarizer.summarize(text, 2)
    for out_sentence in summarization:
        print(out_sentence)
