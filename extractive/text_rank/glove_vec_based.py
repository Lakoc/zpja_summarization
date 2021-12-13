import numpy as np
import pandas as pd
import csv
from extractive.text_rank.text_rank_base import TextRank
from utils.word_vectors import cos_similarity


class GloveVecBasedTextRank(TextRank):
    def __init__(self, lang, vec_file):
        super().__init__(lang)
        self.embeddings = {}
        self.embeddings = pd.read_csv(vec_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE,
                                      na_values=None, keep_default_na=False)
        self.default_embedding = np.zeros((100, 1))

    @staticmethod
    def create_cos_sim_matrix(sentence_vectors):
        sim_mat = cos_similarity(sentence_vectors.T)
        np.fill_diagonal(sim_mat, 0)
        return sim_mat

    def get_vec(self, w):
        if w in self.embeddings.index:
            return self.embeddings.loc[w].to_numpy().reshape((100, 1))
        return self.default_embedding

    def summarize(self, document, num_sentences):
        document = self.en_nlp(document)
        sentences = list(document.sents)
        sentence_vectors = np.concatenate([np.mean([self.get_vec(token.lemma_.lower()) for token in sentence if
                                                    not token.is_stop and not token.is_punct] or [
                                                       self.default_embedding],
                                                   axis=0) for sentence in sentences], axis=1)
        sim_mat = self.create_cos_sim_matrix(sentence_vectors)
        ranked_sentences = self.process_text_ranking(sim_mat, sentences, num_sentences)
        return ranked_sentences


if __name__ == '__main__':
    summarizer = GloveVecBasedTextRank('en_core_web_sm', 'data/glove.6B.100d.txt')
    df = pd.read_csv("tennis_articles.csv")
    text = df['article_text'][0]

    summarization = summarizer.summarize(text, 2)
    for out_sentence in summarization:
        print(out_sentence)
