from abc import ABC
from extractive.base import Extractive
import networkx as nx


class TextRank(Extractive, ABC):
    def __init__(self, lang):
        super().__init__(lang)

    @staticmethod
    def process_text_ranking(sim_mat, sentences, num_sentences):
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        inverse_scores = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:num_sentences]
        ranked_sentences = [sentences[i] for i in inverse_scores]
        return ranked_sentences
