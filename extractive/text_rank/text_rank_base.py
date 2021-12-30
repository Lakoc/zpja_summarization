from abc import ABC
from extractive.base import Extractive
import networkx as nx
from networkx.exception import PowerIterationFailedConvergence
import numpy as np


class TextRank(Extractive, ABC):
    def __init__(self, lang):
        super().__init__(lang)

    @staticmethod
    def process_text_ranking(sim_mat, sentences, num_sentences):
        nx_graph = nx.from_numpy_array(sim_mat)
        try:
            scores = nx.pagerank(nx_graph)
            inverse_scores = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)][
                             :num_sentences]
        except PowerIterationFailedConvergence:
            inverse_scores = np.arange(num_sentences)
        ranked_sentences = [sentences[i] for i in inverse_scores]
        return ranked_sentences
