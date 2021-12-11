import spacy
from abc import abstractmethod, ABC
import networkx as nx


class TextRank(ABC):
    def __init__(self, lang):
        self.en_nlp = spacy.load(lang)

    @abstractmethod
    def summarize(self, document, num_sentences):
        pass

    @staticmethod
    def process_text_ranking(sim_mat, sentences, num_sentences):
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        inverse_scores = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:num_sentences]
        ranked_sentences = [sentences[i] for i in inverse_scores]
        return ranked_sentences
