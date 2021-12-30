import spacy
from abc import abstractmethod, ABC
import torch


class Extractive(ABC):
    def __init__(self, lang):
        self.en_nlp = spacy.load(lang)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def summarize(self, document, num_sentences):
        pass
