import spacy
from abc import abstractmethod, ABC


class Extractive(ABC):
    def __init__(self, lang):
        self.en_nlp = spacy.load(lang)

    @abstractmethod
    def summarize(self, document, num_sentences):
        pass
