from extractive.base import Extractive


class LeadN(Extractive):
    def __init__(self, lang='en_core_web_sm'):
        super().__init__(lang)

    def summarize(self, document, max_length=4):
        document = self.en_nlp(document)
        sentences = [sent.text for sent in document.sents][: max_length]
        return ' '.join(sentences)
