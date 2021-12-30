from extractive.base import Extractive
# noinspection PyUnresolvedReferences
import pytextrank


class PhrasesBasedTextRank(Extractive):
    def __init__(self, lang='en_core_web_sm'):
        super().__init__(lang)
        self.en_nlp.add_pipe("textrank", config={"stopwords": {"word": ["NOUN"]}})

    def summarize(self, document, num_sentences=4):
        doc = self.en_nlp(document)
        tr = doc._.textrank
        summary = [sent.text for sent in tr.summary(limit_sentences=num_sentences)]
        return ' '.join(summary)

