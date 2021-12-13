from extractive.bert.model.model_builder import ExtSummarizer
import pandas as pd
from extractive.base import Extractive
import numpy as np
import torch
from transformers import BertTokenizer

from extractive.bert.utils import _process_src, _block_tri


class DistilBert(Extractive):
    """https://github.com/chriskhanhtran/bert-extractive-summarization/"""

    def __init__(self, checkpoint_path, lang):
        super().__init__(lang)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model = ExtSummarizer(checkpoint=checkpoint, bert_type='distilbert', device='cpu')

    @staticmethod
    def load_text(processed_text, max_pos, device):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        sep_vid = tokenizer.vocab["[SEP]"]
        cls_vid = tokenizer.vocab["[CLS]"]

        src, mask_src, segments_ids, clss, mask_cls = _process_src(tokenizer, max_pos, sep_vid, device, cls_vid,
                                                                   processed_text)
        segs = torch.tensor(segments_ids)[None, :].to(device)
        src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
        return src, mask_src, segs, clss, mask_cls, src_text

    def test(self, input_data, max_length, block_trigram=True):
        with torch.no_grad():
            src, mask, segs, clss, mask_cls, src_str = input_data
            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores, 1)

            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                if len(src_str[i]) == 0:
                    continue
                for j in selected_ids[i][: len(src_str[i])]:
                    if j >= len(src_str[i]):
                        continue
                    candidate = src_str[i][j].strip()
                    if block_trigram:
                        if not _block_tri(candidate, _pred):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if len(_pred) == max_length:
                        break
                pred.extend(_pred)

        return pred

    def preprocess(self, raw_text):
        document = self.en_nlp(raw_text)
        sentences = [sent.text for sent in document.sents]
        processed_text = "[CLS] [SEP]".join(sentences)
        return processed_text

    def summarize(self, document, max_length=3, max_pos=512):
        self.model.eval()
        processed_text = self.preprocess(document)
        input_data = self.load_text(processed_text, max_pos, device="cpu")
        doc_summary = self.test(input_data, max_length, block_trigram=True)
        return doc_summary


if __name__ == '__main__':
    df = pd.read_csv("tennis_articles.csv")
    text = df['article_text'][0]
    bert = DistilBert(f'extractive/bert/checkpoints/distilbert_ext.pt', 'en_core_web_sm')
    summary = bert.summarize(text, max_length=2)
    for sentence in summary:
        print(sentence)
