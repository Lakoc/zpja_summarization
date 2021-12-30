from extractive.bert.model.model_builder import ExtSummarizer
from extractive.base import Extractive
import numpy as np
import torch
from transformers import BertTokenizer
from extractive.bert.utils import _block_tri


class DistilBert(Extractive):
    """BertSumExt distilbert model based by https://github.com/chriskhanhtran/bert-extractive-summarization/"""

    def __init__(self, checkpoint_path='data/checkpoints/distilbert_ext.pt', lang='en_core_web_sm'):
        super().__init__(lang)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model = ExtSummarizer(checkpoint=checkpoint, bert_type='distilbert', device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, local_files_only=True)
        self.model.eval()
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]

    def encode(self, document, max_pos, device):
        document = self.en_nlp(document)
        sentences = [sent.text for sent in document.sents]
        processed_text = "[CLS] [SEP]".join([sent.strip().lower() for sent in sentences])

        src_subtokens = self.tokenizer.tokenize(processed_text)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = self.sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, clss, mask_cls, [sentences]

    def select_sentences(self, input_data, max_length, block_trigram=True):
        with torch.no_grad():
            src, mask, clss, mask_cls, src_str = input_data
            sent_scores, mask = self.model(src, clss, mask, mask_cls)
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

    def summarize(self, document, max_length=4, max_pos=512):
        input_data = self.encode(document, max_pos, device="cpu")
        doc_summary = self.select_sentences(input_data, max_length, block_trigram=True)
        return doc_summary
