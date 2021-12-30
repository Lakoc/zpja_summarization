from abc import ABC
import torch


class Abstractive(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_max_encoder_size = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def summarize(self, document, length_penalty=2.0, summary_min_len=50, summary_max_len=70, n_beams=4,
                  no_repeat_ngram_size=3):
        input_ids = self.tokenizer.encode_plus(document, truncation=True, padding=True, return_tensors='pt',
                                               max_length=self.model_max_encoder_size)['input_ids']
        summary_ids = self.model.generate(input_ids, min_length=summary_min_len, max_length=summary_max_len,
                                          no_repeat_ngram_size=no_repeat_ngram_size, num_beams=n_beams,
                                          length_penalty=length_penalty,
                                          early_stopping=True).squeeze()
        summary = self.tokenizer.decode(summary_ids, skip_special_tokens=True)
        return summary
