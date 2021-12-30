from transformers import BartTokenizer, BartForConditionalGeneration
from abstractive.base import Abstractive


class DistilBart(Abstractive):
    def __init__(self):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        self.model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6').to(self.device)
        self.model_max_encoder_size = 1024
