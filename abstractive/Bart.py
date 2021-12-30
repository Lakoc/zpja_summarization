from transformers import BartForConditionalGeneration, BartTokenizer
from abstractive.base import Abstractive


class Bart(Abstractive):
    def __init__(self):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)
        self.model_max_encoder_size = 1024
