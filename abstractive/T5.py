from transformers import T5Tokenizer, T5ForConditionalGeneration
from abstractive.base import Abstractive


class T5(Abstractive):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model_max_encoder_size = 512
