from abstractive.Bart import Bart
from abstractive.Distilbart import DistilBart
from abstractive.T5 import T5
from extractive.lead_n import LeadN
from extractive.bert.BertSumExt import BertSumExt
from extractive.text_rank.glove_vec_based import GloveVecBasedTextRank
from extractive.text_rank.phrases_based import PhrasesBasedTextRank
from extractive.text_rank.common_word_based import CommonBasedTextRank

methods = [Bart, DistilBart, T5, LeadN, BertSumExt, GloveVecBasedTextRank, PhrasesBasedTextRank, CommonBasedTextRank]
