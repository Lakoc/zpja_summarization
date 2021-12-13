import torch


def _process_src(tokenizer, max_pos, sep_vid, device, cls_vid, raw):
    raw = raw.strip().lower()
    raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
    src_subtokens = tokenizer.tokenize(raw)
    src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
    src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
    src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
    src_subtoken_idxs[-1] = sep_vid
    _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
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
    cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
    clss = torch.tensor(cls_ids).to(device)
    mask_cls = 1 - (clss == -1).float()
    clss[clss == -1] = 0
    return src, mask_src, segments_ids, clss, mask_cls


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i: i + n]))
    return ngram_set


def _block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False
