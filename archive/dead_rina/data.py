import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset


def load_wikitext103(segments=200000, min_len=100):
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [t for t in ds["text"] if len(t) > min_len][:segments]


def train_bpe_tokenizer(texts, vocab_size=4096, ckpt_name="rina"):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trn = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>"])
    tok.train_from_iterator(texts[:30000], trn)
    tok.add_special_tokens(["<pad>"])
    tok.save(ckpt_name)
    return tok


def tokenize_and_cat(tok, texts, seq_len=64, min_seq=64):
    ids_list = []
    for t in texts:
        enc = tok.encode(t).ids
        if len(enc) > min_seq:
            ids_list.append(torch.tensor(enc, dtype=torch.long))
    if ids_list:
        return torch.cat(ids_list)
    return torch.zeros(0, dtype=torch.long)
