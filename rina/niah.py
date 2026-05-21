"""
NIAH data generator — realistic key-value insertion into natural text.
Inspired by RULER (Needle-in-a-Haystack evaluation framework).
"""
import random
import torch
from tokenizers import Tokenizer
from datasets import load_dataset


class NIAHGenerator:
    def __init__(self, vocab_size=4096, rng_seed=42):
        self.vocab_size = vocab_size
        self.rng = random.Random(rng_seed)
        # Reserve 40 tokens for key/value pool (20 keys, 20 values)
        # These are high-index tokens unlikely to collide with common BPE
        self.key_pool = list(range(4056, 4076))
        self.val_pool = list(range(4076, 4096))

    def make_sample(self, text_ids, seq_len=64):
        """
        Insert key→value pair into a real text paragraph.
        Returns: (x, key_id, val_id, kv_pos, q_pos)
          x: [seq_len] sequence with inserted key/value/query
          key_id: the key token
          val_id: the value token
          kv_pos: position where key→value is inserted
          q_pos:  position where query key is (last position)
        """
        key_id = self.rng.choice(self.key_pool)
        val_id = self.rng.choice(self.val_pool)
        while val_id == key_id:
            val_id = self.rng.choice(self.val_pool)

        text = text_ids[:seq_len - 2].tolist() if len(text_ids) > seq_len - 2 else text_ids.tolist()
        if len(text) < seq_len - 2:
            text = text + [0] * (seq_len - 2 - len(text))

        kv_pos = self.rng.randint(1, seq_len - 4)
        text[kv_pos:kv_pos] = [key_id, val_id]
        text = text[:seq_len - 1]
        text.append(key_id)

        x = torch.tensor(text, dtype=torch.long)
        return x, key_id, val_id, kv_pos

    def make_batch(self, texts, batch_size=8, seq_len=64):
        batch = []
        keys = []
        vals = []
        for i in range(batch_size):
            idx = self.rng.randint(0, len(texts) - 1)
            text = texts[idx]
            if isinstance(text, str):
                ids = torch.tensor([], dtype=torch.long)
            else:
                ids = text
            if len(ids) < seq_len:
                ids = torch.cat([ids, torch.zeros(seq_len - len(ids), dtype=torch.long)])
            x, key_id, val_id, _ = self.make_sample(ids, seq_len)
            batch.append(x)
            keys.append(key_id)
            vals.append(val_id)
        return torch.stack(batch), keys, vals

    def load_wikitext_paragraphs(self, num_segments=50000, min_len=128):
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
        texts = []
        for t in ds["text"][:num_segments]:
            if len(t) > min_len:
                ids = tok.encode(t).ids
                if len(ids) > min_len:
                    texts.append(torch.tensor(ids, dtype=torch.long))
        return texts
