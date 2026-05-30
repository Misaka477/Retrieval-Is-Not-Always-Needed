########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, time, random

########################################################################################################
# Tokenizer #1 (reference, naive, slow)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

########################################################################################################
# Tokenizer #2 (trie, faster) https://github.com/TkskKurumi/ChatRWKV-TRIE-Tokenizer
# UPDATE: now much faster
########################################################################################################

class TRIE:
    __slots__ = ("to", "token")
    to:list
    token:int
    def __init__(self):
        self.to = [None for ch in range(256)]
        self.token = 0

    def __repr__(self):
        return "<TRIE token=%s>" % (self.token - 1)

    def add(self, key:bytes, val:int):
        u = self
        for ch in key:
            v = u.to[ch]
            if v is None:
                v = TRIE()
                u.to[ch] = v
            u = v
        u.token = val + 1

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            idx2token[idx] = x

        self.token2idx = {}
        for k,v in idx2token.items():
            self.token2idx[v] = int(k)
        self.idx2token = [b"" for _ in range(max(idx2token) + 1)]
        for idx, token in idx2token.items():
            self.idx2token[idx] = token

        self.root = TRIE()
        for t, i in self.token2idx.items():
            self.root.add(t, val=i)
        for ch in range(256):
            assert self.root.to[ch] is not None

    def encodeBytes(self, src:bytes):
        tokens = []
        append = tokens.append
        root_to = self.root.to
        idx = 0
        src_len = len(src)
        while idx < src_len:
            u = root_to[src[idx]]
            j = idx + 1
            token = u.token
            end = j
            to = u.to
            while j < src_len:
                u = to[src[j]]
                if u is None:
                    break
                j += 1
                tok = u.token
                if tok:
                    token = tok
                    end = j
                to = u.to
            append(token - 1)
            idx = end
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(self.idx2token.__getitem__, tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

########################################################################################################
# Demo
########################################################################################################

if __name__ == "__main__":
    print("Test mode not available when run as script. Import the classes instead.")
