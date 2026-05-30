# DEPRECATED ¡ª legacy TemporalSNN architecture, use rina.mohe.MoHE instead
import torch


class SlotMemory:
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.size = 0
        self.keys = [None] * capacity
        self.values = [None] * capacity
        self.ages = [0] * capacity

    def _hash(self, token_id, position):
        return hash((token_id, position))

    def _find_slot(self, h):
        return h % self.capacity

    def lookup(self, token_id, position=None):
        if position is None:
            for i in range(self.capacity):
                if self.keys[i] is not None and self.keys[i][0] == token_id:
                    return self.values[i]
            return None

        h = self._hash(token_id, position)
        start = self._find_slot(h)
        for i in range(self.capacity):
            idx = (start + i) % self.capacity
            if self.keys[idx] is None:
                return None
            if self.keys[idx] == (token_id, position):
                return self.values[idx]
        return None

    def insert(self, token_id, position, value_token_id):
        h = self._hash(token_id, position)
        start = self._find_slot(h)

        empty_slot = None
        oldest_slot = 0
        oldest_age = self.ages[0]

        for i in range(self.capacity):
            idx = (start + i) % self.capacity

            if self.keys[idx] is None and empty_slot is None:
                empty_slot = idx

            if self.keys[idx] == (token_id, position):
                self.values[idx] = value_token_id
                self.ages[idx] = 0
                return

            if self.ages[idx] > oldest_age:
                oldest_age = self.ages[idx]
                oldest_slot = idx

        target = empty_slot if empty_slot is not None else oldest_slot
        self.keys[target] = (token_id, position)
        self.values[target] = value_token_id
        self.ages[target] = 0
        if empty_slot is not None:
            self.size += 1

    def tick(self):
        for i in range(self.size if self.size < self.capacity else self.capacity):
            if self.keys[i] is not None:
                self.ages[i] += 1

    def clear(self):
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.ages = [0] * self.capacity
        self.size = 0

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"SlotMemory(capacity={self.capacity}, size={self.size})"
