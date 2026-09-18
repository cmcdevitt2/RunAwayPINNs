"""Training and validation sampling utilities."""

from __future__ import annotations

import numpy as np


class GroupedDeepONetBatchSampler:
    """Shuffle pre-grouped case tensors without scanning pointwise inputs."""

    def __init__(self, data, case_batch_size, seed):
        self.data = data
        self.size = int(data["branch"].shape[0])
        self.case_batch_size = min(int(case_batch_size), self.size)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.size)
        self.cursor = 0

    def next(self):
        if self.cursor + self.case_batch_size <= self.size:
            ids = self.order[self.cursor:self.cursor + self.case_batch_size]
            self.cursor += self.case_batch_size
        else:
            first = self.order[self.cursor:]
            self.order = self.rng.permutation(self.size)
            needed = self.case_batch_size - len(first)
            ids = np.concatenate((first, self.order[:needed]))
            self.cursor = needed
        return {name: values[ids] for name, values in self.data.items()}


class GroupedPointSampler:
    """Sample pointwise MLP batches from grouped FV tensors."""

    def __init__(self, data, seed):
        self.data = data
        self.n_cases = int(data["branch"].shape[0])
        self.counts = np.asarray(np.sum(data["mask"], axis=1), dtype=np.int64)
        if self.n_cases == 0 or np.any(self.counts <= 0):
            raise ValueError("grouped point sampler requires nonempty cases")
        self.rng = np.random.default_rng(seed)

    def next(self, size):
        """Sample cases uniformly, then sample only valid cells in each case."""
        case_ids = self.rng.integers(0, self.n_cases, size=size)
        point_ids = (self.rng.random(size) * self.counts[case_ids]).astype(np.int64)
        trunk = self.data["trunk"][case_ids, point_ids]
        branch = self.data["branch"][case_ids]
        z = np.concatenate((trunk, branch), axis=1)
        y = self.data["target"][case_ids, point_ids]
        return z, np.asarray(y, dtype=np.float64)


def split_cases(cases, results, train_fraction, seed):
    """Split complete cases with a seeded permutation, never individual cells."""
    permutation = np.random.default_rng(seed).permutation(len(cases))
    n_train = min(max(1, int(round(train_fraction * len(cases)))), len(cases) - 1)
    train_ids = np.sort(permutation[:n_train])
    test_ids = np.sort(permutation[n_train:])
    return (cases[train_ids], [results[i] for i in train_ids],
            cases[test_ids], [results[i] for i in test_ids])
