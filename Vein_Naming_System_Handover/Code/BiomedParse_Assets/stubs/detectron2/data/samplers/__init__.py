import torch
import torch.utils.data as torchdata


class TrainingSampler(torchdata.Sampler):
    def __init__(self, size, shuffle=True):
        self._size = size
        self._shuffle = shuffle

    def __iter__(self):
        indices = torch.randperm(self._size).tolist() if self._shuffle else list(range(self._size))
        while True:
            yield from indices
            indices = torch.randperm(self._size).tolist() if self._shuffle else list(range(self._size))

    def __len__(self):
        return self._size


class InferenceSampler(torchdata.Sampler):
    def __init__(self, size):
        self._size = size

    def __iter__(self):
        return iter(range(self._size))

    def __len__(self):
        return self._size
