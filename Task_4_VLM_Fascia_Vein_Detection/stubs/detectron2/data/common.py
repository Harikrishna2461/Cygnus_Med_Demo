import copy
import torch.utils.data as torchdata


class DatasetFromList(torchdata.Dataset):
    def __init__(self, lst, copy=True, serialize=False):
        self._lst = lst
        self._copy = copy

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        return copy.deepcopy(self._lst[idx]) if self._copy else self._lst[idx]


class MapDataset(torchdata.Dataset):
    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        return self._map_func(data)
