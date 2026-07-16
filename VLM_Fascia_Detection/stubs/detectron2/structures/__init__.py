import torch
import numpy as np
from enum import IntEnum


class BoxMode(IntEnum):
    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3
    XYWHA_ABS = 4


class Boxes:
    def __init__(self, tensor):
        self.tensor = torch.as_tensor(tensor, dtype=torch.float32) if not isinstance(tensor, torch.Tensor) else tensor.float()
    def __len__(self): return self.tensor.shape[0]
    def to(self, device): return Boxes(self.tensor.to(device))
    @property
    def device(self): return self.tensor.device


class ImageList:
    def __init__(self, tensor, image_sizes):
        self.tensor = tensor
        self.image_sizes = image_sizes

    @staticmethod
    def from_tensors(tensors, size_divisibility=0, pad_value=0.0):
        assert len(tensors) > 0
        max_h = max(t.shape[-2] for t in tensors)
        max_w = max(t.shape[-1] for t in tensors)
        if size_divisibility > 1:
            import math
            max_h = int(math.ceil(max_h / size_divisibility) * size_divisibility)
            max_w = int(math.ceil(max_w / size_divisibility) * size_divisibility)
        image_sizes = [(t.shape[-2], t.shape[-1]) for t in tensors]
        batched = tensors[0].new_full((len(tensors), tensors[0].shape[0], max_h, max_w), pad_value)
        for img, pad_img in zip(tensors, batched):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return ImageList(batched.contiguous(), image_sizes)

    def __len__(self): return len(self.image_sizes)
    def __getitem__(self, idx): return self.tensor[idx]


class Instances:
    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        self._fields = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, name, value): self._fields[name] = value
    def get(self, name): return self._fields[name]
    def has(self, name): return name in self._fields
    def __setattr__(self, name, value):
        if name in ('image_size', '_fields'):
            super().__setattr__(name, value)
        else:
            self._fields[name] = value
    def __getattr__(self, name):
        if '_fields' in self.__dict__ and name in self._fields:
            return self._fields[name]
        raise AttributeError(f"Instances has no attribute '{name}'")
    def __len__(self): return len(next(iter(self._fields.values()))) if self._fields else 0
    def to(self, device):
        ret = Instances(self.image_size)
        for k, v in self._fields.items():
            ret.set(k, v.to(device) if hasattr(v, 'to') else v)
        return ret


class BitMasks:
    def __init__(self, tensor):
        self.tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor, dtype=torch.bool)
    def __len__(self): return self.tensor.shape[0]
    def to(self, device): return BitMasks(self.tensor.to(device))
    @property
    def device(self): return self.tensor.device


class ROIMasks:
    def __init__(self, tensor):
        self.tensor = tensor
    def to(self, device): return ROIMasks(self.tensor.to(device))


class PolygonMasks:
    def __init__(self, polygons): self.polygons = polygons
    def __len__(self): return len(self.polygons)


class Keypoints:
    def __init__(self, keypoints): self.tensor = torch.as_tensor(keypoints, dtype=torch.float32)


class RotatedBoxes(Boxes):
    pass


def pairwise_iou(boxes1, boxes2):
    return torch.zeros(len(boxes1), len(boxes2))
