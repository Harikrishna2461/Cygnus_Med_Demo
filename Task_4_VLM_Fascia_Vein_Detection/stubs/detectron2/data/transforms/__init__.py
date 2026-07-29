import numpy as np
import cv2


def _resize_mask(mask, new_w, new_h):
    """cv2.resize squeezes (H,W,1) → (H',W'). Preserve channel dim."""
    ndim = mask.ndim
    out = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if ndim == 3 and out.ndim == 2:
        out = out[:, :, None]
    return out


class Transform:
    def apply_image(self, img): return img
    def apply_segmentation(self, mask): return mask


class TransformList:
    def __init__(self, transforms):
        self.transforms = transforms

    def apply_segmentation(self, mask):
        for t in self.transforms:
            mask = t.apply_segmentation(mask)
        return mask


class TransformGen:
    def get_transform(self, img): return Transform()


class RandomFlip(TransformGen):
    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def get_transform(self, img):
        do_flip = np.random.random() < self.prob
        h, v = self.horizontal, self.vertical

        class _T(Transform):
            def apply_image(self, img):
                if do_flip:
                    if h: img = img[:, ::-1].copy()
                    if v: img = img[::-1, :].copy()
                return img
            def apply_segmentation(self, mask):
                if do_flip:
                    if h: mask = mask[:, ::-1].copy()
                    if v: mask = mask[::-1, :].copy()
                return mask
        return _T()


class ResizeScale(TransformGen):
    def __init__(self, min_scale, max_scale, target_height, target_width):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.th = target_height
        self.tw = target_width

    def get_transform(self, img):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        new_h = int(self.th * scale)
        new_w = int(self.tw * scale)

        class _T(Transform):
            def apply_image(self, img):
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            def apply_segmentation(self, mask):
                return _resize_mask(mask, new_w, new_h)
        return _T()


class FixedSizeCrop(TransformGen):
    def __init__(self, crop_size):
        self.crop_h, self.crop_w = crop_size

    def get_transform(self, img):
        h, w = img.shape[:2]
        pad_h = max(0, self.crop_h - h)
        pad_w = max(0, self.crop_w - w)
        y0 = np.random.randint(0, max(1, h + pad_h - self.crop_h + 1))
        x0 = np.random.randint(0, max(1, w + pad_w - self.crop_w + 1))
        ch, cw, ph, pw = self.crop_h, self.crop_w, pad_h, pad_w

        class _T(Transform):
            def apply_image(self, img):
                if ph > 0 or pw > 0:
                    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
                return img[y0:y0+ch, x0:x0+cw]

            def apply_segmentation(self, mask):
                if mask.ndim == 3:
                    if ph > 0 or pw > 0:
                        mask = np.pad(mask, ((0, ph), (0, pw), (0, 0)), mode='reflect')
                    return mask[y0:y0+ch, x0:x0+cw]
                else:
                    if ph > 0 or pw > 0:
                        mask = np.pad(mask, ((0, ph), (0, pw)), mode='reflect')
                    return mask[y0:y0+ch, x0:x0+cw]
        return _T()


class ResizeShortestEdge(TransformGen):
    def __init__(self, short_edge_length, max_size=None, sample_style='choice', interp=cv2.INTER_LINEAR):
        if isinstance(short_edge_length, int):
            short_edge_length = [short_edge_length]
        self.short_edge_length = short_edge_length
        self.max_size = max_size or 99999

    def get_transform(self, img):
        h, w = img.shape[:2]
        size = np.random.choice(self.short_edge_length)
        scale = size / min(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)

        class _T(Transform):
            def apply_image(self, img):
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            def apply_segmentation(self, mask):
                return _resize_mask(mask, new_w, new_h)
        return _T()


def apply_transform_gens(transform_gens, img):
    applied = []
    for gen in transform_gens:
        t = gen.get_transform(img)
        img = t.apply_image(img)
        applied.append(t)
    return img, TransformList(applied)
