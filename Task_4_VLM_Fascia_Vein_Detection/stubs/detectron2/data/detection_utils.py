import numpy as np
from PIL import Image


def read_image(file_name, format=None):
    image = Image.open(file_name)
    if format == "BGR":
        image = image.convert("RGB")
        image = np.asarray(image)[:, :, ::-1].copy()
    elif format in ("RGB", None):
        image = np.asarray(image.convert("RGB"))
    elif format == "L":
        image = np.asarray(image.convert("L"))
    else:
        image = np.asarray(image.convert("RGB"))
    return image


def check_image_size(dataset_dict, image):
    if "width" in dataset_dict and "height" in dataset_dict:
        h, w = image.shape[:2]
        dataset_dict["height"] = h
        dataset_dict["width"] = w
