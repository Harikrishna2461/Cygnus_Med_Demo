from enum import Enum

_base_name = "sam2.1_hiera_"
_base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

class SamModel(Enum):
    '''Enumeration of available SAM model sizes with their corresponding names and URLs.'''

    TINY = [f"{_base_name}tiny", f"{_base_name}t", f"{_base_url}/{_base_name}tiny.pt"]
    SMALL = [f"{_base_name}small", f"{_base_name}s", f"{_base_url}/{_base_name}small.pt"]
    BASE_PLUS = [f"{_base_name}base_plus", f"{_base_name}b+", f"{_base_url}/{_base_name}base_plus.pt"]
    LARGE = [f"{_base_name}large", f"{_base_name}l", f"{_base_url}/{_base_name}large.pt"]
    EDGETAM = ["edgetam_video", "edgetam", "./segmentation/checkpoints/edgetam.pt"]