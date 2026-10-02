import numpy as np
from enum import Enum

class Vein(Enum):
    '''Enumeration vein types with their corresponding labels, ids and colors.'''
    BODY = ("BODY", 0, (0, 255, 0))
    TAIL = ("TAIL", 1, (0, 0, 255))

    def __init__(self, label, id, color):
        self.label = label
        self.id = id
        self.color = np.array(color, dtype=np.uint8)

    @classmethod
    def from_id(cls, id):
        for vein in cls:
            if vein.id == id:
                return vein
        raise ValueError(f"No vein with id {id} found.")