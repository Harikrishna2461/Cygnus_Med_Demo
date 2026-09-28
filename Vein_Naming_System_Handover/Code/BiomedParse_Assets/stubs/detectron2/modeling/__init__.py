from ..layers import ShapeSpec

class _Registry:
    def __init__(self, name): self.name = name; self._obj_map = {}
    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                self._obj_map[func_or_class.__name__] = func_or_class
                return func_or_class
            return deco
        self._obj_map[obj.__name__] = obj
        return obj
    def get(self, name): return self._obj_map.get(name)
    def __contains__(self, name): return name in self._obj_map

BACKBONE_REGISTRY = _Registry('BACKBONE')
SEM_SEG_HEADS_REGISTRY = _Registry('SEM_SEG_HEADS')

class Backbone:
    def output_shape(self): return {}
