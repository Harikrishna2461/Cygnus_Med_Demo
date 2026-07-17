class _MetaObj:
    def __getattr__(self, name): return None
    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DatasetCatalog:
    def __init__(self): self._funcs = {}

    def register(self, name, func):
        self._funcs[name] = func

    def get(self, name):
        if name in self._funcs:
            return self._funcs[name]()
        return []

    def __contains__(self, name): return name in self._funcs
    def __iter__(self): return iter(self._funcs)


class _MetadataCatalog:
    def __init__(self): self._data = {}

    def get(self, name):
        if name not in self._data:
            self._data[name] = _MetaObj()
        return self._data[name]

    def set(self, name, **kwargs):
        obj = self.get(name)
        for k, v in kwargs.items():
            setattr(obj, k, v)

    def __contains__(self, name): return name in self._data


DatasetCatalog = _DatasetCatalog()
MetadataCatalog = _MetadataCatalog()
