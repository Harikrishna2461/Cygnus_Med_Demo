class _Catalog:
    def __init__(self): self._data = {}
    def get(self, name): return self._data.get(name, _MetaObj())
    def set(self, name, metadata): self._data[name] = metadata
    def __contains__(self, name): return name in self._data

class _MetaObj:
    def __getattr__(self, name): return None
    def set(self, **kwargs): pass

MetadataCatalog = _Catalog()
DatasetCatalog = _Catalog()
