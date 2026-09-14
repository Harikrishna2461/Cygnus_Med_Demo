class CfgNode(dict):
    def __init__(self, init_dict=None, **kwargs):
        super().__init__(init_dict or {})
        self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value
