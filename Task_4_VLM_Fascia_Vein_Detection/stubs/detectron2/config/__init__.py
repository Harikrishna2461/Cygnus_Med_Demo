import functools

class CfgNode(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(name)
    def __setattr__(self, name, value): self[name] = value

def configurable(init_func=None, *, from_config=None):
    if init_func is None:
        return lambda f: configurable(f, from_config=from_config)
    @functools.wraps(init_func)
    def wrapper(*args, **kwargs):
        return init_func(*args, **kwargs)
    wrapper.from_config = from_config
    return wrapper
