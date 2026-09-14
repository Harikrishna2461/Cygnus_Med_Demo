import os

class PathManager:
    @staticmethod
    def open(path, mode='r', **kwargs): return open(path, mode, **kwargs)
    @staticmethod
    def get_local_path(path, **kwargs): return path
    @staticmethod
    def isfile(path): return os.path.isfile(path)
    @staticmethod
    def exists(path): return os.path.exists(path)
    @staticmethod
    def ls(path): return os.listdir(path)
    @staticmethod
    def mkdirs(path): os.makedirs(path, exist_ok=True)
