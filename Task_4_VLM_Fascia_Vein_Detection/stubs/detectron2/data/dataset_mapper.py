class DatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

    def __call__(self, dataset_dict):
        return dataset_dict
