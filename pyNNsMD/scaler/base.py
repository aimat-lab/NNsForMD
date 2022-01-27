

class SaclerBase:

    def __init__(self, **kwargs):
        pass

    def transform(self, **kwargs):
        raise NotImplementedError("Must be implemented in sub-class.")

    def inverse_transform(self, **kwargs):
        raise NotImplementedError("Must be implemented in sub-class.")

    def fit(self, **kwargs):
        raise NotImplementedError("Must be implemented in sub-class.")

    def fit_transform(self, **kwargs):
        raise NotImplementedError("Must be implemented in sub-class.")

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

    def get_config(self):
        raise NotImplementedError("Must be implemented in sub-class.")

    @classmethod
    def from_config(cls, config):
        cls(**config)
