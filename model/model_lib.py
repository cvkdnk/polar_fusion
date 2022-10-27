from utils.pf_base_class import PFBaseClass


class ModelLibrary(PFBaseClass):
    MODEL = {}

    @classmethod
    def gen_config_template(cls):
        pass

    @classmethod
    def get_model(cls, name, config):
        return cls.MODEL[name](config)

    @staticmethod
    def register(model_class):
        ModelLibrary.MODEL[model_class.__name__] = model_class
        return model_class
