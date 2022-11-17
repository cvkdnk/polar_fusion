class PFBaseClass:
    default_str = "Need To Be Completed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError


class InterfaceBase(PFBaseClass):
    REGISTER = {}

    @classmethod
    def gen_config_template(cls, name=None):
        if isinstance(name, str):
            assert name in cls.REGISTER.keys(), f"{name} not found in {cls.REGISTER.keys()}"
            return_dict = {"name": name}
            return_dict.update(cls.REGISTER[name].gen_config_template())
            return return_dict
        else:
            raise ValueError

    @classmethod
    def get(cls, name, config):
        if isinstance(name, str):
            return cls.REGISTER[name](**config)
        class_type = cls.__name__.replace("Interface", "")
        raise TypeError(f"{class_type} in base.yaml should be str or list")

    @classmethod
    def register(cls, class_name):
        cls.REGISTER[class_name.__name__] = class_name
        return class_name

    @classmethod
    def gen_default(cls, name):
        return cls.get(name, cls.gen_config_template(name))
