import inspect


class PFBaseClass:
    default_str = "Need To Be Completed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

