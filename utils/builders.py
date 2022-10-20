from torch.utils.data import Dataset


class DatasetBuilder(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def get_sub_class(self, subclass_name):
        for subclass in self.__subclasses__():
            if subclass.__name__ == subclass_name:
                return subclass
        raise Exception("Can't track this subclass [{}]".format(subclass_name))

    def get_config_params(self):
        return self.__init__.__code__.co_varnames[1:]


if __name__ == '__main__':
    db = DatasetBuilder()
    print("sub_classes: ", db.__subclasses__())
    print(db.get_config_params())