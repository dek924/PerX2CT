from torch.utils.data import Dataset
from importlib import import_module

class LIDCBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class LIDCTrain(LIDCBase):
    def __init__(self, size, training_images_list_file, dataset_class, opt):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        dataset_module = getattr(import_module("x2ct_nerf.data.base"), dataset_class)
        num_ctslice_per_item = opt.get('num_ctslice_per_item', 1)
        self.data = dataset_module(paths=paths, opt=opt, size=size, num_ctslice_per_item=num_ctslice_per_item)


class LIDCTest(LIDCBase):
    def __init__(self, size, test_images_list_file, dataset_class, opt):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        dataset_module = getattr(import_module("x2ct_nerf.data.base"), dataset_class)
        num_ctslice_per_item = opt.get('num_ctslice_per_item', 1)
        self.data = dataset_module(paths=paths, opt=opt, size=size, num_ctslice_per_item=num_ctslice_per_item)


class LIDCTrainwPrev(LIDCBase):
    def __init__(self, size, training_images_list_file, training_base_images_list_file, dataset_class, opt):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(training_base_images_list_file, "r") as f:
            base_paths = f.read().splitlines()

        dataset_module = getattr(import_module("x2ct_nerf.data.base"), dataset_class)
        self.data = dataset_module(paths=paths, opt=opt, size=size, base_paths=base_paths)


class LIDCTestwPrev(LIDCBase):
    def __init__(self, size, test_images_list_file, test_base_images_list_file, dataset_class, opt):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_base_images_list_file, "r") as f:
            base_paths = f.read().splitlines()
        dataset_module = getattr(import_module("x2ct_nerf.data.base"), dataset_class)
        self.data = dataset_module(paths=paths, opt=opt, size=size, base_paths=base_paths)