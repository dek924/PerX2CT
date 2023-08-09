import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

# For LIDC
import torch
import math
import h5py
import imageio
from x2ct_nerf.preprocessing.X2CT_transform_3d import List_Compose, Limit_Min_Max_Threshold, Normalization, \
    ToTensor, Resize_image


class LIDCOnlyOneInputType(Dataset):
    def __init__(self, paths, opt: dict, size=None, labels=None):
        self.opt = opt
        self.size = size
        self.input_type = opt['input_type']
        assert isinstance(self.input_type, str)

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.CT_MIN_MAX = opt["CT_MIN_MAX"]
        self.XRAY_MIN_MAX = opt["XRAY_MIN_MAX"]
        self.set_preprocessing()

    def __len__(self):
        return self._length

    def set_preprocessing(self):
        input_augment_list = self.opt["augment_list"]
        output_augment_list = []
        if self.input_type in ['ct', 'ctslice']:
            if 'min_max_th' in input_augment_list:
                output_augment_list.append((Limit_Min_Max_Threshold(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
            if 'normalization' in input_augment_list:
                output_augment_list.append((Normalization(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
            if 'resize' in input_augment_list:
                output_augment_list.append((Resize_image(self.RESIZE, self.RESIZE, self.RESIZE),))
        elif self.input_type in ['xray']:
            if 'normalization' in input_augment_list:
                output_augment_list.append((Normalization(self.XRAY_MIN_MAX[0], self.XRAY_MIN_MAX[1]),))
        output_augment_list.append((ToTensor(),))
        self.preprocessing = List_Compose(output_augment_list)

    def preprocess_image(self, image_path):
        ext = image_path.split(".")[-1]
        assert ext in ['png', 'h5']

        if ext in ['png']:
            image = imageio.imread(image_path)
            image = np.asarray(image)
        elif ext in ["h5"]:
            with h5py.File(image_path, 'r') as f:
                image = np.asarray(f['ct'])  # 128 x 128

        image = np.expand_dims(image, -1)
        image = np.concatenate((image, image, image), axis=-1)
        image = self.preprocessing(image)
        return image

    def __getitem__(self, i):
        example = dict()
        example[self.input_type] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class LIDCMultiInputTypes(Dataset):
    def __init__(self, paths, opt: dict, size=None, labels=None, num_ctslice_per_item=1):
        assert num_ctslice_per_item in [1, 3]

        self.opt = opt
        self.size = size
        self.input_types = opt['input_type']

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.num_ctslice_per_item = num_ctslice_per_item

        self.CT_MIN_MAX = opt["CT_MIN_MAX"]
        self.XRAY_MIN_MAX = opt["XRAY_MIN_MAX"]
        self.RESIZE = opt.get("RESIZE", 128)  # number of CT slice
        self.set_preprocessing()
        self.mapping_camera_type2pose = {
            "PA": torch.tensor([0, 0]),
            "Lateral": torch.tensor([math.pi / 2, math.pi / 2]),
        }

    def __len__(self):
        return self._length

    def set_preprocessing(self):
        dict_augment_list = {}
        for input_type in self.input_types:
            i_type = 'ct' if input_type in ['ct', 'ctslice'] else 'xray'
            dict_augment_list[input_type] = self.opt[f"{i_type}_augment_list"]

        self.dict_preprocessing = {}
        for input_type in self.input_types:
            augment_list = []
            if input_type in ['ct', 'ctslice']:
                if 'min_max_th' in dict_augment_list[input_type]:
                    augment_list.append((Limit_Min_Max_Threshold(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
                # if 'resize' in dict_augment_list[input_type]:
                #     augment_list.append((Resize_image(size=(self.opt.RESIZE[0], self.opt.RESIZE[1], self.opt.RESIZE[2])),))
            elif input_type in ['PA', 'Lateral']:
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.XRAY_MIN_MAX[0], self.XRAY_MIN_MAX[1]),))
            augment_list.append((ToTensor(),))
            self.dict_preprocessing[input_type] = List_Compose(augment_list)

    def get_image(self, image_path, data_type='ct'):
        ext = image_path.split(".")[-1]
        assert ext in ['png', 'h5']

        if ext in ['png']:
            image = imageio.imread(image_path)
            image = np.asarray(image)
        elif ext in ["h5"]:
            with h5py.File(image_path, 'r') as f:
                image = np.asarray(f[data_type])  # 128 x 128

        return image

    def apply_preprocessing_xray_according2cam(self, xray_img, src_camtype):
        assert src_camtype in ["PA", "Lateral"]
        if src_camtype == "PA":
            xray_img = np.fliplr(xray_img)
        elif src_camtype == "Lateral":
            xray_img = np.transpose(xray_img, (1, 0))
            xray_img = np.flipud(xray_img)
        src_campose = self.mapping_camera_type2pose[src_camtype]

        return xray_img, src_campose

    def get_ctslice(self, image_path):
        if self.num_ctslice_per_item == 1:
            image = self.get_image(image_path)
            image = np.expand_dims(image, -1)
            image = np.concatenate((image, image, image), axis=-1)
        else:  # self.num_ctslice_per_item == 3:
            curr_slice_id = image_path.split("_")[-1].split(".")[0]
            start_i = int(curr_slice_id) - (self.num_ctslice_per_item - 1) // 2
            image = None
            for i in range(self.num_ctslice_per_item):
                i = start_i + i
                if i < 0:
                    i = f"{0:0>3}"
                elif i > (self.RESIZE - 1):
                    i = f"{(self.RESIZE - 1):0>3}"
                else:
                    i = f"{i:0>3}"
                img_path = image_path.replace(f"{curr_slice_id}.h5", f"{i}.h5")
                img = self.get_image(img_path)
                img = np.expand_dims(img, -1)
                image = np.concatenate((image, img), axis=-1) if image is not None else img

        return image

    def __getitem__(self, i):
        example = dict()
        for index, input_type in enumerate(self.input_types):
            main_image_path = self.labels["file_path_"][i]
            # ex : ./data/LIDC-IDRI/220126/LIDC-HDF5-256_ct128_CTSlice/LIDC-IDRI-0982.20000101.30786.1/ct/coronal_104.h5
            if index == 0:  # ctslice
                image = self.get_ctslice(main_image_path)
                example[input_type] = self.dict_preprocessing[input_type](image)
            else:
                if input_type in ['PA', 'Lateral']:
                    path_dirs = main_image_path.split("/")
                    image_path = "/".join(path_dirs[:-3]).replace("CTSlice", f"{self.opt['rendering_type']}_xray")
                    if input_type == 'PA':
                        image_path = f"{image_path}/{path_dirs[-3]}_xray1.png"
                    else:
                        image_path = f"{image_path}/{path_dirs[-3]}_xray2.png"
                    image = self.get_image(image_path)
                    image, src_campose = self.apply_preprocessing_xray_according2cam(image, input_type)
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
                    example[input_type] = self.dict_preprocessing[input_type](image)
                    example[f"{input_type}_cam"] = src_campose

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class LIDCMultiInputTypesCT(Dataset):
    def __init__(self, paths, opt: dict, size=None, labels=None, **ignore_kwargs):
        self.opt = opt
        self.size = size
        self.input_types = opt['input_type']

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.CT_MIN_MAX = opt["CT_MIN_MAX"]
        self.XRAY_MIN_MAX = opt["XRAY_MIN_MAX"]
        self.RESIZE = opt.get("RESIZE", 128)  # number of CT slice
        self.set_preprocessing()
        self.mapping_camera_type2pose = {
            "PA": torch.tensor([0, 0]),
            "Lateral": torch.tensor([math.pi / 2, math.pi / 2]),
        }

    def __len__(self):
        return self._length

    def set_preprocessing(self):
        dict_augment_list = {}
        for input_type in self.input_types:
            i_type = 'ct' if input_type in ['ct', 'ctslice'] else 'xray'
            dict_augment_list[input_type] = self.opt[f"{i_type}_augment_list"]

        self.dict_preprocessing = {}
        for input_type in self.input_types:
            augment_list = []
            if input_type in ['ct', 'ctslice']:
                if 'min_max_th' in dict_augment_list[input_type]:
                    augment_list.append((Limit_Min_Max_Threshold(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
                if 'resize' in dict_augment_list[input_type]:
                    augment_list.append((Resize_image(size=(self.opt.RESIZE[0], self.opt.RESIZE[1], self.opt.RESIZE[2])),))
            elif input_type in ['PA', 'Lateral']:
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.XRAY_MIN_MAX[0], self.XRAY_MIN_MAX[1]),))
            augment_list.append((ToTensor(),))
            self.dict_preprocessing[input_type] = List_Compose(augment_list)

    def get_image(self, image_path, data_type='ct'):
        ext = image_path.split(".")[-1]
        assert ext in ['png', 'h5']

        if ext in ['png']:
            image = imageio.imread(image_path)
            image = np.asarray(image)
        elif ext in ["h5"]:
            with h5py.File(image_path, 'r') as f:
                image = np.asarray(f[data_type])  # 128 x 128
                if data_type == 'ct':
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
        return image

    def apply_preprocessing_xray_according2cam(self, xray_img, src_camtype):
        assert src_camtype in ["PA", "Lateral"]
        if src_camtype == "PA":
            xray_img = np.fliplr(xray_img)
        elif src_camtype == "Lateral":
            xray_img = np.transpose(xray_img, (1, 0))
            xray_img = np.flipud(xray_img)
        src_campose = self.mapping_camera_type2pose[src_camtype]

        return xray_img, src_campose

    def __getitem__(self, i):
        example = dict()
        for index, input_type in enumerate(self.input_types):
            main_image_path = self.labels["file_path_"][i]
            # ex : ./data/LIDC-IDRI/220126/LIDC-HDF5-256_ct128/LIDC-IDRI-0982.20000101.30786.1/ct_xray_data.h5
            if index == 0:  # ctdata
                image = self.get_image(main_image_path, data_type='ct')
                example[input_type] = self.dict_preprocessing[input_type](image)
            else:
                if input_type in ['PA', 'Lateral']:
                    path_dirs = main_image_path.split("/")
                    image_path = path_dirs[:-2]
                    image_path[-1] = image_path[-1] + f"_{self.opt['rendering_type']}_xray"
                    image_path = "/".join(image_path)
                    if input_type == 'PA':
                        image_path = f"{image_path}/{path_dirs[-2]}_xray1.png"
                    else:
                        image_path = f"{image_path}/{path_dirs[-2]}_xray2.png"
                    image = self.get_image(image_path)
                    image, src_campose = self.apply_preprocessing_xray_according2cam(image, input_type)
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
                    example[input_type] = self.dict_preprocessing[input_type](image)
                    example[f"{input_type}_cam"] = src_campose

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class COVIDMultiInputTypes(LIDCMultiInputTypes):
    def __init__(self, paths, opt: dict, size=None, labels=None, num_ctslice_per_item=1):
        super().__init__(paths, opt, size, labels, num_ctslice_per_item)

    def __getitem__(self, i):
        example = dict()
        for index, input_type in enumerate(self.input_types):
            main_image_path = self.labels["file_path_"][i]
            if index == 0:
                image_path = main_image_path
                image = self.get_image(image_path)
                image = np.expand_dims(image, -1)
                image = np.concatenate((image, image, image), axis=-1)
                example[input_type] = self.dict_preprocessing[input_type](image)
            else:
                if input_type in ['PA', 'Lateral']:
                    path_dirs = main_image_path.split("/")
                    image_path = "/".join(path_dirs[:-3]).replace("CTSlice", f"{self.opt['rendering_type']}_xray")
                    if input_type == 'PA':
                        image_path = f"{image_path}/{path_dirs[-3]}_xray1.png"
                    else:
                        image_path = f"{image_path}/{path_dirs[-3]}_xray2.png"
                    image = self.get_image(image_path)
                    image, src_campose = self.apply_preprocessing_xray_according2cam(image, input_type)
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
                    example[input_type] = self.dict_preprocessing[input_type](image)
                    example[f"{input_type}_cam"] = src_campose

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class LIDCMultiInputTypesSuperResolution(LIDCMultiInputTypes):
    def __init__(self, paths, opt: dict, size=None, labels=None, num_ctslice_per_item=1):
        super().__init__(paths, opt, size, labels, num_ctslice_per_item)
        self.base_data_dir = opt.base_data_dir

    def __getitem__(self, i):
        example = dict()
        for index, input_type in enumerate(self.input_types):
            main_image_path = self.labels["file_path_"][i]
            if index == 0:
                image_path = main_image_path
                image = self.get_image(image_path)
                image = np.expand_dims(image, -1)
                image = np.concatenate((image, image, image), axis=-1)
                example[input_type] = self.dict_preprocessing[input_type](image)
            else:
                if input_type in ['PA', 'Lateral']:
                    path_dirs = main_image_path.split("/")
                    path_dirs[3] = self.base_data_dir.split("/")[3]
                    image_path = "/".join(path_dirs[:-3]).replace("CTSlice", f"{self.opt['rendering_type']}_xray")
                    if input_type == 'PA':
                        image_path = f"{image_path}/{path_dirs[-3]}_xray1.png"
                    else:
                        image_path = f"{image_path}/{path_dirs[-3]}_xray2.png"
                    image = self.get_image(image_path)
                    image, src_campose = self.apply_preprocessing_xray_according2cam(image, input_type)
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
                    example[input_type] = self.dict_preprocessing[input_type](image)
                    example[f"{input_type}_cam"] = src_campose

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class LIDCMultiInputMultiResTypes(Dataset):
    def __init__(self, paths, opt: dict, size=None, labels=None, num_ctslice_per_item=1):
        assert num_ctslice_per_item in [1, 3]

        self.opt = opt
        self.ct_size = opt['ct_size']  # ct resolution, number of CT slice
        self.xray_size = opt['xray_size']  # xray_resolution
        self.input_types = opt['input_type']

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.num_ctslice_per_item = num_ctslice_per_item

        self.CT_MIN_MAX = opt["CT_MIN_MAX"]
        self.XRAY_MIN_MAX = opt["XRAY_MIN_MAX"]

        self.set_preprocessing()
        self.mapping_camera_type2pose = {
            "PA": torch.tensor([0, 0]),
            "Lateral": torch.tensor([math.pi / 2, math.pi / 2]),
        }

    def __len__(self):
        return self._length

    def set_preprocessing(self):
        dict_augment_list = {}
        for input_type in self.input_types:
            i_type = 'ct' if input_type in ['ct', 'ctslice'] else 'xray'
            dict_augment_list[input_type] = self.opt[f"{i_type}_augment_list"]

        self.dict_preprocessing = {}
        for input_type in self.input_types:
            augment_list = []
            if input_type in ['ct', 'ctslice']:
                if 'min_max_th' in dict_augment_list[input_type]:
                    augment_list.append((Limit_Min_Max_Threshold(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
            elif input_type in ['PA', 'Lateral']:
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.XRAY_MIN_MAX[0], self.XRAY_MIN_MAX[1]),))
            augment_list.append((ToTensor(),))
            self.dict_preprocessing[input_type] = List_Compose(augment_list)

    def get_image(self, image_path, data_type='ct'):
        ext = image_path.split(".")[-1]
        assert ext in ['png', 'h5']

        if ext in ['png']:
            image = imageio.imread(image_path)
            image = np.asarray(image)
        elif ext in ["h5"]:
            with h5py.File(image_path, 'r') as f:
                image = np.asarray(f[data_type])  # 128 x 128

        return image

    def apply_preprocessing_xray_according2cam(self, xray_img, src_camtype):
        assert src_camtype in ["PA", "Lateral"]
        if src_camtype == "PA":
            xray_img = np.fliplr(xray_img)
        elif src_camtype == "Lateral":
            xray_img = np.transpose(xray_img, (1, 0))
            xray_img = np.flipud(xray_img)
        src_campose = self.mapping_camera_type2pose[src_camtype]

        return xray_img, src_campose

    def get_ctslice(self, image_path):
        if self.num_ctslice_per_item == 1:
            image = self.get_image(image_path)
            image = np.expand_dims(image, -1)
            image = np.concatenate((image, image, image), axis=-1)
        else:  # self.num_ctslice_per_item == 3:
            curr_slice_id = image_path.split("_")[-1].split(".")[0]
            start_i = int(curr_slice_id) - (self.num_ctslice_per_item - 1) // 2
            image = None
            for i in range(self.num_ctslice_per_item):
                i = start_i + i
                if i < 0:
                    i = f"{0:0>3}"
                elif i > (self.ct_size - 1):
                    i = f"{(self.ct_size - 1):0>3}"
                else:
                    i = f"{i:0>3}"
                img_path = image_path.replace(f"{curr_slice_id}.h5", f"{i}.h5")
                img = self.get_image(img_path)
                img = np.expand_dims(img, -1)
                image = np.concatenate((image, img), axis=-1) if image is not None else img

        return image

    def __getitem__(self, i):
        example = dict()
        for index, input_type in enumerate(self.input_types):
            main_image_path = self.labels["file_path_"][i]
            # ex : ./data/LIDC-IDRI/220126/LIDC-HDF5-256_ct128_CTSlice/LIDC-IDRI-0982.20000101.30786.1/ct/coronal_104.h5
            if index == 0:  # ctslice
                image = self.get_ctslice(main_image_path)
                example[input_type] = self.dict_preprocessing[input_type](image)
            else:
                if input_type in ['PA', 'Lateral']:
                    path_dirs = main_image_path.split("/")
                    src_txt = f'ct{self.ct_size}_CTSlice' 
                    dst_txt = f"ct{self.xray_size}_{self.opt['rendering_type']}_xray"
                    image_path = "/".join(path_dirs[:-3]).replace(src_txt, dst_txt)
                    if input_type == 'PA':
                        image_path = f"{image_path}/{path_dirs[-3]}_xray1.png"
                    else:
                        image_path = f"{image_path}/{path_dirs[-3]}_xray2.png"
                    image = self.get_image(image_path)
                    image, src_campose = self.apply_preprocessing_xray_according2cam(image, input_type)
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
                    example[input_type] = self.dict_preprocessing[input_type](image)
                    example[f"{input_type}_cam"] = src_campose

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example