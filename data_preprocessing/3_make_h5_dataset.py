import os
import cv2
import glob
import h5py
import imageio
import numpy as np
from PIL import Image
import pdb
import shutil

from tqdm import tqdm
from ctxray_utils import load_scan_mhda

def save_axis_image(ct_scan, root_path, prefix, slice_idx=None):
    if len(prefix) > 0:
        prefix = f"{prefix}_"
    for i, axis in zip([0, 1, 2], ['axial', 'coronal', 'sagittal']):
        if slice_idx is None:
            slice_idx = ct_scan.shape[i] // 2
        filename = os.path.join(root_path, f"{prefix}{axis}_output_gt_{slice_idx:0}.png")
        if i == 0:
            slice_gt = ct_scan[slice_idx]
        elif i == 1:
            slice_gt = ct_scan[:, slice_idx]
        else:
            slice_gt = ct_scan[..., slice_idx]
        imageio.imwrite(filename, slice_gt)

def make_h5_dataset_LIDC(xray_size = 128):
    import os
    root_dir = os.path.dirname(os.path.realpath(__file__))

    source_Path = f"{root_dir}/lidc_idri/LIDC-IDRI-MDH_ctpro_woMask/"
    save_ct_path = f"{root_dir}/lidc_idri/LIDC-HDF5-256_ct320"
    save_xray_path = f"{root_dir}/lidc_idri/LIDC-HDF5-256_ct320_plastimatch_xray"

    if not os.path.exists(save_ct_path):
        os.makedirs(save_ct_path)

    if not os.path.exists(f"{save_xray_path}"):
        os.makedirs(f"{save_xray_path}")

    dir_list = glob.glob(f"{source_Path}*")
    print(f"Number of data in source path : {len(dir_list)}")

    with open(f"{root_dir}/dataset_list/train.txt", 'r') as file:
        train_list = file.readlines()
    with open(f"{root_dir}/dataset_list/test.txt", 'r') as file:
        test_list = file.readlines()

    train_list.extend(test_list)
    train_list.sort()
    print(f"Number of data in txt files : {len(train_list)}")

    total_list_dict = {}
    for t in train_list:
        key = t.split(".")[0]
        value = t.split("\n")[0]
        if key not in total_list_dict.keys():
            total_list_dict[key] = value
        else:
            sub_key = total_list_dict[key].split(".")[-2]
            total_list_dict[key] = {sub_key: total_list_dict[key]}
            sub_key = t.split(".")[-2]
            total_list_dict[key][sub_key] = value  ## 5512, 3190 // 5472, 5405

    for dir_ in tqdm(dir_list):
        key = dir_.split("__")[0].split("\\")[-1]
        sub_folder_name = total_list_dict[key]
        if isinstance(sub_folder_name, dict):
            """
            LIDC-IDRI-0132, LIDC-IDRI-0132.20000101.3163.1
            LIDC-IDRI-0132, LIDC-IDRI-0132.20000101.5418.1
            LIDC-IDRI-0151, LIDC-IDRI-0151.20000101.3144.1
            LIDC-IDRI-0151, LIDC-IDRI-0151.20000101.5408.1
            LIDC-IDRI-0315, LIDC-IDRI-0315.20000101.5416.1
            LIDC-IDRI-0315, LIDC-IDRI-0315.20000101.5435.1
            LIDC-IDRI-0332, LIDC-IDRI-0332.20000101.30242.1
            LIDC-IDRI-0332, LIDC-IDRI-0332.20000101.9250.30078.1
            LIDC-IDRI-0355, LIDC-IDRI-0355.20000101.3190.1
            LIDC-IDRI-0355, LIDC-IDRI-0355.20000101.5512.1
            LIDC-IDRI-0365, LIDC-IDRI-0365.20000101.3175.1
            LIDC-IDRI-0365, LIDC-IDRI-0365.20000101.5409.1
            LIDC-IDRI-0442, LIDC-IDRI-0442.20000101.5405.1
            LIDC-IDRI-0442, LIDC-IDRI-0442.20000101.5472.1
            LIDC-IDRI-0484, LIDC-IDRI-0484.20000101.3100.2
            LIDC-IDRI-0484, LIDC-IDRI-0484.20000101.5417.1
            """
            sub_key = dir_.split("__")[2].split(".")[0]
            sub_folder_name = total_list_dict[key][sub_key]

        train_list.remove(f"{sub_folder_name}\n")
        save_subfolder_path = os.path.join(save_ct_path, sub_folder_name)
        if not os.path.exists(save_subfolder_path):
            os.makedirs(save_subfolder_path)  ## ct_xray_data.h5
        ct_path = glob.glob(f"{dir_}/*.mha")[0]
        _, ct_scan, _, _, _ = load_scan_mhda(ct_path)
        save_name = f"{save_subfolder_path}/ct_xray_data.h5"
        f = h5py.File(save_name, 'w')
        f['ct'] = ct_scan
        f.close()
        xray_direction1 = cv2.imread(f"{dir_}/xray1.png")[..., 0]
        xray_direction2 = cv2.imread(f"{dir_}/xray2.png")[..., 0]

        if xray_direction1.shape[1] != xray_size:
            xray_direction1 = Image.fromarray(xray_direction1)
            xray_direction1 = xray_direction1.resize((xray_size, xray_size), Image.ANTIALIAS)
            xray_direction1 = np.array(xray_direction1)
        if xray_direction2.shape[1] != xray_size:
            xray_direction2 = Image.fromarray(xray_direction2)
            xray_direction2 = xray_direction2.resize((xray_size, xray_size), Image.ANTIALIAS)
            xray_direction2 = np.array(xray_direction2)

        cv2.imwrite(f"{save_xray_path}/{sub_folder_name}_xray1.png", xray_direction1)
        cv2.imwrite(f"{save_xray_path}/{sub_folder_name}_xray2.png", xray_direction2)

    print(train_list)

if __name__ == "__main__":
    make_h5_dataset_LIDC()