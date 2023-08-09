import os
import SimpleITK as sitk
import pydicom as dc
import numpy as np
import glob
from tqdm import tqdm
import os
root_dir = os.path.dirname(os.path.realpath(__file__))

Folders_Path = f"{root_dir}/lidc_idri/manifest-1600709154662/LIDC-IDRI/"
save_path = f"{root_dir}/lidc_idri/LIDC-IDRI-MDH"
text_path = "/".join(save_path.split("/")[:-1])
text_path = f"{text_path}/dicom2mhd.txt"

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(text_path, 'w') as file:
    file.write("filename\tRescaleSlope\tRescaleIntercept\tDimension\tSpacing\tOrigin\n")

dir_list = glob.glob(f"{Folders_Path}*/*/*/")
#dir_list = dir_list[132:138]
for dir_ in tqdm(dir_list):
    files = glob.glob(f"{dir_}/*.dcm")
    if len(files) > 10:
        dir_names = dir_[len(Folders_Path):].split("/")
        save_fileName = "__".join(dir_names[:-1])

        DICOM_LIST = sorted(files)
        #To remove /n
        DICOM_LIST = [x.strip() for x in DICOM_LIST]
        ##Sometimes image number and index are not the same
        #To replace image index numebr with image number
        DICOM_LIST = [x[0:x.find(".dcm") + 4] for x in DICOM_LIST]

        # To get first image as refenece image, supposed all images have same dimensions
        ReferenceImage = dc.read_file(DICOM_LIST[0])
        with open(f"{save_path}\\{save_fileName}.txt", 'w') as file:
            file.write(f"{ReferenceImage}")

        # To get Dimensions
        Dimension = (int(ReferenceImage.Rows), int(ReferenceImage.Columns), len(DICOM_LIST))

        # To get 3D spacing
        Spacing = (float(ReferenceImage.PixelSpacing[0]), float(ReferenceImage.PixelSpacing[1]), float(ReferenceImage.SliceThickness))
        Rescale_slope = ReferenceImage.data_element("RescaleSlope").value
        Rescale_intercept = ReferenceImage.data_element("RescaleIntercept").value
        # To get image origin
        Origin = ReferenceImage.ImagePositionPatient

        # To make numpy array
        NpArrDc = np.zeros(Dimension, dtype=ReferenceImage.pixel_array.dtype)

        sliceIDs = []
        for filename in DICOM_LIST:
            image = dc.read_file(filename)
            sliceID = image.data_element("InstanceNumber").value - 1
            sliceIDs.append(sliceID)

        sliceIDs.sort()
        # loop through all the DICOM files
        for filename in DICOM_LIST:
            # To read the dicom file
            df = dc.read_file(filename)
            dimension = [int(df.Rows), int(df.Columns), len(df)]
            spacing = [float(df.PixelSpacing[0]), float(df.PixelSpacing[1]), float(df.SliceThickness)]
            origin = df.ImagePositionPatient
            rescale_slope = df.data_element("RescaleSlope").value
            rescale_intercept = df.data_element("RescaleIntercept").value
            image_pixel_array = df.pixel_array * rescale_slope + rescale_intercept
            # store the raw image data
            sliceID = df.data_element("InstanceNumber").value - 1
            sliceID = sliceIDs.index(sliceID)
            image_pixel_array += 1024
            NpArrDc[:, :, sliceID] = image_pixel_array
        with open(text_path, 'a') as file:
            file.write(f"{save_fileName}\t{rescale_slope}\t{rescale_intercept}\t{dimension}\t{spacing}\t{origin}\n")

        NpArrDc = np.transpose(NpArrDc, (2, 0, 1))
        sitk_img = sitk.GetImageFromArray(NpArrDc, isVector=False)
        sitk_img.SetSpacing(Spacing)
        sitk_img.SetOrigin(Origin)
        sitk.WriteImage(sitk_img, os.path.join(save_path, save_fileName + ".mhd"))