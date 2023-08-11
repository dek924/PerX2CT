# Introduction
This is the method used to genereate synthetic Xray images from CT volumes, as introduced in the CVPR 2019 paper: [X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks.](https://arxiv.org/abs/1905.06902)

# Dependence
glob, scipy, numpy, cv2, matplotlib, pydicom, SiimpleITK, tqdm


# Usage Guidance

- Install the DRR software from https://www.plastimatch.org/ on a Windows computer, version 1.7.3.12-win64. 
- Download LIDC-IDRI dataset to data_preprocessing/lidc_idri from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254

- 0_dicom2mhd.py : to convert dicom to mdh.
    * Folders_Path : original dicom path
    * save_path : save dir
- 1_ctpro_wo_mask.py : to generate the normalized CT volumes.
    * root_path ： original CT volume root 
    * save_root_path ： save dir 
- 2_xraypro.py : to run plastimatch
    * use this code in plastimatch path (ex. 'C:/Program Files/Plastimatch/bin')
    * root_path ： the normalized data path, same as the save_root_path in 1_ctpro_wo_mask.py file
    * save_root_path ：the Xray output path
    * plasti_path ： DDR software executable file location 
- 3_make_h5_dataset.py
- 4_ct_dataset_prepare.py