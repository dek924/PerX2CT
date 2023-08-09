# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import torch
import numpy as np
import numpy.linalg as linalg
from skimage.metrics import structural_similarity as SSIM

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.get_device()))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def tensor_back_to_unMinMax(input_image, min, max):
  image = input_image * (max - min) + min
  return image

def Peak_Signal_to_Noise_Rate_total(tensor1, tensor2, size_average=True, PIXEL_MIN=0.0, PIXEL_MAX=1.0, use_real_max=False):
    arr1 = tensor_back_to_unMinMax(tensor1, PIXEL_MIN, PIXEL_MAX).type(torch.int32)
    arr2 = tensor_back_to_unMinMax(tensor2, PIXEL_MIN, PIXEL_MAX).type(torch.int32)
    arr1 = arr1.data.cpu().numpy()
    arr2 = arr2.data.cpu().numpy()

    PIXEL_MAX = 4095 if use_real_max else PIXEL_MAX
    psnr_d, psnr_h, psnr_w, psnr_avg = Peak_Signal_to_Noise_Rate(arr1, arr2, size_average, PIXEL_MAX)
    psnr_3d_data = Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average, PIXEL_MAX)
    return psnr_d, psnr_h, psnr_w, psnr_avg, psnr_3d_data


def Peak_Signal_to_Noise_Rate_2D(tensor1, tensor2, size_average=True, PIXEL_MIN=0.0, PIXEL_MAX=1.0, use_real_max=False):
    arr1 = tensor_back_to_unMinMax(tensor1, PIXEL_MIN, PIXEL_MAX).type(torch.int32)
    arr2 = tensor_back_to_unMinMax(tensor2, PIXEL_MIN, PIXEL_MAX).type(torch.int32)
    arr1 = arr1.data.cpu().numpy().astype(np.float64)
    arr2 = arr2.data.cpu().numpy().astype(np.float64)

    PIXEL_MAX = 4095 if use_real_max else PIXEL_MAX

    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    psnr[zero_mse] = 100
    psnr = psnr.mean(1)

    return psnr

def Peak_Signal_to_Noise_Rate_w_mask(tensor1, tensor2, mask, size_average=True, PIXEL_MIN=0.0, PIXEL_MAX=1.0, use_real_max=False):
    arr1 = tensor_back_to_unMinMax(tensor1, PIXEL_MIN, PIXEL_MAX).type(torch.int32)
    arr2 = tensor_back_to_unMinMax(tensor2, PIXEL_MIN, PIXEL_MAX).type(torch.int32)
    arr1 = arr1.data.cpu().numpy()[mask]
    arr2 = arr2.data.cpu().numpy()[mask]

    PIXEL_MAX = 4095 if use_real_max else PIXEL_MAX
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean()

    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr

##############################################
"""
3D Metrics
"""
##############################################


def MAE(arr1, arr2, size_average=True):
    """
    :param arr1:
      Format-[NDHW], OriImage
    :param arr2:
      Format-[NDHW], ComparedImage
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    if size_average:
        return np.abs(arr1 - arr2).mean()
    else:
        return np.abs(arr1 - arr2).mean(1).mean(1).mean(1)


def MSE(arr1, arr2, size_average=True):
    """
    :param arr1:
      Format-[NDHW], OriImage
    :param arr2:
      Format-[NDHW], ComparedImage
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    if size_average:
        return np.power(arr1 - arr2, 2).mean()
    else:
        return np.power(arr1 - arr2, 2).mean(1).mean(1).mean(1)


def Cosine_Similarity(arr1, arr2, size_average=True):
    """
    :param arr1:
      Format-[NDHW], OriImage
    :param arr2:
      Format-[NDHW], ComparedImage
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    arr1_squeeze = arr1.reshape((arr1.shape[0], -1))
    arr2_squeeze = arr2.reshape((arr2.shape[0], -1))
    cosineSimilarity = np.sum(arr1_squeeze * arr2_squeeze, 1) / (linalg.norm(arr2_squeeze, axis=1) * linalg.norm(arr1_squeeze, axis=1))
    if size_average:
        return cosineSimilarity.mean()
    else:
        return cosineSimilarity


def Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
      Format-[NDHW], OriImage [0,1]
    :param arr2:
      Format-[NDHW], ComparedImage [0,1]
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # #zero mse, return 100
    psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr


##############################################
"""
2D Metrics
"""
##############################################
def Peak_Signal_to_Noise_Rate(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
      Format-[NDHW], OriImage [0,1]
    :param arr2:
      Format-[NDHW], ComparedImage [0,1]
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    # Depth
    mse_d = se.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(2)
    zero_mse = np.where(mse_d == 0)
    mse_d[zero_mse] = eps
    psnr_d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_d))
    # #zero mse, return 100
    psnr_d[zero_mse] = 100
    psnr_d = psnr_d.mean(1)

    # Height
    mse_h = se.mean(axis=1, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(1)
    zero_mse = np.where(mse_h == 0)
    mse_h[zero_mse] = eps
    psnr_h = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_h))
    # #zero mse, return 100
    psnr_h[zero_mse] = 100
    psnr_h = psnr_h.mean(1)

    # Width
    mse_w = se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
    zero_mse = np.where(mse_w == 0)
    mse_w[zero_mse] = eps
    psnr_w = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_w))
    # #zero mse, return 100
    psnr_w[zero_mse] = 100
    psnr_w = psnr_w.mean(1)

    psnr_avg = (psnr_h + psnr_d + psnr_w) / 3
    if size_average:
        return [psnr_d.mean(), psnr_h.mean(), psnr_w.mean(), psnr_avg.mean()]
    else:
        return [psnr_d, psnr_h, psnr_w, psnr_avg]


def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0, channel_axis=True, gray_scale=False):
    """
    :param arr1:
      Format-[NDHW], OriImage [0,1]
    :param arr2:
      Format-[NDHW], ComparedImage [0,1]
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    if gray_scale:
      arr1_d = arr1_d.squeeze()
      arr2_d = arr2_d.squeeze()
      if N == 1:
        arr1_d = np.expand_dims(arr1_d, 0)
        arr2_d = np.expand_dims(arr2_d, 0)

    for i in range(N):   
        ssim = SSIM(arr1_d[i], arr2_d[i], data_range=PIXEL_MAX, channel_axis=channel_axis)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height, PA
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    if gray_scale:
      arr1_h = arr1_h.squeeze()
      arr2_h = arr2_h.squeeze()
      if N == 1:
        arr1_h = np.expand_dims(arr1_h, 0)
        arr2_h = np.expand_dims(arr2_h, 0)

    for i in range(N):
        ssim = SSIM(arr1_h[i], arr2_h[i], data_range=PIXEL_MAX, channel_axis=channel_axis)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    if gray_scale:
      arr1 = arr1.squeeze()
      arr2 = arr2.squeeze()
      if N == 1:
        arr1 = np.expand_dims(arr1, 0)
        arr2 = np.expand_dims(arr2, 0)
        
    for i in range(N):
        ssim = SSIM(arr1[i], arr2[i], data_range=PIXEL_MAX, channel_axis=channel_axis)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return [ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
    else:
        return [ssim_d, ssim_h, ssim_w, ssim_avg]


def Structural_Similarity_slice(arr1, arr2, PIXEL_MAX=1.0, channel_axis=1):
    """
    :param arr1: B x C x H x W, OriImage [0,1]
    :param arr2: B x C x H x W, ComparedImage [0,1]
    :param channel_axis : which axis is channel?
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    ssim = SSIM(arr1, arr2, data_range=PIXEL_MAX, channel_axis=channel_axis)
    return ssim


#################
# Test
#################
def psnr(img_1, img_2, PIXEL_MAX=255.0):
    mse = np.mean((img_1 - img_2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


if __name__ == "__main__":
    img1 = np.random.randint(0, 256, size=(1, 20, 20, 20), dtype=np.int64)
    img2 = np.random.randint(0, 256, size=(1, 20, 20, 20), dtype=np.int64)
    # print(img1, img2, 'aa')
    img11 = img1 / 255.0
    img21 = img2 / 255.0
    print(img11.shape, type(img11))
    print(psnr(img1[0], img2[0], 255), psnr(img11[0], img21[0], 1.0))
    print(Peak_Signal_to_Noise_Rate_3D(img1, img2, PIXEL_MAX=255), Peak_Signal_to_Noise_Rate_3D(img11, img21, PIXEL_MAX=1.0))
    print(Peak_Signal_to_Noise_Rate(img1, img2, PIXEL_MAX=255), Peak_Signal_to_Noise_Rate(img11, img21, PIXEL_MAX=1.0))