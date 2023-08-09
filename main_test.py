import glob
import yaml
import torch
import imageio
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from importlib import import_module
from torchvision.transforms.functional import resized_crop

from main import instantiate_from_config
from x2ct_nerf.modules.losses.lpips import LPIPS
from utils.logger.TestLogger import ExperimentLogger
from utils.metrics import Peak_Signal_to_Noise_Rate_total, Peak_Signal_to_Noise_Rate_2D, Structural_Similarity_slice, Structural_Similarity, mse2psnr, img2mse, to8b


@torch.no_grad()
def sort_test_data(config):
    load_file_path = config['data']['params']['validation']['params']['test_images_list_file']
    save_file_path = load_file_path.replace(".txt", "_sorted.txt")

    print(save_file_path)
    with open(load_file_path, 'r') as file:
        sub_folders = file.readlines()
    sub_folders.sort()

    with open(save_file_path, 'w') as file:
        for file_name in sub_folders:
            file.write(f"{file_name}")


@torch.no_grad()
def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


@torch.no_grad()
def load_vqgan(config, model_module, ckpt_path=None):
    model_module = model_module.split(".")
    model_module, model_class = model_module[:-1], model_module[-1]
    model_module = ".".join(model_module)
    model_module = getattr(import_module(model_module), model_class)

    model = model_module(**config.model.params)

    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


@torch.no_grad()
def change_config_file_to_sorted(config, val_test):
    config['model']['params']['lossconfig'] = {'target': "taming.modules.losses.vqperceptual.DummyLoss"}
    test_path = config['data']['params']['validation']['params']['test_images_list_file'].split("/")
    config['data']['params']['validation']['params']['opt']['ct_size'] = 128

    test_path[-1] = f"{val_test}_sorted.txt"
    test_path = "/".join(test_path)
    config['data']['params']['validation']['params']['test_images_list_file'] = test_path
    config['data']['params']['batch_size'] = config['input_ct_res']
    return config


@torch.no_grad()
def change_config_file_to_fixed_test(config, batch=3):
    config['model']['params']['lossconfig'] = {'target': "taming.modules.losses.vqperceptual.DummyLoss"}
    test_path = config['data']['params']['validation']['params']['test_images_list_file'].split("/")
    test_path[-1] = f"test.txt"
    test_path = "/".join(test_path)
    config['data']['params']['validation']['params']['test_images_list_file'] = test_path
    config['data']['params']['batch_size'] = batch
    return config


@torch.no_grad()
def change_config_file_to_fixed_val(config, batch=3):
    config['model']['params']['lossconfig'] = {'target': "taming.modules.losses.vqperceptual.DummyLoss"}
    test_path = config['data']['params']['validation']['params']['test_images_list_file'].split("/")
    test_path[-1] = f"val.txt"
    test_path = "/".join(test_path)
    config['data']['params']['validation']['params']['test_images_list_file'] = test_path
    config['data']['params']['batch_size'] = batch
    return config


@torch.no_grad()
def change_config_file_to_fixed_train(config, batch=3):
    config['model']['params']['lossconfig'] = {'target': "taming.modules.losses.vqperceptual.DummyLoss"}
    test_path = config['data']['params']['validation']['params']['test_images_list_file'].split("/")
    test_path[-1] = f"train.txt"
    test_path = "/".join(test_path)
    config['data']['params']['validation']['params']['test_images_list_file'] = test_path
    config['data']['params']['batch_size'] = batch
    return config


@torch.no_grad()
def calculate_metric_per_axis(target, output, perceptual_loss_fn, ct_min_max):
    target_ = target.unsqueeze(1).repeat(1, 3, 1, 1)
    output_ = output.unsqueeze(1).repeat(1, 3, 1, 1)
    lpips = perceptual_loss_fn(target_.contiguous(), output_.contiguous()).mean()
    ### psnr and ssim
    test_ct_recon_psnr = mse2psnr(img2mse(output, target))
    test_psnr_dict = Peak_Signal_to_Noise_Rate_total(
        target.unsqueeze(0), output.unsqueeze(0), size_average=False, PIXEL_MIN=ct_min_max[0],
        PIXEL_MAX=ct_min_max[1]
    )
    test_ssim_dict = Structural_Similarity(target.unsqueeze(0).detach().cpu().numpy(),
                                           output.unsqueeze(0).detach().cpu().numpy(),
                                           PIXEL_MAX=1.0, size_average=True, channel_axis=False, gray_scale=True)
    return lpips, test_ct_recon_psnr.item(), test_psnr_dict[-1].item(), test_ssim_dict[-1]


@torch.no_grad()
def calculate_metric_per_axis_2D(target, output, perceptual_loss_fn, ct_min_max):
    target_ = target.unsqueeze(1).repeat(1, 3, 1, 1)
    output_ = output.unsqueeze(1).repeat(1, 3, 1, 1)
    lpips = perceptual_loss_fn(target_.contiguous(), output_.contiguous()).mean()
    ### psnr and ssim
    test_axis_psnr = Peak_Signal_to_Noise_Rate_2D(target.unsqueeze(0), output.unsqueeze(0),
                                                       PIXEL_MIN=ct_min_max[0], PIXEL_MAX=ct_min_max[1], use_real_max=True)
    test_axis_ssim = Structural_Similarity_slice(target.unsqueeze(0).detach().cpu().numpy(),
                                                         output.unsqueeze(0).detach().cpu().numpy(),
                                                         PIXEL_MAX=None, channel_axis=False)
    return lpips, test_axis_psnr.item(), test_axis_ssim.item()


@torch.no_grad()
def calculate_metric_per_patient(recon, gt, curr_axis, DEVICE, logger, ct_min_max, batch_idx, save_freq, dicts,
                                 testsavedir, prev_patient, perceptual_loss_fn, 
                                 total_ct_recon_lpips, total_ct_recon_psnr, total_ct_recon_ssim):
    test_lpips_avg = 0
    test_psnr_avg = 0
    test_ssim_avg = 0
    test_axis_count = 0

    for axis_name in ['axial', 'coronal', 'sagittal']:
        if len(recon.keys()) == 1 and curr_axis == 'axial':
            target = gt[curr_axis].to(DEVICE)  # 'axial', 'coronal', 'sagittal'
            output = recon[curr_axis].to(DEVICE)
            if axis_name == 'coronal':
                target = target.permute(1, 2, 0)
                output = output.permute(1, 2, 0)
            elif axis_name == 'sagittal':
                target = target.permute(2, 0, 1)
                output = output.permute(2, 0, 1)
        elif len(recon.keys()) == 3:
            target = gt[axis_name].to(DEVICE)  
            output = recon[axis_name].to(DEVICE) 

        lpips_axis, test_psnr_axis, test_ssim_axis = calculate_metric_per_axis_2D(target, output, perceptual_loss_fn, ct_min_max)
        logger.log_loss(loss_name=f"test_lpips_{axis_name}", loss=lpips_axis)
        logger.log_loss(loss_name=f"test_psnr_ct_{axis_name}", loss=test_psnr_axis)
        logger.log_loss(loss_name=f"test_ssim_ct_{axis_name}", loss=test_ssim_axis)

        test_axis_count += 1
        test_lpips_avg += lpips_axis
        test_psnr_avg += test_psnr_axis
        test_ssim_avg += test_ssim_axis

        total_ct_recon_lpips[axis_name] = total_ct_recon_lpips[
            axis_name] + lpips_axis if axis_name in total_ct_recon_lpips.keys() else lpips_axis
        total_ct_recon_psnr[axis_name] = total_ct_recon_psnr[
            axis_name] + test_psnr_axis if axis_name in total_ct_recon_psnr.keys() else test_psnr_axis
        total_ct_recon_ssim[axis_name] = total_ct_recon_ssim[
            axis_name] + test_ssim_axis if axis_name in total_ct_recon_ssim.keys() else test_ssim_axis

        if batch_idx % save_freq == 0 and dicts['save_result']:
            testsave_folder = f"{testsavedir}/{prev_patient}"
            Path(testsave_folder).mkdir(exist_ok=True, parents=True)
            output_img = output.data.cpu().numpy()
            target_img = target.data.cpu().numpy()

            for slice_idx in range(output_img.shape[0]):
                slice_pred = output_img[slice_idx]
                slice_gt = target_img[slice_idx]
                slice_pred = to8b(slice_pred)
                slice_gt = to8b(slice_gt)
                slice = np.concatenate((slice_pred, slice_gt), axis=1)
                filename = f"{testsave_folder}/{axis_name}_output_gt_{slice_idx:03d}.png"
                imageio.imwrite(filename, slice)

    test_lpips_avg = test_lpips_avg / test_axis_count
    test_psnr_avg = test_psnr_avg / test_axis_count
    test_ssim_avg = test_ssim_avg / test_axis_count

    logger.log.write(f"# of test axis: {test_axis_count} / axial, coronal, sagittal")
    logger.log_loss(loss_name=f"test_lpips_avg", loss=test_lpips_avg)
    logger.log_loss(loss_name=f"test_psnr_avg", loss=test_psnr_avg)
    logger.log_loss(loss_name=f"test_ssim_avg", loss=test_ssim_avg)

    total_ct_recon_lpips["total_ct_recon_lpips"] = total_ct_recon_lpips[
        "total_ct_recon_lpips"] + test_lpips_avg if "total_ct_recon_lpips" in total_ct_recon_lpips.keys() else test_lpips_avg
    total_ct_recon_psnr["total_ct_recon_psnr"] = total_ct_recon_psnr[
        "total_ct_recon_psnr"] + test_psnr_avg if "total_ct_recon_psnr" in total_ct_recon_psnr.keys() else test_psnr_avg
    total_ct_recon_ssim["total_ct_recon_ssim"] = total_ct_recon_ssim[
        "total_ct_recon_ssim"] + test_ssim_avg if "total_ct_recon_ssim" in total_ct_recon_ssim.keys() else test_ssim_avg

    return total_ct_recon_lpips, total_ct_recon_psnr, total_ct_recon_ssim


@torch.no_grad()
def run_test_ramdom_image(dicts, prefix):
    # few random slice
    assert prefix is not None
    saved_log_root = dicts['saved_log_root']
    val_test = dicts['val_test']
    testsavedir = f"{saved_log_root}/{val_test}_3dCT/{dicts['sub_folder']}"

    for batch_idx, batch in enumerate(tqdm(dicts['data'].val_dataloader())):
        # Inference one batch
        log = dicts['model'].log_images(batch, split='val', p0=None, zoom_size=None)
        test_output_img_full = log['reconstructions'][:, 0]
        test_gt_img_full = log[dicts['model'].gt_key][:, 0]

        test_output_img_full = test_output_img_full.data.cpu().numpy()
        test_gt_img_full = test_gt_img_full.data.cpu().numpy()
        testsave_folder = f"{testsavedir}"
        Path(testsave_folder).mkdir(exist_ok=True, parents=True)

        slices = None
        for slice_idx in range(test_output_img_full.shape[0]):
            slice_pred = to8b(test_output_img_full[slice_idx])
            slice_target = to8b(test_gt_img_full[slice_idx])
            slice = np.concatenate((slice_pred, slice_target), axis=1)
            slices = np.concatenate((slices, slice), axis=0) if slices is not None else slice

        filename = f"{testsave_folder}/{prefix}_OutputFull_GTFull.png"
        imageio.imwrite(filename, slices)
        return


@torch.no_grad()
def run_test(dicts, config):
    saved_log_root = dicts['saved_log_root']
    val_test = dicts['val_test']
    save_freq = 1
    DEVICE = dicts['device']

    ct_min_max = config['data']['params']['train']['params']['opt']['CT_MIN_MAX']
    testsavedir = f"{saved_log_root}/{val_test}_3dCT/{dicts['sub_folder']}"
    logger = ExperimentLogger(testsavedir)
    perceptual_loss_fn = LPIPS().eval().to(DEVICE)

    count = 0
    total_ct_recon_lpips = {}
    total_ct_recon_psnr = {}
    total_ct_recon_ssim = {}

    recon = {}
    gt = {}
    prev_patient = None
    for batch_idx, batch in enumerate(tqdm(dicts['data'].val_dataloader())):
        for k, v in batch.items():
            num_itr = v.shape[0] // dicts['sub_batch_size']
            break

        # Get output, target per patient
        file_path_ = batch['file_path_'][0].split("/")
        patient = file_path_[-3]
        curr_axis = file_path_[-1].split("_")[0]

        test_output_img = None
        test_target_img = None
        for it in range(num_itr):
            sub_batch = {}
            for k, v in batch.items():
                sub_batch[k] = v[it*dicts['sub_batch_size']:(it+1)*dicts['sub_batch_size']]

            log = dicts['model'].log_images(sub_batch, split='val')
            sub_test_output_img = log['reconstructions'][:, 0]
            sub_test_target_img = log['inputs'][:, 0]
            test_output_img = torch.cat((test_output_img, sub_test_output_img), dim=0) if test_output_img is not None else sub_test_output_img
            test_target_img = torch.cat((test_target_img, sub_test_target_img), dim=0) if test_target_img is not None else sub_test_target_img

        if prev_patient is None or prev_patient == patient:
            gt[curr_axis] = test_target_img.cpu()
            recon[curr_axis] = test_output_img.cpu()
            prev_patient = patient
        else:
            # calculate metric
            logger.log.write("--------------------")
            logger.log.write(f"Patient : {prev_patient}")

            total_ct_recon_lpips, total_ct_recon_psnr, total_ct_recon_ssim = calculate_metric_per_patient(
                recon, gt, curr_axis, DEVICE, logger, ct_min_max, batch_idx, save_freq,
                dicts, testsavedir, prev_patient, perceptual_loss_fn,
                total_ct_recon_lpips, total_ct_recon_psnr, total_ct_recon_ssim)
            count += 1
            gt[curr_axis] = test_target_img.cpu()
            recon[curr_axis] = test_output_img.cpu()
            prev_patient = patient

    logger.log.write("--------------------")
    logger.log.write(f"Patient : {prev_patient}")
    total_ct_recon_lpips, total_ct_recon_psnr, total_ct_recon_ssim = calculate_metric_per_patient(
        recon, gt, curr_axis, DEVICE, logger, ct_min_max, batch_idx, save_freq,
        dicts, testsavedir, prev_patient, perceptual_loss_fn,
        total_ct_recon_lpips, total_ct_recon_psnr, total_ct_recon_ssim)
    count += 1

    logger.log.write("---------- final results ----------")
    for k, v in total_ct_recon_lpips.items():
        k = k if k.split("_")[0] == 'total' else f"lpips_{k}"
        logger.log_loss(loss_name=k, loss=v / count)
    for k, v in total_ct_recon_psnr.items():
        k = k if k.split("_")[0] == 'total' else f"psnr_{k}"
        logger.log_loss(loss_name=k, loss=v / count)
    for k, v in total_ct_recon_ssim.items():
        k = k if k.split("_")[0] == 'total' else f"ssim_{k}"
        logger.log_loss(loss_name=k, loss=v / count)


@torch.no_grad()
def main_test(args):
    saved_log_root = args.path
    ckpt_name = args.ckpt_name
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_file = glob.glob(f"{saved_log_root}/configs/*-project.yaml")[0]
    config = load_config(config_file, display=False)
    config['model']['params']['metadata']['encoder_params']['params']['zoom_resolution_div'] = 1
    config['model']['params']['metadata']['encoder_params']['params']['zoom_min_scale'] = 0

    if args.sub_batch_size is None:
        args.sub_batch_size = config['input_ct_res']
    assert config['input_ct_res'] % args.sub_batch_size == 0

    # Set config
    # prefix "None" for run_test, others for get random sample
    configs = [change_config_file_to_fixed_train(deepcopy(config)),
               change_config_file_to_fixed_val(deepcopy(config)),
               change_config_file_to_sorted(deepcopy(config), args.val_test)
               ]
    prefixs = ['train', args.val_test, None]
    dataset_type = config['data']['params']['validation']['params']['test_images_list_file'].split("/")[-2]

    for prefix, cfg in zip(prefixs, configs):
        model_module = cfg['model']['target']
        model_file = f"{saved_log_root}/checkpoints/{ckpt_name}"
        model = load_vqgan(config=cfg, model_module=model_module, ckpt_path=model_file).to(DEVICE)

        # dataloader
        data = instantiate_from_config(cfg.data)
        data.prepare_data()
        data.setup()
        dicts = {"saved_log_root": saved_log_root, 'sub_folder': ckpt_name.replace(".ckpt", ""),
                 "data": data, "model": model, "device": DEVICE, 'dataset_type': dataset_type,
                 "val_test": args.val_test, "save_result": args.save_result, "sub_batch_size": args.sub_batch_size,
                 "p0": None, "zoom_size": None}
        if prefix is None:
            run_test(dicts, cfg)
        else:
            run_test_ramdom_image(dicts, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text to Radiance Fields')
    parser.add_argument('--path', type=str, help='path to saved_folder')
    parser.add_argument('--ckpt_name', type=str, default='last.ckpt', help='ckpt file name')
    parser.add_argument('--val_test', type=str, default='val', help='val or test')
    parser.add_argument('--save_result', action='store_false', help='save result, default = True, Set this to False to find best model')
    parser.add_argument('--sub_batch_size', type=int, default=None)

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    print()
    print(args.path)
    main_test(args)