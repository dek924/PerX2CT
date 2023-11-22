import os
import json
import glob
import yaml
import torch
import imageio
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from importlib import import_module

from main import instantiate_from_config
from x2ct_nerf.modules.losses.lpips import LPIPS
from utils.metrics import mse2psnr, img2mse, to8b
from utils.metrics import Peak_Signal_to_Noise_Rate_total, Peak_Signal_to_Noise_Rate_2D, Structural_Similarity_slice, Structural_Similarity
from utils.logger.TestLogger import ExperimentLogger
from torchvision.transforms.functional import resized_crop


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
    test_axis_psnr_x2ct = Peak_Signal_to_Noise_Rate_2D(target.unsqueeze(0), output.unsqueeze(0),
                                                       PIXEL_MIN=ct_min_max[0], PIXEL_MAX=ct_min_max[1], use_real_max=True)
    test_axis_ssim_default = Structural_Similarity_slice(target.unsqueeze(0).detach().cpu().numpy(),
                                                         output.unsqueeze(0).detach().cpu().numpy(),
                                                         PIXEL_MAX=None, channel_axis=False)
    return lpips, test_axis_psnr_x2ct.item(), test_axis_ssim_default.item()


@torch.no_grad()
def calculate_metric_per_patient_wo_logging(recons, gt, curr_axis, DEVICE, ct_min_max, slice_save_freq, dicts,
                                            testsavedir, prev_patient, perceptual_loss_fn, dicts_metrics = []):
    """
    recons : list - [zoom, interpolation]
        recons[0] : dict - axial or coronal or siagittal
            recons[0][axial] : (p0, ct0, ct1, ct2)
    gt : dict - axial or coronal or siagittal
        gt['axial'] : (p0, ct0, ct1, ct2)
    """

    assert len(recons) == len(dicts_metrics)

    axis_name = curr_axis
    target = gt[axis_name].to(DEVICE)
    outputs = []
    for recon in recons:
        if len(recon) > 0:
            output = recon[axis_name].to(DEVICE)
        else:
            output = []
        outputs.append(output)

    n_p0 = len(target)
    for i in range(len(outputs)):
        if len(outputs[i]) > 0:
            test_dict = define_eval_dict(0)
            for tar, out in zip(target, outputs[i]):
                lpips_axis, test_psnr_axis_x2ct, test_ssim_axis_default = calculate_metric_per_axis_2D(
                    tar, out,
                    perceptual_loss_fn,
                    ct_min_max)

                test_dict['lpips'] += lpips_axis.item()
                test_dict['psnr'] += test_psnr_axis_x2ct
                test_dict['ssim'] += test_ssim_axis_default

            for mmm, value in test_dict.items():  ## lpips, ...
                value = value / n_p0
                dicts_metrics[i][mmm][curr_axis] = value

    if dicts['save_result']:
        testsave_folder = f"{testsavedir}/{prev_patient}"
        Path(testsave_folder).mkdir(exist_ok=True, parents=True)
        output_zoom_img = outputs[0].data.cpu().numpy()
        output_inter_img = outputs[1].data.cpu().numpy() if len(outputs[1]) > 0 else None
        target_img = target.data.cpu().numpy()
        for slice_idx in range(target_img.shape[1]):
            if slice_idx % slice_save_freq == 0 :
                slices = None
                for i_p0 in range(target_img.shape[0]):
                    slice_zoom = to8b(output_zoom_img[i_p0][slice_idx])
                    slice_inter = to8b(output_inter_img[i_p0][slice_idx]) if output_inter_img is not None else None
                    slice_gt = to8b(target_img[i_p0][slice_idx])
                    if slice_inter is None:
                        slice = np.concatenate((slice_zoom, slice_gt), axis=1)
                        filename = f"{testsave_folder}/{axis_name}_zoom_gt_{slice_idx:03d}.png"
                    else:
                        slice = np.concatenate((slice_inter, slice_zoom, slice_gt), axis=1)
                        filename = f"{testsave_folder}/{axis_name}_inter_zoom_gt_{slice_idx:03d}.png"

                    slices = np.concatenate((slices, slice), axis=0) if slices is not None else slice
                imageio.imwrite(filename, slices)

    return dicts_metrics


@torch.no_grad()
def run_test_ramdom_image(dicts, prefix):
    # few random slice
    assert prefix is not None
    saved_log_root = dicts['saved_log_root']
    val_test = dicts['val_test']
    testsavedir = f"{saved_log_root}/{val_test}_3dCT/{dicts['sub_folder']}"

    for batch_idx, batch in enumerate(tqdm(dicts['data'].val_dataloader())):
        p0_zoom_list = [[(48.0, 48.0), np.float64(32.0)]]

        # Inference one batch
        for i, (p0, zoom_size) in enumerate(p0_zoom_list):
            log = dicts['model'].log_images(batch, split='val', p0=None, zoom_size=None)
            output_img_full = log['reconstructions'][:, 0]
            gt_img_full = log[dicts['model'].gt_key][:, 0]

            log = dicts['model'].log_images(batch, split='val', p0=p0, zoom_size=zoom_size)
            output_img_zoomin = log['reconstructions'][:, 0]
            gt_img_zoomin = log[dicts['model'].gt_key][:, 0]

            output_img_zoomin = output_img_zoomin.data.cpu().numpy()
            gt_img_zoomin = gt_img_zoomin.data.cpu().numpy()

            output_img_inter = resized_crop(output_img_full, int(p0[1]), int(p0[0]), int(zoom_size), int(zoom_size), output_img_zoomin.shape[-1])
            output_img_full = output_img_full.data.cpu().numpy()
            gt_img_full = gt_img_full.data.cpu().numpy()
            output_img_inter = output_img_inter.cpu().numpy()

            testsave_folder = f"{testsavedir}"
            Path(testsave_folder).mkdir(exist_ok=True, parents=True)

            slices = None
            for slice_idx in range(gt_img_full.shape[0]):
                pred_zoomin = to8b(output_img_zoomin[slice_idx])
                gt_zoomin = to8b(gt_img_zoomin[slice_idx])
                pred_full = to8b(output_img_full[slice_idx])
                gt_full = to8b(gt_img_full[slice_idx])
                pred_inter = to8b(output_img_inter[slice_idx])

                slice = np.concatenate((pred_zoomin, pred_inter, gt_zoomin, pred_full, gt_full), axis=1)
                slices = np.concatenate((slices, slice), axis=0) if slices is not None else slice

            filename = f"{testsave_folder}/{prefix}_predZoom_predInter_gtZoom_predFull_gtFull_From({p0[0], p0[1]})_Size{zoom_size}.png"
            imageio.imwrite(filename, slices)
        return


def define_eval_dict(sub_type={}):
    dicts = {
        'lpips': sub_type,
        'psnr': sub_type,
        'ssim': sub_type
    }
    return dicts

@torch.no_grad()
def run_test(dicts, config):
    # Whole CT
    saved_log_root = dicts['saved_log_root']
    val_test = dicts['val_test']
    slice_save_freq = 1
    DEVICE = dicts['device']
    intp = dicts['intp']

    prefix_axis_for_save = ['axial', 'coronal', 'sagittal']

    ct_min_max = config['data']['params']['train']['params']['opt']['CT_MIN_MAX']
    testsavedir = f"{saved_log_root}/{val_test}_3dCT/{dicts['sub_folder']}"
    if dicts['zoom_size'] is not None:
        testsavedir = f"{testsavedir}/zoom_size_{int(dicts['zoom_size'])}"

    logger = ExperimentLogger(testsavedir)
    if intp:
        logger_inter = ExperimentLogger(testsavedir, log_file='log_inter.log', is_print=False)
    perceptual_loss_fn = LPIPS().eval().to(DEVICE)

    recon_zoom = {}
    recon_inter = {}
    gt = {}

    num_patients = 0
    prev_patient = None

    total_zoom_dict = define_eval_dict()
    total_inter_dict = define_eval_dict()
    patient_zoom_dict = define_eval_dict()
    patient_inter_dict = define_eval_dict()
    for key in patient_zoom_dict.keys():
        patient_zoom_dict[key] = {axis: 0 for axis in prefix_axis_for_save}
        patient_inter_dict[key] = {axis: 0 for axis in prefix_axis_for_save}
        total_zoom_dict[key] = {axis: 0 for axis in prefix_axis_for_save}
        total_inter_dict[key] = {axis: 0 for axis in prefix_axis_for_save}

    for batch_idx, batch in enumerate(tqdm(dicts['data'].val_dataloader())):
        # per patient per axis
        file_path_ = batch['file_path_'][0].split("/")
        patient = file_path_[-3]
        curr_axis = file_path_[-1].split("_")[0]
        num_itr = batch[list(batch.keys())[0]].shape[0]

        zoom_size = dicts['zoom_size']
        p0 = dicts['p0'][patient]   # 5 x 128
        num_p0_per_slice = len(p0)
        if intp:
            logger_inter.log.write("--------------------")
            logger_inter.log.write(f"Patient : {patient}")

        if prev_patient is None or prev_patient == patient:
            pass
        else:
            num_patients += 1
            patient_zoom_dict = define_eval_dict()
            patient_inter_dict = define_eval_dict()
            for key in patient_zoom_dict.keys():
                patient_zoom_dict[key] = {axis: 0 for axis in prefix_axis_for_save}
                patient_inter_dict[key] = {axis: 0 for axis in prefix_axis_for_save}

        logger.log.write("--------------------")
        logger.log.write(f"Patient : {patient} [{curr_axis}]")
        output_imgs_zoom = []
        target_imgs_zoom = []
        output_imgs_inter = []
        for _p0 in range(num_p0_per_slice): ## num of crop
            output_img_zoom = None
            target_img_zoom = None
            output_img_inter = None

            for it in range(num_itr):   ## num of slices
                sub_batch = {}
                for k, v in batch.items():
                    sub_batch[k] = v[it*dicts['sub_batch_size']:(it+1)*dicts['sub_batch_size']]

                if intp:
                    log = dicts['model'].log_images(sub_batch, split='val', p0=None, zoom_size=None)
                    output_img_full = log['reconstructions'][:, 0]

                log = dicts['model'].log_images(sub_batch, split='val', p0=p0[_p0][it], zoom_size=zoom_size)
                sub_output_img_zoom = log['reconstructions'][:, 0]
                sub_gt_img_zoom = log[dicts['model'].gt_key][:, 0]

                output_img_zoom = torch.cat((output_img_zoom, sub_output_img_zoom), dim=0) if output_img_zoom is not None else sub_output_img_zoom
                target_img_zoom = torch.cat((target_img_zoom, sub_gt_img_zoom), dim=0) if target_img_zoom is not None else sub_gt_img_zoom

                if intp:
                    sub_output_img_inter = resized_crop(output_img_full, int(p0[_p0][it][1]), int(p0[_p0][it][0]), int(zoom_size), int(zoom_size), sub_output_img_zoom.shape[-1])
                    output_img_inter = torch.cat((output_img_inter, sub_output_img_inter), dim=0) if output_img_inter is not None else sub_output_img_inter

            output_imgs_zoom.append(output_img_zoom.cpu())
            target_imgs_zoom.append(target_img_zoom.cpu())
            if intp:
                output_imgs_inter.append(output_img_inter.cpu())

        gt[curr_axis] = torch.stack(target_imgs_zoom, 0)
        recon_zoom[curr_axis] = torch.stack(output_imgs_zoom, 0)
        if intp:
            recon_inter[curr_axis] = torch.stack(output_imgs_inter, 0)

        [patient_zoom_dict, patient_inter_dict] = calculate_metric_per_patient_wo_logging(
            [recon_zoom, recon_inter], gt, curr_axis, DEVICE, ct_min_max, slice_save_freq,
            dicts, testsavedir, patient, perceptual_loss_fn, dicts_metrics=[patient_zoom_dict, patient_inter_dict])

        for mmm in patient_zoom_dict.keys():    ## metric
            logger.log_loss(loss_name=f"test_{mmm}_{curr_axis}", loss=patient_zoom_dict[mmm][curr_axis])

        for mmm in patient_zoom_dict.keys():
            total_zoom_dict[mmm][curr_axis] += patient_zoom_dict[mmm][curr_axis]

        if intp:
            for key in patient_inter_dict.keys():
                logger_inter.log_loss(loss_name=f"test_{key}_{curr_axis}", loss=patient_inter_dict[key][curr_axis])
            for key in patient_inter_dict.keys():
                total_inter_dict[key][curr_axis] += patient_inter_dict[key][curr_axis]

        prev_patient = patient

    logger.log.write("---------- final results ----------")
    for metric in total_zoom_dict.keys():
        for axis, v in total_zoom_dict[metric].items():
            logger.log_loss(loss_name=f"{metric}_{axis}", loss=v / num_patients)
    for metric in total_zoom_dict.keys():
        logger.log_loss(loss_name=f"{metric}_total", loss=np.mean(np.asarray([k for k in total_zoom_dict[metric].values()])) / num_patients)

    if intp:
        logger_inter.log.write("---------- final results ----------")
        for metric in total_inter_dict.keys():
            for axis, v in total_inter_dict[metric].items():
                logger_inter.log_loss(loss_name=f"{metric}_{axis}", loss=v / num_patients)
        for metric in total_inter_dict.keys():
            logger_inter.log_loss(loss_name=f"{metric}_total", loss=np.mean(np.asarray([k for k in total_inter_dict[metric].values()])) / num_patients)


@torch.no_grad()
def main_test(args):
    saved_log_root = args.save_dir
    ckpt_name = args.ckpt_path.split("/")[-1] if args.ckpt_path is not None else args.ckpt_name
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.config_path is not None:
        config_file = args.config_path
    else:
        config_file = glob.glob(f"{saved_log_root}/configs/*-project.yaml")[0]
    
    config = load_config(config_file, display=False)
    config['model']['params']['metadata']['encoder_params']['params']['zoom_resolution_div'] = 1
    config['model']['params']['metadata']['encoder_params']['params']['zoom_min_scale'] = 0

    configs = [change_config_file_to_sorted(deepcopy(config), args.val_test)]
    prefixs = [None]

    dataset_type = config['data']['params']['validation']['params']['test_images_list_file'].split("/")[-2]
    for prefix, cfg in zip(prefixs, configs):
        model_module = cfg['model']['target']
        if args.ckpt_path is not None:
            model_file = args.ckpt_path
        elif args.ckpt_name is not None:
            model_file = f"{saved_log_root}/checkpoints/{ckpt_name}"
        else:
            raise ValueError("ckpt_path or ckpt_name should be provided")
        model = load_vqgan(config=cfg, model_module=model_module, ckpt_path=model_file).to(DEVICE)

        # dataloader
        data = instantiate_from_config(cfg.data)
        data.prepare_data()
        data.setup()
        dicts = {"saved_log_root": saved_log_root, 'sub_folder': ckpt_name.replace(".ckpt", ""),
                 "data": data, "model": model, "device": DEVICE, 'dataset_type': dataset_type,
                 "val_test": args.val_test, "save_result": args.save_result, "sub_batch_size": 1,
                 "p0": args.p0, "zoom_size": args.zoom_size, "intp": args.intp}
        run_test(dicts, cfg)


def main_test_random_crop(args):
    saved_log_root = args.save_dir
    ckpt_name = args.ckpt_path.split("/")[-1] if args.ckpt_path is not None else args.ckpt_name
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_file = glob.glob(f"{saved_log_root}/configs/*-project.yaml")[0]
    config = load_config(config_file, display=False)
    config['model']['params']['metadata']['encoder_params']['params']['zoom_resolution_div'] = 1
    config['model']['params']['metadata']['encoder_params']['params']['zoom_min_scale'] = 0

    # Set config
    configs = [change_config_file_to_fixed_train(deepcopy(config)),
               change_config_file_to_fixed_val(deepcopy(config))]
    prefixs = ['train', args.val_test]
    dataset_type = config['data']['params']['validation']['params']['test_images_list_file'].split("/")[-2]

    for prefix, cfg in zip(prefixs, configs):
        model_module = cfg['model']['target']
        if args.ckpt_path is not None:
            model_file = args.ckpt_path
        elif args.ckpt_name is not None:
            model_file = f"{saved_log_root}/checkpoints/{ckpt_name}"
        else:
            raise ValueError("ckpt_path or ckpt_name should be provided")
        model = load_vqgan(config=cfg, model_module=model_module, ckpt_path=model_file).to(DEVICE)

        # dataloader
        cfg['data']['params']['batch_size'] = 128
        data = instantiate_from_config(cfg.data)
        data.prepare_data()
        data.setup()
        dicts = {"saved_log_root": saved_log_root, 'sub_folder': ckpt_name.replace(".ckpt", ""),
                 "data": data, "model": model, "device": DEVICE, 'dataset_type': dataset_type,
                 "val_test": args.val_test, "save_result": args.save_result, "sub_batch_size": 1}
        run_test_ramdom_image(dicts, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text to Radiance Fields')
    parser.add_argument('--save_dir', type=str, help='path to saved_folder')
    parser.add_argument('--val_test', type=str, default='val', help='val or test')
    parser.add_argument('--save_result', action='store_false', help='save result, default = True, Set this to False to find best model')
    parser.add_argument('--zoom_size', type=int, default=None, help='zoom image size')
    parser.add_argument('--intp', action='store_true', default=False, help='interpolation')
    parser.add_argument('--config_path', type=str, default="./configs/PerX2CT_global_w_zoomin.yaml")
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints/PerX2CT_global.ckpt")
    parser.add_argument('--ckpt_name', type=str, default='last.ckpt', help='ckpt file name')

    args = parser.parse_args()
    assert (args.ckpt_path is not None) or (args.ckpt_name is not None)
    torch.set_grad_enabled(False)

    if args.zoom_size is None:
        for zoom_size in [64, 32]:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dataset_list/LIDC_zoom_eval/zoom_size_{zoom_size}.json'), 'r') as f:
                args.p0 = json.load(f)
            args.zoom_size = np.float64(zoom_size)

            print()
            print(args.save_dir)
            main_test(args)

    else:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dataset_list/LIDC_zoom_eval/zoom_size_{args.zoom_size}.json'), 'r') as f:
            args.p0 = json.load(f)
        args.zoom_size = np.float64(args.zoom_size)

        print()
        print(args.save_dir)
        main_test(args)
