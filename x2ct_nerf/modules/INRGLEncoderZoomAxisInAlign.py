"""Implicit generator for 3D volumes"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import copy
from importlib import import_module
from omegaconf import OmegaConf
import numpy as np
import pdb

from x2ct_nerf.modules.utils import set_requires_grad
from x2ct_nerf.modules.nerf import model_utils, nerf_helpers


class INRGLEncoderZoomAxisInAlignNeRF(nn.Module):
    print("Model Name : INRGLEncoderZoomAxisInAlignNeRF - Use global and local feature with NeRF(include DummyNeRF)")

    def __init__(self, params):
        super().__init__()
        # to dictionary
        params = OmegaConf.to_container(params, resolve=True)
        self.metadata = params
        self.N_rays_ctslice_grad_on = self.metadata["N_rand_recon"]
        self.npoints_per_chunk = self.metadata["chunk"]
        self.cond_list = self.metadata['cond_list']
        self.use_cond_encoder = True if len(self.cond_list) > 0 else False
        self.axis_in_position = self.metadata.get('axis_in_position', None)

        assert self.use_cond_encoder

        network_module = self.metadata['main_model_of_encoder']['network_module']
        network_module, net_class = network_module.rsplit(".", 1)
        assert net_class in ['PerspectiveConcatAfterINRGLNet']

        if net_class in ['PerspectiveConcatAfterINRGLNet']:
            cfg, self.network_query_fn = model_utils.update_nerf_params(**self.metadata['main_model_of_encoder']['params']['cfg']['nerf_params'])
            self.embed_fn, cfg['input_ch'] = nerf_helpers.get_embedder(cfg['multires'], cfg['i_embed'])
            n_cond = self.metadata['main_model_of_encoder']['params']['cfg']['N_cond']
            if self.metadata['main_model_of_encoder']['params']['cfg']['mn_input_type'] == 'global':
                cfg['input_ch'] += self.metadata['main_model_of_encoder']['params']['cfg']['cond_encoder_params']['cfg']['global_latent_dim'] * n_cond
            elif self.metadata['main_model_of_encoder']['params']['cfg']['mn_input_type'] == 'all':
                cfg['input_ch'] += self.metadata['main_model_of_encoder']['params']['cfg']['cond_encoder_params']['cfg']['global_latent_dim'] * \
                    n_cond + self.metadata['main_model_of_encoder']['params']['cfg']['cond_encoder_params']['cfg']['latent_dim'] * n_cond
            else:
                cfg['input_ch'] += self.metadata['main_model_of_encoder']['params']['cfg']['cond_encoder_params']['cfg']['latent_dim'] * n_cond

            if self.axis_in_position == 'before_inr':
                cfg['input_ch'] += self.metadata['axis_emb_dim']

            cfg['output_ch'] = cfg['output_color_ch']
            self.metadata['main_model_of_encoder']['params']['cfg']['nerf_params']['cfg'] = cfg
        network_module = getattr(import_module(network_module), net_class)
        self.net_class = net_class
        self.network_fn = network_module(**self.metadata['main_model_of_encoder']['params'])
        if self.axis_in_position == 'after_inr':
            self.network_fn.output_ch += self.metadata['axis_emb_dim']
        self.output_ch = self.network_fn.output_ch

        ##########################
        self.set_resolution()

        self.max_length_gt_coord = self.metadata["ct_res"]
        self.max_length_world_coord = 1
        self.center_gt_coord = (self.ct_res - 1) / 2  # (10 - 1) / 2 = 4.5

        # can be float
        self.min_zoom_size = torch.min(self.ct_res).item() / self.metadata['zoom_resolution_div']  # 128 / 4 = 32.0
        self.min_scale = self.metadata['zoom_min_scale']  # 0.25
        self.use_zoom_ratio = self.metadata.get('use_zoom_ratio', 0)

        self.axis_emb = nn.Embedding(3, self.metadata['axis_emb_dim'])  # sagittal, coronal, axial

    def set_resolution(self, full_resolution=False):
        self.ct_res = torch.tensor([self.metadata["ct_res"]] * 3)  # output
        if full_resolution:
            self.feature_res = self.ct_res
            self.fstep = 1
        else:
            self.feature_res = torch.tensor([self.metadata["feature_res"]] * 3)
            assert (self.metadata["ct_res"] % self.metadata["feature_res"]) == 0
            self.fstep = int(self.metadata["ct_res"] / self.metadata["feature_res"])

    def gt2world_coordinate(self, pts_in_gt):
        """
        """
        device = pts_in_gt.get_device()
        pts_in_gt = pts_in_gt - self.center_gt_coord.to(device)
        pts_in_world = (pts_in_gt / self.max_length_gt_coord * self.max_length_world_coord)
        return pts_in_world

    def world2gt_coordinate(self, pts_in_world):
        device = pts_in_world.get_device()
        pts_in_gt = pts_in_world / self.max_length_world_coord * self.max_length_gt_coord
        pts_in_gt = pts_in_gt + self.center_gt_coord.to(device)
        return pts_in_gt

    def get_p0_zoom_size(self, ct_size_xy):
        if random.random() < self.use_zoom_ratio:
            p0 = np.random.uniform(low=0, high=0, size=2).tolist()  # [0, 0]
            zoom_size = np.random.uniform(low=(ct_size_xy), high=(ct_size_xy), size=1)[0]  # 128
        else:
            min_p0 = self.min_zoom_size * self.min_scale   # 8.0
            zoom_size = np.random.uniform(low=self.min_zoom_size, high=(ct_size_xy), size=1)[0]  # 32
            # sampling point from -min_p0 to ct_size_xy + min_p0 - zoom_size
            max_p0 = ct_size_xy + min_p0 - zoom_size   # 128 + 8 - 32 = 104
            p0 = np.random.uniform(low=min_p0 * -1, high=max_p0, size=2).tolist()
        return p0, zoom_size

    def rendering_from_ctslice(self, p0, zoom_size, ct_slice, output_ct_res):

        input_ct_res = [i for i in ct_slice.shape[-2:]]  # 320 x 320
        output_ct_res = [output_ct_res for i in input_ct_res]  # 320 x 320

        p1 = p0 + zoom_size - 1  # -1 because torch.linspace include end point
        p0 = [p / (out_res - 1) * (in_res - 1) for p, in_res, out_res in zip(p0, input_ct_res, output_ct_res)]
        p1 = [p / (out_res - 1) * (in_res - 1) for p, in_res, out_res in zip(p1, input_ct_res, output_ct_res)]

        # Normalization : from int to float within (-1 ~ 1)
        p0 = [p / (res-1) * 2 - 1 for p, res in zip(p0, input_ct_res)]
        p1 = [p / (res-1) * 2 - 1 for p, res in zip(p1, input_ct_res)]  # 80 / 320 * 2 - 1

        pts = torch.meshgrid(torch.linspace(p0[0], p1[0], output_ct_res[0]),
                             torch.linspace(p0[1], p1[1], output_ct_res[1]))
        pts_z = torch.zeros_like(pts[0])
        pts = torch.cat((pts[0].unsqueeze(-1),
                         pts[1].unsqueeze(-1),
                         pts_z.unsqueeze(-1)), dim=-1)
        pts = pts.unsqueeze(0).unsqueeze(0).cuda()
        raw = F.grid_sample(ct_slice.unsqueeze(0), pts, mode='bilinear', align_corners=True, padding_mode='zeros').squeeze(0).permute(0, 1, 3, 2)
        raw = raw.repeat(1, 3, 1, 1)
        return raw

    def get_rays_for_no_rendering(self, inputs: dict, p0, zoom_size):
        """
        p0 : start point (include) : list or None
        zoom_size : zoom image size : float or None
        transformed_points: (batch_size, metadata.img_res, metadata.img_res, metadata.N_samples, 3)
        transformed_ray_directions_expanded: (batch_size, metadata.img_res, metadata.img_res, metadata.N_samples, 3)
        z_vals: (batch_size, metadata.img_res, metadata.img_res, metadata.N_samples, 1)
        """
        # gt_resolution = np.asarray(metadata.ct_res) ## 128, 128, 128

        batch_size, _, H, W = inputs[inputs['image_key']].shape
        sampling_resolution = self.ct_res
        transformed_points_in_gt = torch.meshgrid(torch.linspace(0, self.ct_res[0] - 1, sampling_resolution[0]),  # 0, 1, ..., 126, 127
                                                  torch.linspace(0, self.ct_res[1] - 1, sampling_resolution[1]),
                                                  torch.linspace(0, self.ct_res[2] - 1, sampling_resolution[2]))
        transformed_points_in_gt = torch.cat((transformed_points_in_gt[0].unsqueeze(-1),
                                              transformed_points_in_gt[1].unsqueeze(-1),
                                              transformed_points_in_gt[2].unsqueeze(-1)), dim=-1)
        transformed_points_in_gt = transformed_points_in_gt.unsqueeze(0).cuda()
        transformed_points = self.gt2world_coordinate(transformed_points_in_gt)
        device = transformed_points.get_device()
        self.ct_res = self.ct_res.to(device)
        if self.axis_emb.weight.get_device() != device:
            self.axis_emb = self.axis_emb.to(device)

        transformed_feature_points = []
        gt_ctslices = []
        axis = []
        for b in range(batch_size):
            file_path = inputs['file_path_'][b].split("/")[-1]
            recon_axis, slice_idx = os.path.splitext(file_path)[0].split("_")
            slice_idx = int(slice_idx)
            gt_ctslice = inputs[inputs['image_key']][b:b + 1]
            p0_, zoom_size_ = p0, zoom_size
            if recon_axis == 'sagittal':  # sagittal : img_res = (gt_resolution[-3], gt_resolution[-2]), N_samples = gt_resolution[-1]
                axis_emb = self.axis_emb(torch.LongTensor([[[0]]]).to(device))
                feature_point = transformed_points[0, ::self.fstep, ::self.fstep, slice_idx:slice_idx + 1, :]
                offset = (feature_point[1, 1, :, :] - feature_point[0, 0, :, :]) * (self.fstep-1) / self.fstep
                feature_point = feature_point + offset / 2
                if p0_ is None and zoom_size_ is None:
                    p0_, zoom_size_ = self.get_p0_zoom_size(torch.min(self.ct_res[:2]).item())
                shift_to_zero = torch.mul(feature_point[0, 0, 0, :2], (zoom_size_ / self.ct_res[:2])) - feature_point[0, 0, 0, :2]
                feature_point[..., 0] = feature_point[..., 0] * (zoom_size_ / self.ct_res[0]) + (p0_[0] / self.ct_res[0]) - shift_to_zero[0]
                feature_point[..., 1] = feature_point[..., 1] * (zoom_size_ / self.ct_res[1]) + (p0_[1] / self.ct_res[1]) - shift_to_zero[1]
            elif recon_axis == 'coronal':  # coronal  : img_res = (gt_resolution[-3], gt_resolution[-1]), N_samples = gt_resolution[-2]
                axis_emb = self.axis_emb(torch.LongTensor([[[1]]]).to(device))
                feature_point = transformed_points[0, ::self.fstep, slice_idx:slice_idx + 1, ::self.fstep, :]
                offset = (feature_point[1, :, 1, :] - feature_point[0, :, 0, :]) * (self.fstep-1) / self.fstep
                feature_point = feature_point + offset / 2
                if p0_ is None and zoom_size_ is None:
                    p0_, zoom_size_ = self.get_p0_zoom_size(torch.min(self.ct_res[:3:2]).item())
                shift_to_zero = torch.mul(feature_point[0, 0, 0, :3:2], (zoom_size_ / self.ct_res[:3:2])) - feature_point[0, 0, 0, :3:2]
                feature_point[..., 0] = feature_point[..., 0] * (zoom_size_ / self.ct_res[0]) + (p0_[0] / self.ct_res[0]) - shift_to_zero[0]
                feature_point[..., 2] = feature_point[..., 2] * (zoom_size_ / self.ct_res[2]) + (p0_[1] / self.ct_res[2]) - shift_to_zero[1]
                feature_point = feature_point.permute(0, 2, 1, 3)
            else:  # axial    : img_res = (gt_resolution[-2], gt_resolution[-1]), N_samples = gt_resolution[-3]
                axis_emb = self.axis_emb(torch.LongTensor([[[2]]]).to(device))
                feature_point = transformed_points[0, slice_idx:slice_idx + 1, ::self.fstep, ::self.fstep, :]  # .permute(1, 2, 0, 3)
                offset = (feature_point[:, 1, 1, :] - feature_point[:, 0, 0, :]) * (self.fstep-1) / self.fstep
                feature_point = feature_point + offset / 2
                if p0_ is None and zoom_size_ is None:
                    p0_, zoom_size_ = self.get_p0_zoom_size(torch.min(self.ct_res[1:]).item())
                shift_to_zero = torch.mul(feature_point[0, 0, 0, 1:], (zoom_size_ / self.ct_res[1:])) - feature_point[0, 0, 0, 1:]
                feature_point[..., 1] = feature_point[..., 1] * (zoom_size_ / self.ct_res[1]) + (p0_[0] / self.ct_res[1]) - shift_to_zero[0]
                feature_point[..., 2] = feature_point[..., 2] * (zoom_size_ / self.ct_res[2]) + (p0_[1] / self.ct_res[2]) - shift_to_zero[1]
                feature_point = feature_point.permute(1, 2, 0, 3)
            axis.append(recon_axis)
            axis_emb = axis_emb.repeat(feature_point.shape[0], feature_point.shape[1], feature_point.shape[2], 1)

            feature_point = torch.cat((feature_point, axis_emb), dim=-1)
            transformed_feature_points.append(feature_point)
            gt_ctslices.append(self.rendering_from_ctslice(p0_, zoom_size_, gt_ctslice, output_ct_res=sampling_resolution[0].item()))

        transformed_feature_points = torch.stack(transformed_feature_points)  # (batch, feature_res, feature_res, 1, 3)
        gt_ctslices = torch.cat(gt_ctslices, dim=0)
        return transformed_feature_points, gt_ctslices, axis

    def _get_nograd_nerf(self):
        nograd_nerf = copy.deepcopy(self.network_fn)
        set_requires_grad(nograd_nerf, False)
        return nograd_nerf

    def run_nerf(self, org_shape, transformed_points, nrays_grad_on, latent_zs_dict=None, full_render_partial_grad=False):
        """
        transformed_points: batch_size, H, W, N_samples, coord_dim (coord_dim : 3 + self.metadata['axis_emb_dim'])
        """

        (batch_size, H, W, N_samples) = org_shape
        global_inputs = latent_zs_dict['global']
        local_inputs = latent_zs_dict['local']
        nerf_inputs = transformed_points

        all_smp_idx = [i for i in range(0, (transformed_points.shape[1]))]
        if full_render_partial_grad:
            random.shuffle(all_smp_idx)
            grad_on_smp_idx = all_smp_idx[:(nrays_grad_on*N_samples)]
            grad_off_smp_idx = all_smp_idx[(nrays_grad_on*N_samples):]
            grad_on_smp_idx.sort()
            grad_off_smp_idx.sort()
        else:
            grad_on_smp_idx = all_smp_idx
            grad_off_smp_idx = []

        device = transformed_points.get_device()
        all_outputs = {}
        all_outputs['outputs'] = torch.zeros((batch_size, transformed_points.shape[1], self.output_ch)).to(device)

        for grad, smp_idxs in zip(["off", "on"], [grad_off_smp_idx, grad_on_smp_idx]):
            n_split = (len(smp_idxs) // self.npoints_per_chunk) + 1  # add 1 because of residual
            for split in range(n_split):
                smp_idx = smp_idxs[split * self.npoints_per_chunk:(split + 1) * self.npoints_per_chunk]
                if len(smp_idx) > 0:
                    nerf_input = nerf_inputs[:, smp_idx]
                    nerf_input, axis_info = nerf_input[..., :3], nerf_input[..., 3:]

                    local_input = local_inputs[:, smp_idx]
                    global_input = global_inputs
                    if grad == "off":  # Not use this code
                        with torch.no_grad():
                            nograd_nerf = self._get_nograd_nerf()
                            embedded = self.embed_fn(nerf_input)
                            if self.axis_in_position == 'before_inr':
                                embedded = torch.cat((embedded, axis_info), dim=-1)
                            all_output = nograd_nerf(embedded, local_input, global_input)
                            if self.axis_in_position == 'after_inr':
                                all_output['outputs'] = torch.cat((all_output['outputs'], axis_info), dim=-1)
                            for k in all_output:
                                all_output[k] = all_output[k].detach()
                            del nograd_nerf
                    else:
                        embedded = self.embed_fn(nerf_input)
                        if self.axis_in_position == 'before_inr':
                            embedded = torch.cat((embedded, axis_info), dim=-1)
                        all_output = self.network_fn(embedded, local_input, global_input)
                        if self.axis_in_position == 'after_inr':
                            all_output['outputs'] = torch.cat((all_output['outputs'], axis_info), dim=-1)

                    assert isinstance(all_output, dict)
                    for k in all_output:
                        if all_output[k].dtype != all_outputs[k].dtype:
                            all_outputs[k] = all_outputs[k].type(all_output[k].dtype)
                        all_outputs[k][:, smp_idx] = all_output[k]

        return all_outputs

    def forward(self, inputs: dict, full_render_partial_grad=False, gt_ct=None, p0=None, zoom_size=None):
        """
        inputs : from dataloader, must have image_key
        full_render_partial_grad : rendering full resolution but gradiendt is calculated partially
        """
        # Generate initial camera rays and sample points.

        if gt_ct is not None:
            self.set_resolution(full_resolution=True)

        with torch.no_grad():
            transformed_points, gt_ctslices, axis = self.get_rays_for_no_rendering(inputs, p0, zoom_size)
            nrays_grad_on = self.N_rays_ctslice_grad_on
            (batch_size, H, W, N_samples, coord_dim) = transformed_points.shape  # coord_dim : 3 + self.metadata['axis_emb_dim']
            transformed_points = transformed_points.reshape(batch_size, -1, coord_dim)

        if gt_ct is not None:
            all_outputs = self.rendering_from_gt((batch_size, H, W, N_samples), transformed_points[..., :3], gt_ct)
        else:
            src_imgs = []
            src_camposes = []
            for cond_key in self.cond_list:
                src_imgs.append(inputs[cond_key])
                src_camposes.append(inputs[f"{cond_key}_cam"])
            src_imgs = torch.stack(src_imgs, dim=0)  # src_images : NS x B x C x H x W
            src_camposes = torch.stack(src_camposes, dim=0)  # src_camposes : (NS x B x 2), 2 is pitch and yaw or None
            latent_zs_dict = self.network_fn.encode(src_imgs, src_camposes, transformed_points[..., :3])

            all_outputs = self.run_nerf((batch_size, H, W, N_samples), transformed_points, nrays_grad_on,
                                        latent_zs_dict, full_render_partial_grad=full_render_partial_grad)
            all_outputs['outputs'] = all_outputs['outputs'].permute(0, 2, 1)
            all_outputs['outputs'] = all_outputs['outputs'].reshape(batch_size, -1, H, W)

        all_outputs['cropped_ctslice'] = gt_ctslices
        return all_outputs

    def rendering_from_gt(self, org_shape, transformed_points, gt_ct):
        """
        org_shape : 128 x 128 x 128 x 1
        transformed_points.shape : 128 x 16384 x 3
        gt_ct.shape : 128 x 320 x 320
        """

        org_gt_ct_shape = gt_ct.shape
        gt_ct = gt_ct.unsqueeze(0)  # 1 x 128 x 128 x 128
        (batch_size, H, W, N_samples) = org_shape
        pts_in_world = transformed_points.reshape(batch_size, H * W, N_samples, 3)
        pts_in_gt = self.world2gt_coordinate(pts_in_world)

        device = pts_in_gt.get_device()
        shape = pts_in_gt.shape  # 128 x (128 x 128) x 1 x 3
        batch_idx = torch.zeros((shape[0], shape[1] * shape[2], 1))

        neighbor_idx = torch.round(pts_in_gt).reshape(shape[0], -1, 3).long()  # Nx3
        neighbor_idx = torch.clamp(neighbor_idx, min=0, max=(self.max_length_gt_coord - 1))
        neighbor_idx = torch.cat((batch_idx.to(device), neighbor_idx), dim=-1)
        neighbor_idx = neighbor_idx.reshape(-1, 4)
        neighbor_idx = neighbor_idx.permute(1, 0).data.cpu().numpy()  # 4 x N
        neighbor_idx = neighbor_idx.tolist()

        raw = gt_ct[neighbor_idx]
        raw = raw.reshape(org_gt_ct_shape[0], 1, org_gt_ct_shape[1], org_gt_ct_shape[2])
        raw = raw.repeat(1, 3, 1, 1)
        return {'outputs': raw}
