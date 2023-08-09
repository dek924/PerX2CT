import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
from importlib import import_module
import pdb

from x2ct_nerf.modules import volume_rendering as vr

# Model
class PerspectiveINRNet(nn.Module):
    def __init__(self, cfg):
        """ """
        super(PerspectiveINRNet, self).__init__()

        self.cfg = cfg
        self.cond_encoder = self.get_model(self.cfg['cond_encoder_module'])(**self.cfg['cond_encoder_params'])
        self.cond_encoder_output_ch = self.cond_encoder.output_ch
        featureonly = cfg.get("featureonly", False)
        print(f"featureonly : {featureonly}")
        if featureonly:
            self.cfg['nerf_params']['cfg']['input_ch'] = (self.cond_encoder_output_ch * self.cfg['N_cond'])
        else:
            self.cfg['nerf_params']['cfg']['input_ch'] += (self.cond_encoder_output_ch * self.cfg['N_cond'])

        self.nerf = self.get_model(self.cfg['nerf_module'])(**self.cfg['nerf_params'])
        self.nerf_input_ch = self.cfg['nerf_params']['cfg']['input_ch']
        self.output_ch = self.nerf.output_ch

    def get_model(self, module_name):
        module_, module_class = module_name.rsplit(".", 1)
        module_ = getattr(import_module(module_), module_class)
        return module_

    def forward(self, x):
        """
        output : dictionary type
        """
        x = x[..., -self.nerf_input_ch:]
        output = self.nerf(x)
        return output

    def encode(self, src_images, src_camposes, render_pts):
        """
        NS : number of source
        src_images : NS x B x C x H x W
        src_camposes : (NS x B x 2), 2 is pitch and yaw or None
        render_pts: (B, H * W * N_samples, 3)
        """
        if src_camposes is not None:
            features = None
            ## src_image: (B x C x H x W), src_campose : (B x 2)
            for src_image, src_campose in zip(src_images, src_camposes):
                feature = self.cond_encoder(src_image)  # (B * C * H * W, for clip, [:, 64, 80, 80])
                feature = self.convert_feature_from2d_to3d(feature, src_campose, render_pts)    ### batch x n_rays_steps x feature_ch
                features = torch.cat((features, feature), dim=-1) if features is not None else feature
        else:
            features = self.cond_encoder(src_images)
            features = features.repeat(1, render_pts.shape[1], 1)
        return features ## B x (H * W * N_samples) x (feature_ch * NS)

    def convert_feature_from2d_to3d(self, features, src_campose, render_pts):
        """
        features : (B x C x H x W)
        src_camposes : (B x 2), 2 is pitch and yaw
        render_pts: (B, H * W * N_samples, 3)
        """

        device = features.get_device()
        batch, n_rays_steps, _ = render_pts.shape

        ones = torch.ones((batch, n_rays_steps, 1))
        render_pts = torch.cat((render_pts, ones.cuda()), dim=-1)

        pitch = src_campose[..., 0:1]
        yaw = src_campose[..., 1:2]
        camera_origin_input, _, _ = vr.sample_camera_positions(n=batch, r=1, device=device, phi=pitch, theta=yaw)
        # if camera_origin_input[:, 0].unique()[0] == 0:
        camera_origin_input[:, 0][camera_origin_input[:, 0] == 0] = 1e-5
        forward_vector_input = vr.normalize_vecs(-camera_origin_input)
        cam2world = vr.create_cam2world_matrix(forward_vector_input, camera_origin_input, device=device)
        world2inputcam = torch.inverse(cam2world.float())

        ### batch x n_rays_steps x 4
        points_in_src = torch.bmm(world2inputcam, render_pts.permute(0, 2, 1)).permute(0, 2, 1)
        points_uv_in_src = -points_in_src[..., :2] / points_in_src[..., 2:3]  ### batch x n_rays_steps x 2

        points_xy_in_src = points_uv_in_src / np.tan((2 * math.pi * self.cfg['fov'] / 360) / 2)
        points_xy_in_src = torch.cat([points_xy_in_src[..., 0:1], -points_xy_in_src[..., 1:2]], -1) ### batch x n_rays_steps x 2

        points_xy_in_src = points_xy_in_src.unsqueeze(2)
        feature_in_src = F.grid_sample(features, points_xy_in_src, mode='bilinear', align_corners=True, padding_mode='zeros').permute(0, 2, 3, 1)
        feature_in_src = feature_in_src.reshape(batch, n_rays_steps, -1)  ### batch x n_rays_steps x feature_ch

        return feature_in_src


class OrthogonalINRNet(nn.Module):
    def __init__(self, cfg):
        """ """
        super(OrthogonalINRNet, self).__init__()

        self.cfg = cfg
        self.cond_encoder = self.get_model(self.cfg['cond_encoder_module'])(**self.cfg['cond_encoder_params'])
        self.cond_encoder_output_ch = self.cond_encoder.output_ch
        featureonly = cfg.get("featureonly", False)
        print(f"featureonly : {featureonly}")
        if featureonly:
            self.cfg['nerf_params']['cfg']['input_ch'] = (self.cond_encoder_output_ch * self.cfg['N_cond'])
        else:
            self.cfg['nerf_params']['cfg']['input_ch'] += (self.cond_encoder_output_ch * self.cfg['N_cond'])

        self.nerf = self.get_model(self.cfg['nerf_module'])(**self.cfg['nerf_params'])
        self.nerf_input_ch = self.cfg['nerf_params']['cfg']['input_ch']
        self.output_ch = self.nerf.output_ch

    def get_model(self, module_name):
        module_, module_class = module_name.rsplit(".", 1)
        module_ = getattr(import_module(module_), module_class)
        return module_

    def forward(self, x):
        """
        output : dictionary type
        """
        x = x[..., -self.nerf_input_ch:]
        output = self.nerf(x)
        return output

    def encode(self, src_images, src_camposes, render_pts):
        """
        NS : number of source
        src_images : NS x B x C x H x W
        src_camposes : (NS x B x 2), 2 is pitch and yaw or None
        render_pts: (B, H * W * N_samples, 3)
        """
        if src_camposes is not None:
            features = None
            ## src_image: (B x C x H x W), src_campose : (B x 2)
            for src_image, src_campose in zip(src_images, src_camposes):
                feature = self.cond_encoder(src_image)  # (B * C * H * W, for clip, [:, 64, 80, 80])
                feature = self.convert_feature_from2d_to3d(feature, src_campose, render_pts)    ### batch x n_rays_steps x feature_ch
                features = torch.cat((features, feature), dim=-1) if features is not None else feature
        else:
            features = self.cond_encoder(src_images)
            features = features.repeat(1, render_pts.shape[1], 1)
        return features ## B x (H * W * N_samples) x (feature_ch * NS)

    def convert_feature_from2d_to3d(self, features, src_campose, render_pts):
        """
        features : (B x C x H x W)
        src_camposes : (B x 2), 2 is pitch and yaw
        render_pts: (B, H * W * N_samples, 3)
        """
        device = features.get_device()
        batch, n_rays_steps, _ = render_pts.shape

        ones = torch.ones((batch, n_rays_steps, 1))
        render_pts = torch.cat((render_pts, ones.cuda()), dim=-1)

        pitch = src_campose[..., 0:1]
        yaw = src_campose[..., 1:2]
        camera_origin_input, _, _ = vr.sample_camera_positions(n=batch, r=1, device=device, phi=pitch, theta=yaw)

        forward_vector_input = vr.normalize_vecs(-camera_origin_input)
        cam2world = vr.create_cam2world_matrix(forward_vector_input, camera_origin_input, device=device)
        world2inputcam = torch.inverse(cam2world.float())

        ### batch x n_rays_steps x 4
        points_in_src = torch.bmm(world2inputcam, render_pts.permute(0, 2, 1)).permute(0, 2, 1)
        points_xy_in_src = -points_in_src[..., :2]
        points_xy_in_src = torch.cat([points_xy_in_src[..., 0:1], -points_xy_in_src[..., 1:2]], -1) ### batch x n_rays_steps x 2

        points_xy_in_src = points_xy_in_src.unsqueeze(2)
        feature_in_src = F.grid_sample(features, points_xy_in_src, mode='bilinear', align_corners=True, padding_mode='zeros').permute(0, 2, 3, 1)
        feature_in_src = feature_in_src.reshape(batch, n_rays_steps, -1)  ### batch x n_rays_steps x feature_ch

        return feature_in_src