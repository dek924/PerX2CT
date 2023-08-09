### from pigan

"""
Differentiable volumetric implementation used by pi-GAN generator.
"""
import math
import numpy as np
import torch.nn.functional as F
import random
import torch


def get_rays(args, batch_size=1, pitch=None, yaw=None):
    """
    transformed_points: (batch_size, args.img_res * args.img_res, args.N_samples, 3)
    transformed_ray_directions_expanded: (batch_size, args.img_res * args.img_res, args.N_samples, 3)
    z_vals: (batch_size, args.img_res * args.img_res, args.N_samples, 1)
    """
    # points_cam : rendering points in ray, z_vals: sampling distance btw near/far, rays_d_cam: ray direction
    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, args.N_samples,
                                                           resolution=(args.img_res, args.img_res),
                                                           fov=args.fov, ray_start=args.near,
                                                           ray_end=args.far)  # batch_size, pixels, num_steps, 1

    # c2w
    transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
        points_cam, z_vals, rays_d_cam, h_stddev=args.h_stddev, v_stddev=args.v_stddev, h_mean=args.h_mean,
        v_mean=args.v_mean,
        pitch=pitch, yaw=yaw, perturb=args.perturb)

    transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, args.N_samples, -1)
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size,
                                                                                      args.img_res, args.img_res,
                                                                                      args.N_samples,
                                                                                      3)
    transformed_points = transformed_points.reshape(batch_size, args.img_res, args.img_res, args.N_samples, 3)
    z_vals = z_vals.reshape(batch_size, args.img_res, args.img_res, args.N_samples, 1)
    return transformed_points, transformed_ray_directions_expanded, z_vals, pitch, yaw


def get_rays_for_no_rendering(args, gt_renderer, trained_ct_res=None):
    """
    trained_ct_res: if trained_ct_res is different from visualize ct_res
    transformed_points: (batch_size, args.img_res, args.img_res, args.N_samples, 3)
    transformed_ray_directions_expanded: (batch_size, args.img_res, args.img_res, args.N_samples, 3)
    z_vals: (batch_size, args.img_res, args.img_res, args.N_samples, 1)

    """
    # gt_resolution = np.asarray(args.ct_res) + 2 ## 322, 322, 322
    gt_resolution = gt_renderer.ct_res
    trained_ct_res = torch.from_numpy(gt_resolution).to(torch.float32).cuda() if trained_ct_res is None else torch.tensor(trained_ct_res).cuda() + 2
    batch_size = args.batch_size
    # if len(gt_resolution) == 4: ## batch x axial x coronal x sagittal
    #     batch_size = gt_resolution[0]
    #     gt_resolution = gt_resolution[1:]
    # else:
    #     batch_size = 1
    # skip_slice = gt_resolution[-1] // args.N_samples
    # slice_idx = np.random.choice(gt_resolution[-1]-1, args.N_samples, replace=False).flatten() + 1
    # slice_idx = slice_idx.tolist()
    if args.slice_sampling == 'fixed':
        assert (gt_resolution[-1] - 2) % (args.N_samples) == 0, print(args.N_samples)
        slice_idx = list(range(1, gt_resolution[-1] - 1, (gt_resolution[-1] - 2) // (args.N_samples)))
    elif args.slice_sampling == 'random':
        slice_idx = [i for i in range(0, gt_resolution[-1] - 2)]  ## slice_idx : 0 ~ 319
        # import pdb;pdb.set_trace()
        # slice_idx = slice_idx[-1:]
        slice_idx = np.random.choice(slice_idx, args.N_samples, replace=False).flatten() + 1
        slice_idx = slice_idx.tolist()
        slice_idx.sort()

    transformed_points_in_gt = torch.meshgrid(torch.linspace(0, gt_resolution[0] - 1, gt_resolution[0]),
                                              torch.linspace(0, gt_resolution[1] - 1, gt_resolution[1]),
                                              torch.linspace(0, gt_resolution[2] - 1, gt_resolution[2]))
    transformed_points_in_gt = torch.cat((transformed_points_in_gt[0].unsqueeze(-1),
                                          transformed_points_in_gt[1].unsqueeze(-1),
                                          transformed_points_in_gt[2].unsqueeze(-1)), dim=-1)
    transformed_points_in_gt = transformed_points_in_gt.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda()
    transformed_points = gt_renderer.gt2world_coordinate(transformed_points_in_gt)
    gt_resolution_torch = torch.from_numpy(gt_resolution).to(torch.float32).cuda()
    #transformed_points_prev = transformed_points.clone()
    transformed_points = transformed_points * gt_resolution_torch / (gt_resolution_torch - 2) * (
                trained_ct_res - 2) / trained_ct_res

    z_vals = torch.linspace(args.near, args.far, gt_resolution[2]).reshape(1, gt_resolution[2], 1).repeat(
        gt_resolution[0] * gt_resolution[1], 1, 1)
    z_vals = z_vals.reshape(gt_resolution[0], gt_resolution[1], gt_resolution[2], 1)
    z_vals = z_vals.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda()

    transformed_ray_directions_expanded = transformed_points[..., 1:2, :] - transformed_points[..., 0:1, :]
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.repeat(1, 1, 1, gt_resolution[2], 1)

    transformed_points = transformed_points[:, 1:-1, 1:-1, slice_idx, :]
    z_vals = z_vals[:, 1:-1, 1:-1, slice_idx, :]
    transformed_ray_directions_expanded = transformed_ray_directions_expanded[:, 1:-1, 1:-1, slice_idx, :]

    return transformed_points, transformed_ray_directions_expanded, z_vals, slice_idx


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None,
                      fill_mode=None, rgb_channel=3):
    """Performs NeRF volumetric rendering."""

    rgbs = rgb_sigma[..., :rgb_channel]
    sigmas = rgb_sigma[..., rgb_channel:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights


def get_initial_rays_trig(n, num_steps, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""

    device = 'cuda'
    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                              torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam

def perturb_points(points, z_vals, ray_directions, device, perturb=1):
    distance_between_points = z_vals[:, :, 1:2, :] - z_vals[:, :, 0:1, :]
    offset = (torch.rand(z_vals.shape, device=device) - 0.5) * distance_between_points
    z_vals = z_vals + offset * perturb

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals

def transform_sampled_points(points, z_vals, ray_directions, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5,
                             v_mean=math.pi * 0.5, mode='normal', pitch=None, yaw=None, perturb=1):
    """Samples a camera position and maps points in camera space to world space."""

    device = 'cuda'
    n, num_rays, num_steps, channels = points.shape

    points, z_vals = perturb_points(points, z_vals, ray_directions, device, perturb=perturb)

    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev,
                                                        vertical_stddev=v_stddev, horizontal_mean=h_mean,
                                                        vertical_mean=v_mean, device=device, mode=mode,
                                                        phi=pitch, theta=yaw)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1),
                                    device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)


    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)

    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal', phi=None, theta=None):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    # r= rand(0.5, 1.5 )

    if phi is None or theta is None:
        if mode == 'uniform':
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

        elif mode == 'normal' or mode == 'gaussian':
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

        elif mode == 'hybrid':
            if random.random() < 0.5:
                theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
                phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
            else:
                theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
                phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

        elif mode == 'truncated_gaussian':
            theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
            phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

        elif mode == 'spherical_uniform':
            theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
            v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
            v = ((torch.rand((n, 1), device=device) - .5) * 2 * v_stddev + v_mean)
            v = torch.clamp(v, 1e-5, 1 - 1e-5)
            phi = torch.arccos(1 - 2 * v)

        elif mode == 'spherical_bimodal':
            theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
            v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
            v = ((torch.rand((n, 1), device=device) - .5) * 2 * v_stddev + v_mean)
            v = torch.clamp(v, 1e-5, 1 - 1e-5)
            phi = torch.arccos(1 - 2 * v)

        else:
            # Just use the mean.
            theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
            phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r * torch.cos(phi)

    return output_points, phi, theta


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def create_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples
