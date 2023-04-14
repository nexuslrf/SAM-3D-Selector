import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torch_scatter import scatter_min
import cv2

def project_pcd(pnt_w, K, c2w):
    cam_pos = c2w[:3, 3]
    pnt_cam = ((pnt_w - cam_pos)[..., None, :] * c2w[:3, :3].transpose()).sum(-1)
    uv_cam = (pnt_cam / pnt_cam[..., 2:]) @ K.T
    return uv_cam, pnt_cam, pnt_cam[..., 2:] # w , h

def unproject_pcd(pnt_cam, c2w):
    pnt_w = c2w[:3, :3] @ pnt_cam.T + c2w[:3, 3, np.newaxis]
    return pnt_w.transpose()

def get_depth_map(uv, depth, h, w, bg_depth=1e10, scale=2):
    _h, _w = int(h / scale), int(w / scale)
    uv = (uv / scale).round().astype(np.int32)[..., :2]
    uv = uv.clip(0, np.array([_w, _h]) - 1)
    depth_ten = torch.from_numpy(depth).float().reshape(-1)
    uv_int_flat_ten = torch.from_numpy(uv[:, 0] * _h + uv[:, 1]).long()
    depth_map_ten = torch.ones((_w, _h, 1), dtype=torch.float32).reshape(-1) * 1e10
    # min scatter
    _, index = scatter_min(depth_ten, uv_int_flat_ten, dim=0, out=depth_map_ten)
    depth_map = depth_map_ten.reshape(_w, _h).transpose(0, 1).numpy()
    # upscale
    depth_map = cv2.resize(depth_map, (h, w), interpolation=cv2.INTER_NEAREST)
    return depth_map, index

def covisibility_mask(depth_1, depth_2, c2w_1, c2w_2, K, bg_depth=1e10, thres=0.05, pool_size=1):
    w, h = depth_1.shape[1], depth_1.shape[0]
    uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
    uv = np.concatenate([uv, np.ones_like(uv[..., :1])], -1)
    pnt_cam_1 = (uv @ np.linalg.inv(K).T) * depth_1[..., np.newaxis]
    pnt_w_1 = unproject_pcd(pnt_cam_1.reshape(-1, 3), c2w_1)
    uv_1t2, pnt_cam_1t2, depth_1t2 = project_pcd(pnt_w_1, K, c2w_2)
    bg_mask_1 = depth_1.reshape(-1) >= bg_depth
    bg_mask_2 = depth_2.reshape(-1) >= bg_depth

    # method 1: grid sample
    # center_offset = np.array([w, h]) / 2
    # uv_1t2_rescale = (uv_1t2[:,:2] - center_offset) / center_offset
    # depth_2_ten = torch.from_numpy(depth_2).float()[None, None, ...]
    # uv_1t2_rescale_ten = torch.from_numpy(uv_1t2_rescale).float()[None, None, ...].clamp(-1, 1)
    # depth_2_sample = F.grid_sample(
    #     depth_2_ten, uv_1t2_rescale_ten, 
    #     padding_mode="border", align_corners=True).reshape_as(depth_1).numpy()

    # method 2: round to nearest
    uv_1t2 = uv_1t2[:, :2].round().astype(np.int32).clip(0, np.array([w, h]) - 1)
    depth_2_sample = depth_2[uv_1t2[:, 1], uv_1t2[:, 0]]
    depth_1t2 = depth_1t2.reshape(-1)
    covis_1 = (depth_1t2 > 0) & (np.abs(depth_1t2 - depth_2_sample) < thres) & (~bg_mask_1)
    covis_1 = covis_1.reshape(depth_1.shape)
    covis_2 = np.zeros_like(covis_1)
    covis_2[uv_1t2[:, 1], uv_1t2[:, 0]] = covis_1.reshape(-1)
    # max pool on covis_2
    covis_2 = F.max_pool2d(
            torch.from_numpy(covis_2.astype(np.float32))[None, None, ...], 
                2*pool_size+1, 1, pool_size).numpy().reshape(covis_2.shape)
    return covis_1, covis_2

    # if sel_mode == 1:
        # if pnt_mask is None:
        #     pnt_mask = pnt_frame_mask.copy()
        # else:
        #     pnt_mask = np.logical_or(pnt_mask, pnt_frame_mask)
            # for ref_pnt_mask, ref_c2w, ref_depth_map, uv_ref in pnt_frame_buffer:
            #     covis_mask, covis_mask_ref = covisibility_mask(depth_map, ref_depth_map, c2w, ref_c2w, data.K, thres=0.2, pool_size=2)
            #     pnt_covis_mask = mask_pcd_2d(uv_cam, covis_mask)[..., None] * pnt_frame_mask
            #     pnt_covis_mask_ref = mask_pcd_2d(uv_ref, covis_mask_ref)[..., None] * ref_pnt_mask
            #     pnt_mask = np.logical_and(pnt_mask, ~pnt_covis_mask, ~pnt_covis_mask_ref)
            #     pnt_mask = np.logical_or(pnt_mask, pnt_covis_mask * pnt_covis_mask_ref)
        # if pnt_mask is None:
            # color[pnt_mask[..., 0]] = pnt_sel_color_global / 225. 


def mask_pcd_2d(uv, mask, thresh=0.5, depth=None, pnt_depth=None, depth_thresh=0.1):
    h, w = mask.shape
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten, 
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    sample_mask = sample_mask > thresh
    if depth is not None:
        sample_depth = F.grid_sample(
            torch.from_numpy(depth).float()[None, None, ...], uv_rescale_ten,
            padding_mode="border", align_corners=True).reshape(-1).numpy()
        sample_mask = sample_mask & (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)

    return sample_mask

