import os
import glob
import argparse
import time
import nibabel as nib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from net.unet import UNet

from utils.utils import (
    affine_transform,
    rotate_matrix,
    grid_sample_2d,
    find_plane_from_affine,
    find_mutli_view_plane,
    connected_component_filter
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UK Biobank CMR label completion')
    parser.add_argument('--data_dir', default='YOUR_UKBB_DIR', type=str, help='dataset directory')
    parser.add_argument('--save_dir', default='./data/ukbb/', type=str, help='saving directory')
    parser.add_argument('--n_frame', default=50, type=int, help='number of time frames')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    n_frame = args.n_frame
    device = args.device

    
    # initialize grid for 2D sampling
    width = 64  # width for 2D images
    grid_2d = [torch.arange(0, 2*width), torch.arange(0, 2*width)]
    grid_2d = torch.stack(torch.meshgrid(grid_2d, indexing='ij')).permute(1,2,0)
    grid_2d = grid_2d - torch.tensor([width-0.5, width-0.5])
    grid_2d = grid_2d[None].to(device)

    
    # initialize 3D grid for grid sampling
    voxel_size = 2.0
    grid_size_3d = [96,96,128]
    # affine matrix
    affine_3d = np.eye(4) * voxel_size
    affine_3d[-1,-1] = 1
    affine_3d[:3,-1] = - np.array(grid_size_3d) + 1
    affine_3d_ = torch.Tensor(affine_3d).to(device)
    # grid for 3d sampling
    grid_3d = [torch.arange(0, s) for s in grid_size_3d]
    grid_3d = torch.stack(torch.meshgrid(
        grid_3d, indexing='ij')).permute(1,2,3,0).to(device).float()
    grid_3d = grid_3d[None] @ affine_3d_[:3,:3].T + affine_3d_[:3,-1]
    
    
    # load template 3D seg to compute center of mass
    seg_atlas_nii = nib.load('./template/whs_template_seg.nii.gz')
    seg_atlas = seg_atlas_nii.get_fdata()
    seg_atlas = torch.LongTensor(seg_atlas[None]).to(device)
    seg_atlas = F.one_hot(seg_atlas, num_classes=6).permute(0,4,1,2,3).float()
    seg_atlas = F.interpolate(
        seg_atlas, size=grid_size_3d, mode='trilinear', align_corners=True)
    seg_atlas = seg_atlas.argmax(1)[0].cpu().numpy().astype(np.uint8)
    center_atlas = np.stack(np.where(seg_atlas>0)).mean(-1)
    volume_atlas = (seg_atlas>0).sum()
    
    
    # directory of ukbb data
    subj_list = sorted(glob.glob(data_dir+'*'))
    n_data = len(subj_list)
    print(n_data)
    
    # random select data
    np.random.seed(12345)
    data_indices = np.random.permutation(n_data)

    # --------------------
    # load nn model
    # --------------------
    unet = UNet(
        dim_in=6,
        dim_hid=[32,64,128,256,256],
        dim_out=6,
        drop_rate=0.2).to(device)
    
    unet.load_state_dict(torch.load(
        './ckpts/YOUR_MODEL.pt', map_location=device))
    unet.eval();

    n_proc = 0
    n_total = 1200
    
    for idx in tqdm(data_indices[:]):
        subj_id = subj_list[idx].split('/')[-1]

        # ------------------------
        # load UK biobank data
        # ------------------------
        # check if there are missing data
        no_missing_data = os.path.exists(data_dir+subj_id+'/seg_la_2ch.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/seg4_la_4ch.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/seg_sa.nii.gz')
        if not no_missing_data:
            continue
    
        seg_2ch_nii = nib.load(data_dir+subj_id+'/seg_la_2ch.nii.gz')
        seg_4ch_nii = nib.load(data_dir+subj_id+'/seg4_la_4ch.nii.gz')
        seg_sax_nii = nib.load(data_dir+subj_id+'/seg_sa.nii.gz')
    
        # (L,W,H,T) --> (H,T,L,W)
        seg_2ch = seg_2ch_nii.get_fdata().transpose(2,3,0,1)
        seg_4ch = seg_4ch_nii.get_fdata().transpose(2,3,0,1)
        seg_sax = seg_sax_nii.get_fdata().transpose(2,3,0,1)
    
        n_slice = seg_sax.shape[0]
    
        affine_2ch = seg_2ch_nii.affine
        affine_4ch = seg_4ch_nii.affine
        affine_sax = seg_sax_nii.affine
    
        seg_2ch_la = (seg_2ch==1).astype(np.float32)
        seg_4ch_lv = (seg_4ch==1).astype(np.float32)
        seg_4ch_my = (seg_4ch==2).astype(np.float32)
        seg_4ch_rv = (seg_4ch==3).astype(np.float32)
        seg_4ch_la = (seg_4ch==4).astype(np.float32)
        seg_4ch_ra = (seg_4ch==5).astype(np.float32)
        seg_sax_lv = (seg_sax==1).astype(np.float32)
        seg_sax_my = (seg_sax==2).astype(np.float32)
        seg_sax_rv = (seg_sax==3).astype(np.float32)
    
        # check if there are missing segmentation labels at each frame
        no_missing_seg = seg_2ch_la.sum(axis=(0,2,3)).all() & \
                         seg_4ch_lv.sum(axis=(0,2,3)).all() & \
                         seg_4ch_my.sum(axis=(0,2,3)).all() & \
                         seg_4ch_rv.sum(axis=(0,2,3)).all() & \
                         seg_4ch_la.sum(axis=(0,2,3)).all() & \
                         seg_4ch_ra.sum(axis=(0,2,3)).all() & \
                         seg_sax_lv.sum(axis=(0,2,3)).all() & \
                         seg_sax_my.sum(axis=(0,2,3)).all() & \
                         seg_sax_rv.sum(axis=(0,2,3)).all()
        if not no_missing_seg:
            continue
    
        if not os.path.exists(save_dir+subj_id):
            os.makedirs(save_dir+subj_id)
    
        # ------------------------
        # correct the image center in 3D space
        # ------------------------
        # compute three viewing planes to obtain the intersection points of 3 planes
        n_2ch, p_2ch = find_plane_from_affine(affine_2ch)
        n_4ch, p_4ch = find_plane_from_affine(affine_4ch)
        n_sax, p_sax = find_plane_from_affine(affine_sax)
        isect_pt = np.linalg.inv(np.vstack([n_2ch, n_4ch, n_sax])) @ \
                   np.array([p_2ch, p_4ch, p_sax])
    
        # move the intersection point along the intersection line
        # such that it is located in the centroid of the heart
        d_2ch_4ch = 25  # direction towards the LV
        n_2ch_4ch = np.cross(n_2ch, n_4ch)
        isect_pt = isect_pt + d_2ch_4ch * n_2ch_4ch
        d_4ch_sax = 25  # direciton towards the right heart
        n_4ch_sax = np.cross(n_4ch, n_sax)
        isect_pt = isect_pt + d_4ch_sax * n_4ch_sax
        d_2ch_sax = 15  # direciton along the 2ch viewing plane
        n_2ch_sax = np.cross(n_2ch, n_sax)
        isect_pt = isect_pt + d_2ch_sax * n_2ch_sax
    
        # the center points are no longer in the original view planes (2D spaces)
        # it should be projected from the real-world space to the 2D space
        center_2ch = affine_transform(isect_pt, np.linalg.inv(affine_2ch))
        center_4ch = affine_transform(isect_pt, np.linalg.inv(affine_4ch))
        center_sax = affine_transform(isect_pt, np.linalg.inv(affine_sax))
        center_2ch[2] = 0
        center_4ch[2] = 0
        center_sax[2] = 0
    
        # ------------------------
        # correct the image center and clip 2D images
        # ------------------------
        # we manipulate the inverse affine matrix such that center (x0,y0)
        # is mapped to (width-1,width-1), which is the center of the clipped image
        affine_2ch_proc = affine_2ch.copy()
        inv_affine_2ch_proc = np.linalg.inv(affine_2ch_proc)
        inv_affine_2ch_proc[:2,-1] -= center_2ch[:2] - width + 0.5
        affine_2ch_proc = np.linalg.inv(inv_affine_2ch_proc)
    
        affine_4ch_proc = affine_4ch.copy()
        inv_affine_4ch_proc = np.linalg.inv(affine_4ch_proc)
        inv_affine_4ch_proc[:2,-1] -= center_4ch[:2] - width + 0.5
        affine_4ch_proc = np.linalg.inv(inv_affine_4ch_proc)
    
        affine_sax_proc = affine_sax.copy()
        inv_affine_sax_proc = np.linalg.inv(affine_sax_proc)
        inv_affine_sax_proc[:2,-1] -= center_sax[:2] - width + 0.5
        affine_sax_proc = np.linalg.inv(inv_affine_sax_proc)
    
        # resample the image such that the projected center is on the image center
        # the resampeld image has the size of (128,128)
        center_2ch = torch.Tensor(center_2ch[:2]).to(device)
        center_4ch = torch.Tensor(center_4ch[:2]).to(device)
        center_sax = torch.Tensor(center_sax[:2]).to(device)
    
        seg_2ch_tensor = torch.Tensor(seg_2ch).to(device)
        seg_4ch_tensor = torch.Tensor(seg_4ch).to(device)
        seg_sax_tensor = torch.Tensor(seg_sax).to(device)
    
        grid_2ch = grid_2d + center_2ch
        grid_4ch = grid_2d + center_4ch
        grid_sax = (grid_2d + center_sax).repeat(n_slice,1,1,1)
    
        seg_2ch_proc = grid_sample_2d(seg_2ch_tensor, grid_2ch, mode='nearest')
        seg_4ch_proc = grid_sample_2d(seg_4ch_tensor, grid_4ch, mode='nearest')
        seg_sax_proc = grid_sample_2d(seg_sax_tensor, grid_sax, mode='nearest')
    
        # move the center of cardiac shapes to the intersection points of 3 planes
        affine_2ch_proc[:3,-1] -= isect_pt
        affine_4ch_proc[:3,-1] -= isect_pt
        affine_sax_proc[:3,-1] -= isect_pt
    
        # ------------------------
        # rotate the image in 3D space
        # ------------------------
        # we rotate the heart in the world space to make sure that 
        # the cardiac shapes have similar orientations
    
        # i) we first rotate the sax viewing plane to normal direction (0,0,1)
        n_z = np.array([0,0,1])  # target direction
        rot_z = np.zeros([4,4])
        rot_z[:3,:3] = rotate_matrix(n_sax, n_z)
        rot_z[-1,-1] = 1
    
        # ii) the 4ch viewing plane is almost perpendicular to the sax plane
        # we then rotate the 4ch viewing plane to normal direction (0,1,0) 
        n_y = np.array([0,1,0])  # target direction
        rot_y = np.zeros([4,4])
        rot_y[:3,:3] = rotate_matrix(affine_transform(n_4ch, rot_z), n_y)
        rot_y[-1,-1] = 1
    
        # apply rotation matrix to original affine matrix
        affine_2ch_proc = rot_y @ rot_z @ affine_2ch_proc
        affine_4ch_proc = rot_y @ rot_z @ affine_4ch_proc
        affine_sax_proc = rot_y @ rot_z @ affine_sax_proc
    
        # compute inverse affine matrix for reconstructing 3D from 2D
        inv_affine_2ch = np.linalg.inv(affine_2ch_proc)
        inv_affine_4ch = np.linalg.inv(affine_4ch_proc)
        inv_affine_sax = np.linalg.inv(affine_sax_proc)
    
        inv_affine_2ch = torch.Tensor(inv_affine_2ch).to(device)[None]
        inv_affine_4ch = torch.Tensor(inv_affine_4ch).to(device)[None]
        inv_affine_sax = torch.Tensor(inv_affine_sax).to(device)[None]
    
        # 2D+t segmentations
        seg_2d_2ch = seg_2ch_proc.long().permute(1,2,3,0).repeat(1,1,1,2)
        seg_2d_4ch = seg_4ch_proc.long().permute(1,2,3,0).repeat(1,1,1,2)
        seg_2d_sax = seg_sax_proc.long().permute(1,2,3,0)
    
        # one-hot segmentations
        seg_2d_2ch = F.one_hot(seg_2d_2ch, num_classes=2).float().permute(0,4,1,2,3)
        seg_2d_4ch = F.one_hot(seg_2d_4ch, num_classes=6).float().permute(0,4,1,2,3)
        seg_2d_sax = F.one_hot(seg_2d_sax, num_classes=4).float().permute(0,4,1,2,3)
    
        # --------------------
        # 2d --> 3d seg 
        # --------------------
        # find multi-view cutting plane
        mask_2ch, mask_4ch, mask_sax = find_mutli_view_plane(
            grid_3d[0].cpu().numpy(), affine_2ch_proc, affine_4ch_proc, affine_sax_proc, n_slice, voxel_size)
        mask_2ch = torch.LongTensor(mask_2ch[None]).to(device)
        mask_4ch = torch.LongTensor(mask_4ch[None]).to(device)
        mask_sax = torch.LongTensor(mask_sax[None]).to(device)
    
        # inverse affine transform the 3D grid
        grid_2ch = grid_3d @ inv_affine_2ch[:,:3,:3].transpose(2,1) + inv_affine_2ch[:,:3,-1]
        grid_4ch = grid_3d @ inv_affine_4ch[:,:3,:3].transpose(2,1) + inv_affine_4ch[:,:3,-1]
        grid_sax = grid_3d @ inv_affine_sax[:,:3,:3].transpose(2,1) + inv_affine_sax[:,:3,-1]
    
        grid_size_2ch = torch.Tensor([seg_2d_2ch.shape[2:]]).to(device)
        grid_size_4ch = torch.Tensor([seg_2d_4ch.shape[2:]]).to(device)
        grid_size_sax = torch.Tensor([seg_2d_sax.shape[2:]]).to(device)
    
        grid_2ch = (2 * grid_2ch / (grid_size_2ch - 1) - 1).flip(-1)
        grid_4ch = (2 * grid_4ch / (grid_size_4ch - 1) - 1).flip(-1)
        grid_sax = (2 * grid_sax / (grid_size_sax - 1) - 1).flip(-1)
    
        # perform grid sampling from 2d to 3d
        seg_3d_2ch = F.grid_sample(
            seg_2d_2ch.reshape(1,-1,*seg_2d_2ch.shape[2:]),
            grid_2ch, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_2ch.shape[1:-1]).argmax(1).long() * 4
    
        seg_3d_4ch = F.grid_sample(
            seg_2d_4ch.reshape(1,-1,*seg_2d_4ch.shape[2:]),
            grid_4ch, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_4ch.shape[1:-1]).argmax(1).long()
    
        seg_3d_sax = F.grid_sample(
            seg_2d_sax.reshape(1,-1,*seg_2d_sax.shape[2:]),
            grid_sax, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_sax.shape[1:-1]).argmax(1).long()
    
        seg_3d_2ch = seg_3d_2ch * mask_2ch
        seg_3d_4ch = seg_3d_4ch * mask_4ch
        seg_3d_sax = seg_3d_sax * mask_sax
    
        # merge multi-view segmentations
        seg_3d = seg_3d_sax.clone()
        seg_3d = seg_3d_2ch + seg_3d * (1-(seg_3d_2ch > 0).long())
        seg_3d = seg_3d_4ch + seg_3d * (1-(seg_3d_4ch > 0).long())
    
        # --------------------
        # 3d shape completion
        # --------------------
        seg_4d = np.zeros([n_frame, *grid_size_3d])
        for t in range(n_frame):
            # shape completion for each time frame
            seg_3d_t = F.one_hot(seg_3d[t:t+1], num_classes=6).permute(0,4,1,2,3).float()
    
            with torch.no_grad():
                seg_4d_t = unet(seg_3d_t)
    
            seg_3d_t = seg_3d_t.argmax(1)[0].cpu().numpy()
            seg_4d_t = seg_4d_t.argmax(1)[0].cpu().numpy()
            seg_4d[t] = seg_4d_t
    
        # --------------------
        # scale and shift
        # --------------------
        # compute center of mass and volume
        center_4d = np.stack(np.where(seg_4d[0]>0)).mean(-1)
        volume_4d = (seg_4d[0]>0).sum()
        scale_factor = (volume_4d / volume_atlas)**(1/3)
    
        # for inverse affine transformation
        inv_affine_align = np.eye(4) / scale_factor
        inv_affine_align[-1,-1] = 1
        trans_4d = affine_3d[:3,:3] @ center_4d + affine_3d[:3,-1]
        trans_atlas = affine_3d[:3,:3] @ center_atlas + affine_3d[:3,-1]
        inv_affine_align[:3,-1] = - trans_4d / scale_factor + trans_atlas
        affine_align = np.linalg.inv(inv_affine_align)
        affine_align_ = torch.Tensor(affine_align).to(device)
    
        # grid sampling for scale and shift
        grid_align = grid_3d @ affine_align_[:3,:3].T + affine_align_[:3,-1]
        grid_size_align = torch.Tensor(grid_size_3d).to(device)
        grid_align = (grid_align / (grid_size_align - 1)).flip(-1)
    
        seg_4d_align = torch.LongTensor(seg_4d).to(device)
        seg_4d_align = F.one_hot(
            seg_4d_align, num_classes=6).permute(0,4,1,2,3).float() #.to(device)
        seg_4d_align = F.grid_sample(
            seg_4d_align.reshape(1,-1,*seg_4d_align.shape[2:]),
            grid_align, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_align.shape[1:-1]).argmax(1).long()
        seg_4d_align = seg_4d_align.cpu().numpy().astype(np.uint8)

        # --------------------
        # clean noise voxels
        # --------------------
        seg_4d_cc = np.zeros_like(seg_4d_align)
        for t in range(n_frame):
            seg_lv_t_cc = connected_component_filter(seg_4d_align[t]==1)
            seg_my_t_cc = connected_component_filter(seg_4d_align[t]==2)
            seg_rv_t_cc = connected_component_filter(seg_4d_align[t]==3)
            seg_la_t_cc = connected_component_filter(seg_4d_align[t]==4)
            seg_ra_t_cc = connected_component_filter(seg_4d_align[t]==5)

            seg_4d_cc[t][np.where(seg_lv_t_cc)] = 1
            seg_4d_cc[t][np.where(seg_my_t_cc)] = 2
            seg_4d_cc[t][np.where(seg_rv_t_cc)] = 3
            seg_4d_cc[t][np.where(seg_la_t_cc)] = 4
            seg_4d_cc[t][np.where(seg_ra_t_cc)] = 5
        
        # save affine matrix
        np.save(save_dir+subj_id+'/affine.npy',
            [affine_2ch_proc.astype(np.float32),
             affine_4ch_proc.astype(np.float32),
             affine_sax_proc.astype(np.float32)])
        
        # save nifti file
        affine_4d = affine_align @ affine_3d
        seg_4d_nii = nib.Nifti1Image(
            seg_4d_cc.transpose(1,2,3,0).astype(np.uint8), affine=affine_4d)
        nib.save(seg_4d_nii, save_dir+subj_id+'/seg_4d.nii.gz')
        torch.cuda.empty_cache()
        
        n_proc += 1
        if n_proc >= n_total:
            break
