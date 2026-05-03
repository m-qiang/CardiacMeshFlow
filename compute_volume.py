import numpy as np
import glob
import argparse
import nibabel as nib
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ffdnet import FFDNet
from utils.utils import affine_transform
from utils.mesh import mesh_volume
from utils.ffd import bspline_basis, bspline_ffd
import pyvista as pv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mesh-based volume estimation')
    parser.add_argument('--data_dir', default='./data/ukbb/', type=str, help='dataset directory')
    parser.add_argument('--n_frame', default=50, type=int, help='number of time frames')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    args = parser.parse_args()
    data_dir = args.data_dir
    n_frame = args.n_frame
    device = args.device

    
    # ------ load templates ------ 
    # load template 3D seg
    seg_atlas_nii = nib.load('./template/whs_template_seg.nii.gz')
    seg_atlas = seg_atlas_nii.get_fdata()
    affine_atlas = seg_atlas_nii.affine
    voxel_size = 2
    seg_atlas = torch.LongTensor(seg_atlas[None]).to(device)
    seg_atlas = F.one_hot(seg_atlas, num_classes=6).permute(0,4,1,2,3).float()
    seg_atlas = F.interpolate(
        seg_atlas, size=[96,96,128], mode='trilinear', align_corners=True
    ).argmax(1)[None].float()
    
    # load template mesh
    mesh_lv_in = pv.read('./template/whs_template_mesh_lv.vtk')
    mesh_my_in = pv.read('./template/whs_template_mesh_my.vtk')
    mesh_rv_in = pv.read('./template/whs_template_mesh_rv.vtk')
    mesh_la_in = pv.read('./template/whs_template_mesh_la.vtk')
    mesh_ra_in = pv.read('./template/whs_template_mesh_ra.vtk')
    
    vert_lv_in, face_lv_in_vtk = mesh_lv_in.points, mesh_lv_in.faces.reshape(-1,4)
    vert_my_in, face_my_in_vtk = mesh_my_in.points, mesh_my_in.faces.reshape(-1,4)
    vert_rv_in, face_rv_in_vtk = mesh_rv_in.points, mesh_rv_in.faces.reshape(-1,4)
    vert_la_in, face_la_in_vtk = mesh_la_in.points, mesh_la_in.faces.reshape(-1,4)
    vert_ra_in, face_ra_in_vtk = mesh_ra_in.points, mesh_ra_in.faces.reshape(-1,4)
    
    face_lv_in = face_lv_in_vtk[:,1:]
    face_my_in = face_my_in_vtk[:,1:]
    face_rv_in = face_rv_in_vtk[:,1:]
    face_la_in = face_la_in_vtk[:,1:]
    face_ra_in = face_ra_in_vtk[:,1:]
    
    vert_lv_in = affine_transform(vert_lv_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_my_in = affine_transform(vert_my_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_rv_in = affine_transform(vert_rv_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_la_in = affine_transform(vert_la_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_ra_in = affine_transform(vert_ra_in, np.linalg.inv(affine_atlas)) / voxel_size
    
    vert_lv_in = torch.Tensor(vert_lv_in[None]).to(device)
    vert_my_in = torch.Tensor(vert_my_in[None]).to(device)
    vert_rv_in = torch.Tensor(vert_rv_in[None]).to(device)
    vert_la_in = torch.Tensor(vert_la_in[None]).to(device)
    vert_ra_in = torch.Tensor(vert_ra_in[None]).to(device)
    
    face_lv_in = torch.LongTensor(face_lv_in[None].copy()).to(device)
    face_my_in = torch.LongTensor(face_my_in[None].copy()).to(device)
    face_rv_in = torch.LongTensor(face_rv_in[None].copy()).to(device)
    face_la_in = torch.LongTensor(face_la_in[None].copy()).to(device)
    face_ra_in = torch.LongTensor(face_ra_in[None].copy()).to(device)
    
    # number of vertices
    n_vert_lv_in = vert_lv_in.shape[1]
    n_vert_my_in = vert_my_in.shape[1]
    n_vert_rv_in = vert_rv_in.shape[1]
    n_vert_la_in = vert_la_in.shape[1]
    n_vert_ra_in = vert_ra_in.shape[1]
    
    # concatenate the vertices of all chambers
    vert_all_in = torch.cat(
        [vert_lv_in, vert_my_in, vert_rv_in, vert_la_in, vert_ra_in], dim=1)
    
    # index to split the tensor of all vertices
    idx_vert_lv_in = n_vert_lv_in
    idx_vert_my_in = idx_vert_lv_in + n_vert_my_in
    idx_vert_rv_in = idx_vert_my_in + n_vert_rv_in
    idx_vert_la_in = idx_vert_rv_in + n_vert_la_in
    idx_vert_all_in = [idx_vert_lv_in, idx_vert_my_in, idx_vert_rv_in, idx_vert_la_in]
    

    # ------ load neural network ------ 
    ffd_net = FFDNet(
        n_frame=1, dim_hid=128, drop_rate=0.0).to(device)
    ffd_net.load_state_dict(
        torch.load('./ckpts/YOUR_MODEL.pt', map_location=device))
    ffd_net.eval();


    # ------ extract subject information ------ 
    subj_list = sorted(glob.glob(data_dir+'/*/seg_4d.nii.gz'))
    
    for subj_dir in tqdm(subj_list[:]):
        # ------ load 3D+t segmentation ------
        subj_id = subj_dir.split('/')[-2]
        seg_4d_nii = nib.load(subj_dir)
        affine = seg_4d_nii.affine
        scale = affine[0,0]
        seg_4d = seg_4d_nii.get_fdata().transpose(3,0,1,2)
    
        volume_lv = []
        volume_my = []
        volume_rv = []
        volume_la = []
        volume_ra = []
    
        for t in range(n_frame):
            seg_4d_t = torch.Tensor(seg_4d[t])[None,None].to(device)
            seg_in_t = torch.cat([seg_atlas, seg_4d_t], dim=1) / 5.
    
            # ------ mesh reconstruction ------
            with torch.no_grad():
                ffd_pred_t = ffd_net(seg_in_t)
                
            vert_all_pred_t = vert_all_in.clone()
            for s in range(3):  # multi-scale
                vert_all_pred_t = vert_all_pred_t + bspline_ffd(
                    vert_all_pred_t / (2**(4-s)), ffd_pred_t[s])
                
            vert_lv_pred_t, vert_my_pred_t, vert_rv_pred_t, \
            vert_la_pred_t, vert_ra_pred_t = torch.tensor_split(
                    vert_all_pred_t * scale, idx_vert_all_in, dim=1)

            # ------ volume calculation ------
            volume_lv_t = mesh_volume(vert_lv_pred_t, face_lv_in).item() / 1000
            volume_my_t = mesh_volume(vert_my_pred_t, face_my_in).item() / 1000
            volume_rv_t = mesh_volume(vert_rv_pred_t, face_rv_in).item() / 1000
            volume_la_t = mesh_volume(vert_la_pred_t, face_la_in).item() / 1000
            volume_ra_t = mesh_volume(vert_ra_pred_t, face_ra_in).item() / 1000
            
            volume_lv.append(volume_lv_t)
            volume_my.append(volume_my_t - volume_lv_t)
            volume_rv.append(volume_rv_t)
            volume_la.append(volume_la_t)
            volume_ra.append(volume_ra_t)
    
        volume_lv = np.stack(volume_lv)
        volume_my = np.stack(volume_my)
        volume_rv = np.stack(volume_rv)
        volume_la = np.stack(volume_la)
        volume_ra = np.stack(volume_ra)
    
        volume_all = np.vstack([volume_lv, volume_my, volume_rv, volume_la, volume_ra])
        np.save(data_dir+subj_id+'/volume.npy', volume_all)
        