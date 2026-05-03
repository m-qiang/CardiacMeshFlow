import numpy as np
from tqdm import tqdm
import glob
import logging
import argparse
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from net.ffdnet import FFDNet
from net.cmflow import (
    gaussian_encoding,
    MultiFusion,
    MultiDiUNet
)

from utils.data import FFDDataset
from utils.utils import affine_transform, estimate_phenotype
from utils.mesh import mesh_volume
from utils.ffd import bspline_ffd
from utils.metrics import vFID

import pyvista as pv


def train_loop(args):
    
    data_dir = args.data_dir
    tag = args.tag
    device = args.device
    lr = args.lr
    n_epoch = args.n_epoch
    n_frame = args.n_frame  # number of frames
    n_train = args.n_train  # number of training data
    n_valid = args.n_valid  # number of validation data
    drop_rate = args.drop_rate
    sigma = args.sigma  # standard deviation of Gaussian encoding
    density = 1.05
    
    # start training logging
    logging.basicConfig(
        filename='./ckpts/log_cmflow_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')

    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    
    n_data = len(glob.glob(data_dir+'*'))
    print(n_data)
    
    np.random.seed(12345)
    data_indices = np.random.permutation(n_data)
    train_indices = data_indices[:n_train]
    valid_indices = data_indices[n_train:n_train+n_valid]
    
    trainset = FFDDataset(
        data_dir=data_dir, data_indices=train_indices[:], data_type='train',
        n_frame=n_frame, device=device)
    validset = FFDDataset(
        data_dir=data_dir, data_indices=valid_indices[:], data_type='valid',
        n_frame=n_frame, device=device)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    
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
    
    # faces for saving vtk files
    face_lv_pred = face_lv_in[0].cpu().numpy()
    face_my_pred = face_my_in[0].cpu().numpy()
    face_rv_pred = face_rv_in[0].cpu().numpy()
    face_la_pred = face_la_in[0].cpu().numpy()
    face_ra_pred = face_ra_in[0].cpu().numpy()
    
    face_my_pred += face_lv_pred.max() + 1
    face_rv_pred += face_my_pred.max() + 1
    face_la_pred += face_rv_pred.max() + 1
    face_ra_pred += face_la_pred.max() + 1
    
    face_all_pred = np.vstack([
        face_lv_pred,
        face_my_pred,
        face_rv_pred,
        face_la_pred,
        face_ra_pred])
    
    face_all_pred_vtk = np.hstack([
        3*np.ones([face_all_pred.shape[0], 1], dtype=int), face_all_pred])


    # ------ initialize model ------ 
    logging.info("initalize model ...")
    
    cardiac_mesh_flow = MultiDiUNet(
        dim_in=3,
        dim_hid=256,
        dim_cond=1+9,
        dim_emb=256,
        drop_rate=0.0).to(device)

    fusion = MultiFusion(
        dim_latent=256,
        dim_cond=50,
        dim_vol=[6,6,8]).to(device)
    
    optimizer = optim.Adam(
        list(cardiac_mesh_flow.parameters()) + \
        list(fusion.parameters()), lr=1e-4)
    
    # beta distribution
    beta = torch.distributions.beta.Beta(0.1, 2.0)
    
    # ------ training loop ------ 
    for epoch in tqdm(range(n_epoch+1)):
        
        # --------------------
        #  training
        # --------------------
        train_loss = []
        cardiac_mesh_flow.train()
        fusion.train()
    
        for idx, data in enumerate(trainloader):
            subj_idx, phenotype, ffd1_gt, ffd2_gt, ffd3_gt = data
            ffd1_gt = ffd1_gt.float().to(device)
            ffd2_gt = ffd2_gt.float().to(device)
            ffd3_gt = ffd3_gt.float().to(device)
            subj_idx = subj_idx.long().to(device)
    
            phenotype = phenotype.float().to(device)
            # normalise phenotypes to input condition
            cond_pheno = phenotype.clone()
            cond_pheno[:,0] /= density  # mass to volume
            # EDV ~ LVM + LVEDV + RVEDV + LAMINV + RAMINV
            volume_ed = phenotype[:,[0,1,3,6,8]].sum()
            cond_pheno = cond_pheno / volume_ed * 5

            for j in np.random.permutation(np.arange(n_frame)):
                optimizer.zero_grad()
    
                # time for cmr frame
                tau = gaussian_encoding(mu=j, sigma=sigma, n_class=n_frame)
                tau = torch.Tensor(tau[None]).float().to(device)
                
                # time for flow matching
                t = beta.sample([1,1]).float().to(device)  # beta sampling

                # random noise
                eps = torch.randn(1,256).float().to(device)
                
                cond = torch.cat([t, cond_pheno], dim=-1)
                
                # initial and target distribution
                z1_0, z2_0, z3_0 = fusion(eps, tau)
                z1_1 = ffd1_gt[:,j]
                z2_1 = ffd2_gt[:,j]
                z3_1 = ffd3_gt[:,j]
    
                # flow matching
                z1_t = t * z1_1 + (1-t) * z1_0
                z2_t = t * z2_1 + (1-t) * z2_0
                z3_t = t * z3_1 + (1-t) * z3_0
                
                dz1_t, dz2_t, dz3_t = cardiac_mesh_flow(z1_t, z2_t, z3_t, cond)
    
                loss = nn.MSELoss(reduction='mean')(dz1_t, z1_1 - z1_0) + \
                       nn.MSELoss(reduction='mean')(dz2_t, z2_1 - z2_0) + \
                       nn.MSELoss(reduction='mean')(dz3_t, z3_1 - z3_0)
    
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
    
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(train_loss)))
    
        # --------------------
        #  validation
        # --------------------
        if epoch % 5 == 0:
            cardiac_mesh_flow.eval()
            fusion.eval()
            
            # save model checkpoints
            torch.save(
                cardiac_mesh_flow.state_dict(),
                './ckpts/model_cmflow_'+tag+'_'+str(epoch)+'epochs.pt')
            torch.save(
                fusion.state_dict(),
                './ckpts/model_fusion_'+tag+'_'+str(epoch)+'epochs.pt')
        
            rmsd_pheno_list = []
            n_step = 1
            h = 1 / n_step
            
            for idx, data in enumerate(validloader):
                phenotype_gt, _ = data
                phenotype_gt = phenotype_gt.float().to(device)
                
                # use validation phenotypes as input
                cond_pheno = phenotype_gt.clone()
                cond_pheno[:,0] /= density  # mass to volume
                # EDV ~ LVM + LVEDV + RVEDV + LAMINV + RAMINV
                volume_ed_gt = phenotype_gt[:,[0,1,3,6,8]].sum()
                cond_pheno = cond_pheno / volume_ed_gt * 5
                
                # fixed noise and phenotypes for each sample
                eps = torch.randn(1,256).float().to(device)
                
                # ------ generate 3D+t meshes based on phenotypes ------
                volume_lv = []
                volume_my = []
                volume_rv = []
                volume_la = []
                volume_ra = []
                
                for j in range(n_frame):
                    tau = gaussian_encoding(mu=j, sigma=sigma, n_class=n_frame)
                    tau = torch.Tensor(tau[None]).float().to(device)
                    
                    with torch.no_grad():
                        z1_t, z2_t, z3_t = fusion(eps, tau)
                        t = torch.zeros([1,1]).float().to(device)
                        for n in range(n_step):
                            cond = torch.cat([t, cond_pheno], dim=-1)
                            dz1_t, dz2_t, dz3_t = cardiac_mesh_flow(z1_t, z2_t, z3_t, cond)
                            z1_t += h * dz1_t
                            z2_t += h * dz2_t
                            z3_t += h * dz3_t
                            t += h
                    ffd1_pred_j = F.pad(z1_t, pad=[1]*6)[0].permute(1,2,3,0)
                    ffd2_pred_j = F.pad(z2_t, pad=[1]*6)[0].permute(1,2,3,0)
                    ffd3_pred_j = F.pad(z3_t, pad=[1]*6)[0].permute(1,2,3,0)
                    ffd_pred_j = [ffd1_pred_j, ffd2_pred_j, ffd3_pred_j]

                    # ------ apply ffd ------
                    vert_all_pred_j = vert_all_in.clone()
                    for s in range(3):  # multi-scale
                        vert_all_pred_j = vert_all_pred_j + bspline_ffd(
                            vert_all_pred_j / (2**(4-s)), ffd_pred_j[s])
        
                    vert_lv_pred_j, vert_my_pred_j, vert_rv_pred_j, \
                    vert_la_pred_j, vert_ra_pred_j = torch.tensor_split(
                        vert_all_pred_j, idx_vert_all_in, dim=1)
    
                    # ------ volume calculation ------
                    volume_lv_j = mesh_volume(vert_lv_pred_j * voxel_size, face_lv_in).item() / 1000
                    volume_my_j = mesh_volume(vert_my_pred_j * voxel_size, face_my_in).item() / 1000
                    volume_rv_j = mesh_volume(vert_rv_pred_j * voxel_size, face_rv_in).item() / 1000
                    volume_la_j = mesh_volume(vert_la_pred_j * voxel_size, face_la_in).item() / 1000
                    volume_ra_j = mesh_volume(vert_ra_pred_j * voxel_size, face_ra_in).item() / 1000
                    
                    volume_lv.append(volume_lv_j)
                    volume_my.append(volume_my_j - volume_lv_j)
                    volume_rv.append(volume_rv_j)
                    volume_la.append(volume_la_j)
                    volume_ra.append(volume_ra_j)
    
                volume_lv = np.stack(volume_lv)
                volume_my = np.stack(volume_my)
                volume_rv = np.stack(volume_rv)
                volume_la = np.stack(volume_la)
                volume_ra = np.stack(volume_ra)
    
                phenotype_pred = estimate_phenotype(volume_lv, volume_my, volume_rv, volume_la, volume_ra)
                phenotype_pred = torch.Tensor(phenotype_pred[None]).float().to(device)
                phenotype_pred[:,0] /= density  # mass to volume
                volume_ed_pred = phenotype_pred[:,[0,1,3,6,8]].sum()  # EDV
                phenotype_pred = phenotype_pred / volume_ed_pred * volume_ed_gt  # rescale volume
                phenotype_pred[:,0] *= density  # volume to mass
    
                rmsd_pheno = ((phenotype_pred - phenotype_gt) ** 2).mean().sqrt()
                rmsd_pheno_list.append(rmsd_pheno.item())
    
            logging.info("------------------------------")
            logging.info("epoch:{}, valid rmsd:{}".format(epoch, np.mean(rmsd_pheno_list)))
            logging.info("------------------------------")
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cardiac Mesh Flow")
    parser.add_argument('--data_dir', default='./dataset/', type=str, help="data directory")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    
    parser.add_argument('--n_epoch', default=200, type=int, help="number of training epochs")
    parser.add_argument('--n_frame', default=50, type=int, help="number of frames")
    parser.add_argument('--n_train', default=600, type=int, help="number of training data")
    parser.add_argument('--n_valid', default=100, type=int, help="number of validation data")
    
    parser.add_argument('--drop_rate', default=0.0, type=float, help="dropout rate")
    parser.add_argument('--sigma', default=1.0, type=float, help="standard deviation of gaussian encoding")

    args = parser.parse_args()
    
    train_loop(args)