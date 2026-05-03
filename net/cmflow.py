import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.module import ConvBlock, cConvBlock


class NoiseSampler(nn.Module):
    """
    noise sampler

    Args:
    - dim_latent: dim of latent vectors
    - n_data: num of data
    """
    def __init__(
        self,
        dim_latent=256,
        n_data=600
    ):
        super(NoiseSampler, self).__init__()
        self.epsilon = nn.Embedding(
            n_data, dim_latent, max_norm=None)
        nn.init.normal_(self.epsilon.weight.data, 0.0, 0.1)
        self.distribution()
        
    def forward(self, idx):
        eps = self.epsilon(idx)  # (B, dim)
        return eps
        
    def distribution(self):
        epsilon = self.epsilon.weight.data.detach()
        mean = epsilon.mean(0)
        cov = epsilon.T.cov()
        self.gaussian = torch.distributions.MultivariateNormal(mean, cov)
        
    def sample(self):
        eps = self.gaussian.sample()[None]  # (B, dim)
        return eps


def gaussian_encoding(mu=0, sigma=1.0, n_class=50, eps=1e-10):
    """periodic Gaussian kernel encoding"""
    x = (np.arange(n_class) - mu + n_class//2) % n_class - n_class//2
    p = (np.exp(-x**2/(2*sigma**2 + eps)) / (np.sqrt(2*np.pi) * sigma + eps))
    return p / p.sum()



class MultiFusion(nn.Module):
    """
    multiscale fusion network to fuse input noise,
    temporal encoding, and phenotypes

    Args:
    - dim_latent: dim of noise
    - dim_cond: dim of temporal encoding and condition
    - n_data: num of data
    
    Inputs:
    - eps: noise  (B, dim_latent)
    - cond: temporal encoding + other codition (B, dim_cond)
    """
    def __init__(
        self,
        dim_latent=256,
        dim_cond=50,
        dim_vol=[6,6,8],
        K=3
    ):
        super(MultiFusion, self).__init__()
                
        self.fuse = nn.Sequential(
            nn.Linear(dim_latent+dim_cond, dim_latent),
            nn.SiLU()
        )
        self.scale_up = nn.Sequential(
            nn.Linear(1,dim_latent),
            nn.SiLU(),
            nn.Linear(dim_latent,np.prod(dim_vol))
        )
        self.deconv3 = ConvBlock(dim_in=dim_latent, dim_out=dim_latent, K=K, stride=1, drop_rate=0.0)
        self.deconv2 = ConvBlock(dim_in=dim_latent, dim_out=dim_latent//2, K=K, stride=1, drop_rate=0.0)
        self.deconv1 = ConvBlock(dim_in=dim_latent//2, dim_out=dim_latent//4, K=K, stride=1, drop_rate=0.0)

        self.out1 = ConvBlock(dim_in=dim_latent, dim_out=3, K=K, stride=1, drop_rate=0.0)
        self.out2 = ConvBlock(dim_in=dim_latent//2, dim_out=3, K=K, stride=1, drop_rate=0.0)
        self.out3 = ConvBlock(dim_in=dim_latent//4, dim_out=3, K=K, stride=1, drop_rate=0.0)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dim_vol = dim_vol
        
    def forward(self, eps, c):
        z = torch.cat([eps, c], dim=1)
        z = self.fuse(z).unsqueeze(-1)
        z = self.scale_up(z).reshape(
            z.shape[0], z.shape[1], *self.dim_vol)
        
        z = self.deconv3(z)
        z1 = self.out1(z)

        z = self.up(z)
        z = self.deconv2(z)
        z2 = self.out2(z)

        z = self.up(z)
        z = self.deconv1(z)
        z3 = self.out3(z)
        return z1, z2 ,z3



class MultiDiUNet(nn.Module):
    """
    Multiscale 3D diffusion U-Net for flow matching

    Args:
    - dim_hid: dim of hidden channels
    - dim_hid: dim of conditions
    - dim_emb: dim of conditional embeddings
    - K: kernel size

    Inputs:
    - x: input volume (B,C_in,L,W,H)

    Returns:
    - x: output volume (B,C_out,L,W,H)
    """
    def __init__(
        self,
        dim_in=3,
        dim_hid=256,
        dim_cond=1,
        dim_emb=256,
        K=3,
        drop_rate=0.0
    ):
        super(MultiDiUNet, self).__init__()

        # compute temporal embedding
        self.embedding = nn.Sequential(
            nn.Linear(dim_cond, dim_emb//2),
            nn.SiLU(),
            nn.Linear(dim_emb//2, dim_emb),
            nn.SiLU(),
            nn.Linear(dim_emb, dim_emb)
        )
        
        self.conv0 = nn.Conv3d(in_channels=dim_in, out_channels=dim_hid//4, kernel_size=K, stride=1, padding=K//2)
        self.conv1 = cConvBlock(dim_in=dim_hid//4, dim_out=dim_hid//4, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.conv2 = cConvBlock(dim_in=dim_hid//4, dim_out=dim_hid//2, dim_emb=dim_emb, K=K, stride=2, drop_rate=drop_rate)
        self.conv3 = cConvBlock(dim_in=dim_in+dim_hid//2, dim_out=dim_hid//2, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.conv4 = cConvBlock(dim_in=dim_hid//2, dim_out=dim_hid, dim_emb=dim_emb, K=K, stride=2, drop_rate=drop_rate)
        self.conv5 = cConvBlock(dim_in=dim_in+dim_hid, dim_out=dim_hid, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.conv6 = cConvBlock(dim_in=dim_hid, dim_out=dim_hid, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)

        self.deconv5 = cConvBlock(dim_in=dim_hid*2, dim_out=dim_hid, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.deconv4 = cConvBlock(dim_in=dim_hid, dim_out=dim_hid//2, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.deconv3 = cConvBlock(dim_in=dim_hid, dim_out=dim_hid//2, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.deconv2 = cConvBlock(dim_in=dim_hid//2, dim_out=dim_hid//4, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.deconv1 = cConvBlock(dim_in=dim_hid//2, dim_out=dim_hid//4, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)

        self.out1 = cConvBlock(dim_in=dim_hid, dim_out=dim_in, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.out2 = cConvBlock(dim_in=dim_hid//2, dim_out=dim_in, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        self.out3 = cConvBlock(dim_in=dim_hid//4, dim_out=dim_in, dim_emb=dim_emb, K=K, stride=1, drop_rate=drop_rate)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, z1, z2, z3, c):
        # conditional embedding
        emb = self.embedding(c)
        
        x1 = self.conv0(z3)
        x1 = self.conv1(x1, emb)

        
        x2 = self.conv2(x1, emb)
        x2 = torch.cat([z2, x2], dim=1)
        x2 = self.conv3(x2, emb)

        x3 = self.conv4(x2, emb)
        x3 = torch.cat([z1, x3], dim=1)
        x3 = self.conv5(x3, emb)

        x = self.conv6(x3, emb)
        
        x = torch.cat([x, x3], dim=1)
        x = self.deconv5(x, emb)
        dz1 = self.out1(x, emb)
        
        x = self.deconv4(x, emb)
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.deconv3(x, emb)
        dz2 = self.out2(x, emb)

        x = self.deconv2(x, emb)
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.deconv1(x, emb)
        dz3 = self.out3(x, emb)
        
        return dz1, dz2, dz3
        