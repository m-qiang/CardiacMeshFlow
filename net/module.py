import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    3D convolutional block

    Args:
    - dim_in: dim of input channels
    - dim_out: dim of output channels
    - K: kernel size

    Input:
    - x: input features
    - t_emb: time embedding
    """
    def __init__(self, dim_in, dim_out, K=3, stride=1, drop_rate=0.0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=dim_in, out_channels=dim_out,
            kernel_size=K, stride=stride, padding=K//2)
        self.norm = nn.InstanceNorm3d(dim_in, affine=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        # norm -> activation -> dropout -> conv -> norm
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

    
class AdaIN(nn.Module):
    """
    3D adapative instance normalization block

    Args:
    - dim_feat: dim of features 
    - dim_emb: dim of embeddings
    Input:
    - x: input features
    - emb: conditional embedding
    """
    def __init__(self, dim_feat, dim_emb):
        super(AdaIN, self).__init__()

        self.norm = nn.InstanceNorm3d(dim_feat, affine=False)
        self.affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_emb, 2*dim_feat)
        )
        
    def forward(self, x, emb):
        scale, shift = self.affine(emb)[...,None,None,None].chunk(2, dim=1)
        x = self.norm(x) * (1 + scale) + shift 
        return x
        
        
        
class AdaLN(nn.Module):
    """
    adaptive layer normalization block

    Args:
    - dim_feat: dim of features
    - dim_emb: dim of conditional embedding
    Input:
    - x: input features (B, dim_channel, dim_feat)
    - emb: conditional embedding  (B,C,D)
    """
    def __init__(self, dim_feat, dim_emb):
        super(AdaLN, self).__init__()

        self.norm = nn.LayerNorm(dim_feat, elementwise_affine=False)
        self.affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_emb, 2*dim_feat)
        )
        
    def forward(self, x, emb):
        scale, shift = self.affine(emb)[:,None].chunk(2, dim=-1)
        y = self.norm(x) * (1 + scale) + shift 
        return y
    
    

class cConvBlock(nn.Module):
    """
    3D conditional convolutional block

    Args:
    - dim_in: dim of input channels
    - dim_out: dim of output channels
    - dim_emb: dim of conditional embeddings
    - K: kernel size

    Input:
    - x: input features
    - t_emb: temporal embedding
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_emb,
        K=3,
        stride=1,
        drop_rate=0.0
    ):
        super(cConvBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels=dim_in, out_channels=dim_out,
            kernel_size=K, stride=stride, padding=K//2
        )
        self.norm = AdaIN(dim_in, dim_emb)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, emb):
        x = self.norm(x, emb)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x