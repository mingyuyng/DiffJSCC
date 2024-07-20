
import math
from typing import Any, Dict, Set

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import functools
from torch.nn import init
import torch.utils.checkpoint as checkpoint
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from utils.metrics import calculate_psnr_pt, LPIPS
from .mixins import ImageLoggerMixin

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from .swin_decoder import *
from .swin_encoder import *
from random import choice
import numpy as np

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


class DeepJSCC_SWIN(pl.LightningModule, ImageLoggerMixin):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
    """

    def __init__(
        self,
        img_size_w=512,
        img_size_h=512,
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        embed_dims_d=[320, 256, 192, 128],
        depths=[2, 2, 6, 2],
        depths_d=[2, 6, 2, 2],
        num_heads=[4, 6, 8, 10],
        num_heads_d=[10, 8, 6, 4],
        bottleneck_dim=16,
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        SNR_low=0,
        SNR_high=20,
        loss_type='MSE',
        channel='AWGN',
        learning_rate: float=None,
        weight_decay: float=None,
        max_iteration: int=None
    ) -> "DeepJSCC_SWIN":
        super(DeepJSCC_SWIN, self).__init__()

        self.encoder = WITT_Encoder(img_size=(img_size_h, img_size_w), patch_size=patch_size, in_chans=in_chans,
                                    embed_dims=embed_dims, depths=depths, num_heads=num_heads, C=bottleneck_dim,
                                    window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    norm_layer=norm_layer, patch_norm=patch_norm,
                                    bottleneck_dim=bottleneck_dim)

        self.decoder = WITT_Decoder(img_size=(img_size_h, img_size_w), embed_dims=embed_dims_d, depths=depths_d,
                                    num_heads=num_heads_d, C=bottleneck_dim, window_size=window_size, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer, ape=True, patch_norm=patch_norm,
                                    bottleneck_dim=bottleneck_dim)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lpips_metric = LPIPS(net="alex")
        self.SNR_low = SNR_low
        self.SNR_high = SNR_high
        self.max_iteration = max_iteration
        self.loss_type = loss_type
        self.channel = channel
        self.H = self.W = 0

        if self.loss_type == 'MSSSIM':
            self.MSSSIM_loss = MS_SSIM(data_range=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, H, W = x.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** 4), W // (2 ** 4))
            self.H = H
            self.W = W

        SNR = torch.rand(x.shape[0], 1) * (self.SNR_high-self.SNR_low) + self.SNR_low
        SNR = SNR.to(x.device)
        
        latent = self.encoder(x, SNR)

        # Pass through the complex channel
        N, L, C = latent.shape
        latent_complex = latent[:, :, :C//2] + 1j* latent[:, :, C//2:]

        # Normalization
        latent_pwr = torch.sqrt((latent_complex.abs()**2).mean((-2, -1), keepdim=True))
        latent_tx = latent_complex / latent_pwr

        # Pass through the complex channel
        with torch.no_grad():
            sigma = 10**(-SNR / 20) / np.sqrt(2)
            noise = sigma.view(N, 1, 1) * torch.randn_like(latent)
            noise = noise[:, :, :C//2] + 1j* noise[:, :, C//2:]

            # Generate the fading factor h if we are using Rayleigh Slow Fading Channel
            if self.channel == 'AWGN':
                h = torch.ones_like(latent_tx)
            elif self.channel == 'Rayleigh':
                h = (torch.randn_like(SNR) + torch.randn_like(SNR)*1j) / np.sqrt(2)
                h = h.view(N, 1, 1)

        latent_rx = latent_tx * h  + noise
        latent_rx = torch.cat((latent_rx.real, latent_rx.imag), dim=-1)
        
        x_hat = self.decoder(latent_rx, SNR)

        x_hat = torch.sigmoid(x_hat)

        return x_hat, SNR 

    
    def get_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model predictions and labels.
        
        Args:
            pred (torch.Tensor): Batch model predictions.
            label (torch.Tensor): Batch labels.
        
        Returns:
            loss (torch.Tensor): The loss tensor.
        """

        if self.loss_type == 'MSE':
            return F.mse_loss(input=pred, target=label, reduction="mean")
        elif self.loss_type == 'MSSSIM':
            return 1-self.MSSSIM_loss(pred, label)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (Dict[str, torch.Tensor]): A dict contains LQ and HQ (NHWC, RGB, 
                LQ range in [0, 1] and HQ range in [-1, 1]).
            batch_idx (int): Index of this batch.
        
        Returns:
            outputs (torch.Tensor): The loss tensor.
        """     

        hq = batch['jpg']

        pred, _ = self(hq)
        loss = self.get_loss(pred, hq)
        self.log("train_loss", loss, on_step=True)
        return loss

    def on_validation_start(self) -> None:
        self.lpips_metric.to(self.device)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        hq = batch['jpg']
        pred, _ = self(hq)

        # requiremtns for lpips model inputs:
        # https://github.com/richzhang/PerceptualSimilarity
        lpips = self.lpips_metric(pred, hq, normalize=True).mean()
        self.log("val_lpips", lpips)

        pnsr = calculate_psnr_pt(pred, hq, crop_border=0).mean()
        self.log("val_pnsr", pnsr)

        loss = self.get_loss(pred, hq)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> optim.AdamW:
        """
        Configure optimizer for this model.

        Returns:
            optimizer (optim.AdamW): The optimizer for this model.
        """
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iteration, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    @torch.no_grad()
    def log_images(self, batch: Any) -> Dict[str, torch.Tensor]:
        hq = batch['jpg']
        pred, _ = self(hq)
        return dict(lq=hq, pred=pred, hq=hq)
