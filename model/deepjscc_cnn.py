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
import numpy as np


class Identity(nn.Module):
    def forward(self, x):
        return x


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class JSCC_Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, n_blocks=2, C_channel=16, C_extend=1, n_downsampling=2, norm_layer=nn.BatchNorm2d):

        assert(n_downsampling >= 0)
        super(JSCC_Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        activation = nn.ReLU(True)

        # Downscale network
        model = [nn.ReflectionPad2d((7 - 1) // 2),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]
        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult, max_ngf), min(ngf * mult * 2, max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]
        self.net = nn.Sequential(*model)

        mult = 2 ** n_downsampling
        self.res_list = nn.ModuleList([])

        for i in range(n_blocks):
            self.res_list.append(ResnetBlock(min(ngf * mult, max_ngf), padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias))
            self.res_list.append(modulation(min(ngf * mult, max_ngf), C_extend=C_extend))

        self.projection = nn.Conv2d(min(ngf * mult, max_ngf), C_channel, kernel_size=3, padding=1, stride=1)
        self.n_blocks = n_blocks

    def forward(self, input, SNR):

        z = self.net(input)

        for i in range(self.n_blocks):
            z = self.res_list[i * 2 + 1](self.res_list[i * 2](z), SNR)
        latent = self.projection(z)

        return latent


class JSCC_Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, C_extend=1, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert (n_blocks >= 0)
        assert(n_downsampling >= 0)

        super(JSCC_Decoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)

        self.mask_conv = nn.Conv2d(C_channel, ngf_dim, kernel_size=3, padding=1, stride=1, bias=use_bias)

        self.res_list = nn.ModuleList([])

        for i in range(n_blocks):
            self.res_list.append(ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias))
            self.res_list.append(modulation(ngf_dim, C_extend=C_extend))

        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult // 2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((5 - 1) // 2), nn.Conv2d(ngf, output_nc, kernel_size=5, padding=0)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)
        self.n_blocks = n_blocks

    def forward(self, input, SNR):
        z = self.mask_conv(input)
        for i in range(self.n_blocks):
            z = self.res_list[i * 2 + 1](self.res_list[i * 2](z), SNR)

        return self.model(z)


class modulation(nn.Module):
    def __init__(self, C_channel, C_extend=1):

        super(modulation, self).__init__()

        activation = nn.ReLU(True)

        # Policy network
        model_multi = [nn.Linear(C_channel + C_extend, C_channel), activation,
                       nn.Linear(C_channel, C_channel), nn.Sigmoid()]

        model_add = [nn.Linear(C_channel + C_extend, C_channel), activation,
                     nn.Linear(C_channel, C_channel)]

        self.model_multi = nn.Sequential(*model_multi)
        self.model_add = nn.Sequential(*model_add)

    def forward(self, z, CSI):

        # Policy/gate network
        N, C, W, H = z.shape

        z_mean = torch.mean(z, (-2, -1))
        z_cat = torch.cat((z_mean, CSI), -1)

        factor = self.model_multi(z_cat).view(N, C, 1, 1)
        addition = self.model_add(z_cat).view(N, C, 1, 1)

        return z * factor + addition


def define_Encoder(input_nc, ngf, max_ngf, C_channel, C_extend, n_blocks, n_downsample, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = JSCC_Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, C_extend=C_extend, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_Decoder(output_nc, ngf, max_ngf, n_downsample, C_channel, C_extend, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = JSCC_Decoder(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, C_extend=C_extend, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    return init_net(net, init_type, init_gain, gpu_ids)


class DeepJSCC(pl.LightningModule, ImageLoggerMixin):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
    """

    def __init__(
        self,
        input_nc=3,
        ngf=64,
        max_ngf_E=256,
        max_ngf_D=384,
        n_downsample=2,
        norm='batch',
        init_type='normal',
        init_gain=0.02,
        C_channel=16,
        output_nc=3,
        n_blocks_E=2,
        n_blocks_D=4,
        SNR_low=0,
        SNR_high=20,
        loss_type='MSE',
        channel='AWGN',
        min_lr: float=None,
        learning_rate: float=None,
        weight_decay: float=None,
        max_iteration: int=None
    ) -> "DeepJSCC":
        super(DeepJSCC, self).__init__()

        if channel == 'AWGN':
            C_extend = 1
        elif channel == 'Rayleigh':
            C_extend = 3

        self.E = define_Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf_E, C_channel=C_channel, C_extend=C_extend,
                           n_blocks=n_blocks_E, n_downsample=n_downsample, norm=norm, init_type=init_type,
                           init_gain=init_gain)


        self.G = define_Decoder(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf_D,
                              n_downsample=n_downsample, C_channel=C_channel, C_extend=C_extend,
                              n_blocks=n_blocks_D, norm=norm, init_type=init_type,
                              init_gain=init_gain)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lpips_metric = LPIPS(net="alex")
        self.SNR_low = SNR_low
        self.SNR_high = SNR_high
        self.max_iteration = max_iteration
        self.loss_type = loss_type
        self.channel = channel
        self.min_lr = min_lr

        if self.loss_type == 'MSSSIM':
            self.MSSSIM_loss = MS_SSIM(data_range=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        SNR = torch.rand(x.shape[0], 1) * (self.SNR_high - self.SNR_low) + self.SNR_low
        SNR = SNR.to(x.device)
        
        with torch.no_grad():
            if self.channel == 'AWGN':
                h = torch.ones_like(SNR)
                h = h.view(x.shape[0], 1, 1, 1)
                CSI = SNR
            elif self.channel == 'Rayleigh':
                h_real = torch.randn_like(SNR) / np.sqrt(2)
                h_imag = torch.randn_like(SNR) / np.sqrt(2)
                h = (h_real + h_imag * 1j)
                h = h.view(x.shape[0], 1, 1, 1)
                CSI = torch.cat([SNR, h_real, h_imag], dim=1)

        latent = self.E(x, CSI)

        N, C, H, W = latent.shape
        latent_complex = latent[:, :C // 2, :, :] + 1j * latent[:, C // 2:, :, :]

        # Normalization
        latent_pwr = (latent_complex.abs()**2).mean((-3, -2, -1), keepdim=True)
        latent_tx = latent_complex / torch.sqrt(latent_pwr)

        # Pass through the complex channel
        with torch.no_grad():
            sigma = 10**(-SNR / 20)
            noise = (sigma.view(N, 1, 1, 1) / np.sqrt(2)) * torch.randn_like(latent)
            noise = noise[:, :C // 2, :, :] + 1j * noise[:, C // 2:, :, :]

        latent_rx = latent_tx * h + noise
        latent_eua = latent_rx / h
        latent_eua = torch.cat((latent_eua.real, latent_eua.imag), dim=1)

        x_hat = self.G(latent_eua, CSI)

        return x_hat, SNR, h

    def get_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:

        if self.loss_type == 'MSE':
            return F.mse_loss(input=pred, target=label, reduction="mean")
        elif self.loss_type == 'MSSSIM':
            return 1 - self.MSSSIM_loss(pred, label)

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

        pred, _, _ = self(hq)
        loss = self.get_loss(pred, hq)
        self.log("train_loss", loss, on_step=True)
        return loss

    def on_validation_start(self) -> None:
        self.lpips_metric.to(self.device)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        hq = batch['jpg']
        pred, _, _ = self(hq)
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
        params = list(self.E.parameters()) + list(self.G.parameters())
        optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iteration, eta_min=self.min_lr)
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
        pred, _, _ = self(hq)
        return dict(pred=pred, hq=hq)
