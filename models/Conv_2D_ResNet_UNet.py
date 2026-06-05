import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from .embeddings import PositionalEmbedding
import math


class Affine2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)


class Conv2DCondResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        dropout=0.0,
        skip_scale=1.0,
        adaptive_scale=True,
        affine=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

        if adaptive_scale:
            self.map_cond = nn.Linear(emb_dim, out_channels * 2)
        else:
            self.map_cond = nn.Linear(emb_dim, out_channels)

        if affine:
            self.pre_norm = Affine2D(in_channels)
            self.post_norm = Affine2D(out_channels)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x, emb):
        residual = x

        x = self.pre_norm(x)
        x = self.conv1(x)
        x = F.silu(x)

        params = F.silu(self.map_cond(emb))

        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            scale = (scale + 1).unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * scale + shift
            x = F.silu(x)
        else:
            x = x + params.unsqueeze(-1).unsqueeze(-1)
            x = F.silu(x)

        x = self.conv2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.post_norm(x)

        return (x + self.res_conv(residual)) * self.skip_scale


class Conv2DResNetUNetCFG(nn.Module):
    def __init__(self, 
                 x_resolution,
                 y_resolution, 
                 cond_size,
                 model_channel       = 128,
                 channel_multiply    = [1,1,1,1],
                 dim_mult_emb        = 4,
                 num_blocks          = 2,
                 dropout             = 0.,
                 emb_type            = "sinusoidal",
                 dim_mult_time       = 1,
                 dim_mult_cond       = 1,
                 cond_drop_prob      = 0.0,
                 adaptive_scale      = True,
                 skip_scale          = 1.0,
                 affine              = False,
                 data_structure      = 'sdf',
                 **kwargs):

        super().__init__()


        self.cond_size = cond_size
        self.model_channel = model_channel
        self.cond_drop_prob = cond_drop_prob
        self.channel_multiply = channel_multiply
        self.num_blocks = num_blocks
        self.data_structure = data_structure

        if self.data_structure == 'sdf':
            self.in_channels = 1
            self.h = y_resolution
            self.w = x_resolution
        elif self.data_structure == '3D_coordinates':

            self.in_channels = 3
            self.h = 512 # number of points per profile
            self.w = 16 # number of profiles


        emb_dim  = model_channel * dim_mult_emb
        time_dim = model_channel * dim_mult_time
        cond_dim = model_channel * dim_mult_cond
        self.emb_dim = emb_dim

        self.null_emb = nn.Parameter(torch.randn(emb_dim))

        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim * cond_size, emb_dim)

        self.first_layer = nn.Conv2d(self.in_channels, model_channel, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = model_channel
        for mult in channel_multiply:
            out_ch = model_channel * mult
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(
                    Conv2DCondResNetBlock(in_ch, out_ch, emb_dim,
                                          dropout=dropout,
                                          skip_scale=skip_scale,
                                          adaptive_scale=adaptive_scale,
                                          affine=affine)
                )
                in_ch = out_ch
            self.down_blocks.append(blocks)

        bottleneck_ch = in_ch * 2
        self.bottleneck_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.bottleneck_blocks.append(
                Conv2DCondResNetBlock(in_ch, bottleneck_ch, emb_dim,
                                      dropout=dropout,
                                      skip_scale=skip_scale,
                                      adaptive_scale=adaptive_scale,
                                      affine=affine)
            )
            in_ch = bottleneck_ch

        self.up_blocks = nn.ModuleList()

        for mult in reversed(channel_multiply):
            skip_ch = model_channel * mult
            in_block_ch = in_ch + skip_ch
            out_block_ch = skip_ch

            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(
                    Conv2DCondResNetBlock(in_block_ch, out_block_ch, emb_dim,
                                          dropout=dropout,
                                          skip_scale=skip_scale,
                                          adaptive_scale=adaptive_scale,
                                          affine=affine)
                )
                in_block_ch = out_block_ch

            self.up_blocks.append(blocks)
            in_ch = out_block_ch

        self.final_layer = nn.Conv2d(in_ch, self.in_channels, kernel_size=3, padding=1)

    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.rand(shape, device=device) < prob

    def _forward(self, x, cond, time, context_mask=None, cond_drop_prob=None):
        B = x.shape[0]
        # print(x.shape, 'unet input')
        device = x.device

        time_emb = F.silu(self.map_time_layer(self.map_time(time)))
        cond_emb = F.silu(self.map_cond_layer(self.map_cond(cond).reshape(B, -1)))

        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        if cond_drop_prob > 0:
            keep = self.prob_mask_like((B,), 1-cond_drop_prob, device)
            cond_emb = torch.where(keep.unsqueeze(-1), cond_emb, self.null_emb.unsqueeze(0))

        emb = time_emb + cond_emb
        
        if self.data_structure == 'sdf':
            x = x.view(B, self.in_channels, self.h, self.w)
        
        elif self.data_structure == '3D_coordinates':

            x = x.view(B, self.w, self.h, self.in_channels)
            # print(x.shape, 'shape after view')
            x = x.permute(0, 3, 1, 2)
            # print(x.shape, 'shape after permute')
        
        

        x = self.first_layer(x)

        skips = []
        for blocks in self.down_blocks:
            for block in blocks:
                x = block(x, emb)
            skips.append(x)
            x = self.maxpool(x)

        for block in self.bottleneck_blocks:
            x = block(x, emb)

        for level, blocks in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            skip = skips[-(level+1)]
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, emb)

        x = self.final_layer(F.silu(x))
        
        
        # reshape the tensor back to the original form
        
        if self.data_structure == 'sdf':
            x = x.permute(0, 2, 3, 1).reshape(B, 1, self.w, self.h*1)
        elif self.data_structure == '3D_coordinates':
            x = x.permute(0, 2, 3, 1).reshape(B, 1, self.w, self.h*3)


        return x

    def forward(self, x, cond, time,
                context_mask=None,
                cond_drop_prob=0.,
                cond_scale=1.,
                rescaled_phi=0.,
                sampling=False):

        if not sampling:
            return self._forward(x, cond, time, context_mask, cond_drop_prob)

        logits = self._forward(x, cond, time, None, 0.)

        if cond_scale == 1:
            return logits

        null_logits = self._forward(x, cond, time, None, 1.)

        scaled = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0:
            return scaled

        std_fn = partial(torch.std, dim=tuple(range(1, scaled.ndim)), keepdim=True)
        rescaled = scaled * (std_fn(logits) / std_fn(scaled))

        return rescaled * rescaled_phi + scaled * (1 - rescaled_phi)
