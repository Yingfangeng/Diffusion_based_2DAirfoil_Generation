import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from .embeddings import PositionalEmbedding


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class ConvBlock1D(nn.Module):

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0, affine = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        if affine:
            self.pre_norm = Affine(in_ch)
            self.post_norm = Affine(out_ch)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()



    def forward(self, x, emb):
        x = self.pre_norm(x)
        x = self.conv(x)
        emb_out = self.emb_proj(emb).unsqueeze(-1)
        x = x + emb_out
        x = self.act(x)
        x = self.dropout(x)
        x = self.post_norm(x)
        return x



class Conv1DUNetCFG(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim, 
                 cond_size,
                 model_channel       = 128, 
                 channel_multiply        = [1,1,1,1],
                 dim_mult_emb    = 4,
                 num_blocks      = 4, 
                 dropout         = 0.,
                 emb_type        = "sinusoidal",
                 dim_mult_time   = 1,
                 dim_mult_cond   = 1,
                 cond_drop_prob  = 0.0,
                 adaptive_scale  = True,
                 skip_scale      = 1.0,
                 affine          = False):

        super().__init__()

        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.cond_size      = cond_size
        self.model_dim      = model_channel
        self.cond_drop_prob = cond_drop_prob
        self.dim_mult       = channel_multiply
        self.num_blocks     = num_blocks

        # 600 numbers → 300 points × 2 channels
        self.in_channels = 2
        self.num_points  = in_dim // self.in_channels

        # Embedding dims
        emb_dim  = model_channel * dim_mult_emb
        time_dim = model_channel * dim_mult_time
        cond_dim = model_channel * dim_mult_cond
        self.emb_dim = emb_dim

        # Null embedding for classifier-free guidance
        self.null_emb = nn.Parameter(torch.randn(emb_dim))

        # Time and condition embeddings
        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim * cond_size, emb_dim)

        # First layer, lift 300x2 to 300x128
        self.first_layer = nn.Conv1d(self.in_channels, self.model_dim, kernel_size=3, padding=1)

        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        self.maxpool     = nn.MaxPool1d(kernel_size=2, stride=2)

        in_ch = self.model_dim
        for mult in channel_multiply:
            out_ch = self.model_dim * mult
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(ConvBlock1D(in_ch, out_ch, emb_dim, dropout=dropout, affine = affine))
                in_ch = out_ch
            self.down_blocks.append(blocks)

        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList()
        bottleneck_ch = in_ch * 2
        for _ in range(num_blocks):
            self.bottleneck_blocks.append(ConvBlock1D(in_ch, bottleneck_ch, emb_dim, dropout=dropout, affine = affine))
            in_ch = bottleneck_ch

        # Decoder blocks
        self.up_blocks = nn.ModuleList()

        for mult in reversed(channel_multiply):
            skip_ch = self.model_dim * mult

            in_block_ch  = in_ch + skip_ch
            out_block_ch = skip_ch

            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(ConvBlock1D(in_block_ch, out_block_ch, emb_dim, dropout=dropout, affine = affine))
                in_block_ch = out_block_ch

            self.up_blocks.append(blocks)
            in_ch = out_block_ch

        # Final layer, convert back to 300x2
        self.final_layer = nn.Conv1d(in_ch, self.in_channels, kernel_size=3, padding=1)


    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


    def forward(self, x, cond, time,
                context_mask=None,
                cond_drop_prob=0.,
                cond_scale = 1.,
                rescaled_phi = 0.,
                sampling=False):

        if sampling:
            logits = self._forward(x, cond, time, context_mask=None, cond_drop_prob=0.)
            if cond_scale == 1:
                return logits

            null_logits = self._forward(x, cond, time, context_mask=None, cond_drop_prob=1.)
            scaled_logits = null_logits + (logits - null_logits) * cond_scale

            if rescaled_phi == 0.:
                return scaled_logits

            std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
            rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

            return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)
        else:
            return self._forward(x, cond, time, context_mask, cond_drop_prob)


    def _forward(self, x, cond, time, context_mask=None, cond_drop_prob=None):
        
        B, D = x.shape
        assert D == self.in_dim

        device = x.device

        # embeddings
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        time_emb = F.silu(self.map_time_layer(time_emb))
        cond_emb = F.silu(self.map_cond_layer(cond_emb.reshape(cond_emb.shape[0], -1)))

        # classifier-free guidance dropout on the condition embedding
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        if cond_drop_prob > 0:
            keep_mask = self.prob_mask_like((B,), 1 - cond_drop_prob, device=device)
            null_cond_emb = repeat(self.null_emb, 'd -> b d', b=B)
            cond_emb = torch.where(rearrange(keep_mask, 'b -> b 1'), cond_emb, null_cond_emb)

        # combined embedding used in all blocks
        emb = time_emb + cond_emb

        # Covnert 600x1 to 300x2
        x = x.view(B, self.num_points, self.in_channels)
        x = x.permute(0, 2, 1)

        # First layer, 300x2 to 300x128
        x = self.first_layer(x)

        # Encoder blocks
        skips = []
        for level_blocks in self.down_blocks:
            for block in level_blocks:
                x = block(x, emb)
            skips.append(x)
            x = self.maxpool(x)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, emb)

        # Decoder
        for level_idx, level_blocks in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            skip = skips[-(level_idx + 1)] # reverse the lsit

            # check the dimension before skip concat.
            if x.shape[-1] != skip.shape[-1]:
                print(x.shape[-1], skip.shape[-1],'upsampling dimension mismatched')
            
            
            # concat skip connections
            x = torch.cat([x, skip], dim=1)
            for block in level_blocks:
                x = block(x, emb)

        # Final layer, convert back to 300x2
        x = self.final_layer(F.silu(x))

        # Convert back to 600x1
        x = x.permute(0, 2, 1).reshape(B, -1)
        
        return x