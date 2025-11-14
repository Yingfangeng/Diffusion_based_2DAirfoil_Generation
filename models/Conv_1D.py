import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from .embeddings import PositionalEmbedding
from functools import partial

class Conv1DCondResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        cond_emb_dim,
        dropout=0.0,
        skip_scale=1.0,
        adaptive_scale=True,
        affine=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = time_emb_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        # Conv1D layers instead of Linear
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual projection (1x1 conv) if channels change
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

        self.map_cond = nn.Linear(time_emb_dim + cond_emb_dim, out_channels * (2 if adaptive_scale else 1))

        self.pre_norm = nn.Identity()
        self.post_norm = nn.Identity()


    def forward(self, x, time_emb=None, cond_emb=None):

        orig = x  # for residual connection
        # Combine time and condition embeddings
        emb = torch.cat((time_emb, cond_emb), dim=-1) 
        params = F.silu(self.map_cond(emb).to(x.dtype))

        x = self.pre_norm(x)
        x = self.conv1(x)
        x = F.silu(x)

        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            scale = (scale + 1).unsqueeze(-1)
            shift = shift.unsqueeze(-1)
            x = F.silu(shift + x * scale)
        else:
            params = params.unsqueeze(-1)
            x = F.silu(x + params)

        x = self.conv2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.post_norm(x)

        x = x + self.res_conv(orig) # add together the skipped signal
        x = x * self.skip_scale

        return x

class Conv1DCFGResNet(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim, 
                 cond_size,
                 model_dim       = 128,      
                 dim_mult        = [1,1,1,1],
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

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_size = cond_size
        self.model_dim = model_dim
        self.cond_drop_prob = cond_drop_prob

        self.num_points = in_dim // 2       
        self.in_channels = 2                

        # Embedding dimensions (same as your original)
        emb_dim  = model_dim * dim_mult_emb
        time_dim = model_dim * dim_mult_time
        cond_dim = model_dim * dim_mult_cond
        block_kwargs = dict(dropout = dropout,
                            skip_scale=skip_scale,
                            adaptive_scale=adaptive_scale,
                            affine=affine)

        # Null embedding for classifier-free guidance
        self.null_emb = nn.Parameter(torch.randn(emb_dim))

        # Time & condition embeddings (same style as before)
        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim * cond_size, emb_dim)

        # First Conv1D layer: (B, 2, L) -> (B, model_dim, L)
        self.first_layer = nn.Conv1d(self.in_channels, model_dim, kernel_size=3, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        cout = model_dim
        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(
                    Conv1DCondResNetBlock(
                        cin, cout, emb_dim, emb_dim, **block_kwargs
                    )
                )

        # Final Conv1D layer back to 2 channels, kernel_size=1 to keep length
        self.final_layer = nn.Conv1d(cout, self.in_channels, kernel_size=1)


    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


    def forward(self, x, cond, time,
                context_mask=None,
                cond_drop_prob=0,
                cond_scale = 1.,
                rescaled_phi = 0.,
                sampling=False):
        if sampling:
            logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=0.)
            if cond_scale == 1:
                return logits

            null_logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=1.)
            scaled_logits = null_logits + (logits - null_logits) * cond_scale

            if rescaled_phi == 0.:
                return scaled_logits

            std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
            rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

            return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)
        else:
            # training path
            return self._forward(x, cond, time, context_mask, cond_drop_prob)


    def _forward(self, x, cond, time, context_mask=None, cond_drop_prob=None):

        batch_size = x.shape[0]

        time_emb = self.map_time(time)                      
        cond_emb = self.map_cond(cond)                      
        time_emb = F.silu(self.map_time_layer(time_emb))    
        cond_emb = F.silu(self.map_cond_layer(
            cond_emb.reshape(cond_emb.shape[0], -1)         
        ))                                                  

    
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        if cond_drop_prob > 0:
            keep_mask = self.prob_mask_like((batch_size,), 1 - cond_drop_prob, device = x.device)
            null_cond_emb = repeat(self.null_emb, 'd -> b d', b = batch_size)

            cond_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                cond_emb,
                null_cond_emb
            )


        B, D = x.shape
        assert D == self.in_dim
        x = x.view(B, self.num_points, self.in_channels)    
        x = x.permute(0, 2, 1)                              
 
        x = self.first_layer(x)                            

        for block in self.blocks:
            x = block(x, time_emb, cond_emb)                

        x = self.final_layer(F.silu(x))                     

        x = x.permute(0, 2, 1).reshape(B, -1)               

        return x