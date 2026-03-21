import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from .embeddings import PositionalEmbedding
from functools import partial


# Affine normalisation, no need to know details by know, blackbox
class Affine(nn.Module):
    #https://github.com/facebookresearch/deit/blob/263a3fcafc2bf17885a4af62e6030552f346dc71/resmlp_models.py#L16C9-L16C9
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta



class CondResNetBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, cond_emb_dim, dropout=0,
                 skip_scale=1, adaptive_scale=True, affine=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = time_emb_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.res_linear = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.map_cond = nn.Linear(time_emb_dim+cond_emb_dim, out_dim*(2 if adaptive_scale else 1))

        if affine:
            self.pre_norm = Affine(in_dim)
            self.post_norm = Affine(out_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    # linear is the 1D vector equivalent of 2d conv
    def forward(self, x, time_emb=None, cond_emb=None):
        #print(x.shape, emb.shape)
        orig = x
        emb = torch.cat((time_emb, cond_emb), dim = -1)
        params = nn.functional.silu(self.map_cond(emb).to(x.dtype))
        x = self.pre_norm(x)
        x = self.linear1(nn.functional.silu(x))
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            x = nn.functional.silu(torch.addcmul(shift, x, scale+1))
        else:
            x = nn.functional.silu(x.add_(params))

        x = self.linear2(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = self.post_norm(x)
        x = x.add_(self.res_linear(orig))
        x = x * self.skip_scale

        return x



# This is the neural network for diffusion model, classier free guidance (CFG)
class CFGResNet(torch.nn.Module):
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    def __init__(self, 
                 in_dim,
                out_dim, 
                cond_size,
                model_channel       = 128,      # model depth, 128 filters
                channel_multiply        = [1,1,1,1],# dim multiplier for each resblock layer
                dim_mult_emb    = 4,
                num_blocks      = 4,        # Number of resblocks(mid) per level.
                dropout         = 0.,           # Dropout rate.
                emb_type        = "sinusoidal",# Timestep embedding type
                dim_mult_time   = 1,        # Time embedding size
                dim_mult_cond   = 1,        # Conditional embedding size
                cond_drop_prob  = 0.0,      # Probability of using null emb
                adaptive_scale  = True,     # Feature-wise transformations, FiLM
                skip_scale      = 1.0,      # Skip connection scaling
                affine          = False,    # Affine normalization for MLP
                **kwargs
                ):

        super().__init__()

        # embedment dimension is the dimension of the labels: in our case the 5 flow conditions and the time step in diffusion
        emb_dim  = model_channel * dim_mult_emb
        time_dim = model_channel * dim_mult_time
        cond_dim = model_channel * dim_mult_cond
        block_kwargs = dict(dropout = dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)

        self.null_emb = nn.Parameter(torch.randn(emb_dim)) 
        self.cond_size = cond_size
        self.cond_drop_prob = cond_drop_prob

        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim*cond_size, emb_dim)
        self.first_layer = nn.Linear(in_dim, model_channel)
        self.blocks = nn.ModuleList()
        cout = model_channel

        # for each block (layer) apply the ResNet Block once
        # The outer loop increases the ResNet block in depth, the inner loop stacks n ResNet block with the same depth
        for level, mult in enumerate(channel_multiply):
            for _ in range(num_blocks):
                cin = cout
                cout = model_channel * mult
                self.blocks.append(CondResNetBlock(cin, cout, emb_dim, emb_dim, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_dim) # the final layer use a fully connected layer

    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def forward(self, x, cond, time, context_mask=None, cond_drop_prob=0, cond_scale = 1., rescaled_phi = 0., sampling=False):
        if sampling:
            logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=0.)
            if cond_scale == 1:
                return logits
            # drop all conditions, so this is unconditional part
            null_logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=1.) 

            # the core of CFG, cond_scale is the guidance strength. When cond_scale = 1 this is purely conditional
            scaled_logits = null_logits + (logits - null_logits) * cond_scale

            if rescaled_phi == 0.:
                return scaled_logits

            # this is to maintian robustness and stability
            std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
            rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

            # blend the original and the stabalised signals by a certain ratio
            return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)
        else:
            # if this is used for training, then just use the forward path
            return self._forward(x, cond, time, context_mask, cond_drop_prob)

    # One larger ResNet block (which contains numerous single ResNet blocks)
    def _forward(self, x, cond, time, context_mask=None, cond_drop_prob=None):
        # context_mask dummy var
        batch_size = x.shape[0]
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb.reshape(cond_emb.shape[0], -1)))
        #emb = nn.functional.silu(self.map_layer1(emb))
        
        # drop ot randomly some conditions in order to teach the model how to denoise even without a condition
        if cond_drop_prob == None:
            cond_drop_prob = self.cond_drop_prob
        if cond_drop_prob > 0:
            keep_mask = self.prob_mask_like((batch_size,), 1 - cond_drop_prob, device = x.device)
            null_cond_emb = repeat(self.null_emb, 'd -> b d', b = batch_size) 

            cond_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                cond_emb,
                null_cond_emb
            )
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x, time_emb, cond_emb)
        x = self.final_layer(nn.functional.silu(x))
        return x