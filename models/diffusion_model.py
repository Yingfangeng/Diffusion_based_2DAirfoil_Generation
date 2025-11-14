import io
from PIL import Image
import os
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.decomposition import PCA
from functools import partial
from einops import rearrange, repeat

from .embeddings import PositionalEmbedding
from .Conv_1D import *
from .MLP import *





# The seeding process to ensure training process is repeatable
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# EDM is a modified diffusion model (comapred to the orignal DDPM)
# and its core is still the CFG architecture, which then use the ResNet neural network
class EDM_CFG(torch.nn.Module):
    def __init__(self,
        in_dim, out_dim, cond_size,
        model_dim      = 128,      # dim multiplier.
        dim_mult        = [1,1,1,1],# dim multiplier for each resblock layer.
        dim_mult_emb    = 4,
        num_blocks      = 4,        # Number of resblocks(mid) per level.
        dropout         = 0.,      # Dropout rate.
        emb_type        = "sinusoidal",# Timestep embedding type
        dim_mult_time  = 1,        # Time embedding size
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'MLP',
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()  # pass all the objects from the torch.nn.Module to this class
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.label_dim = cond_size
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        if model_type == 'MLP':
            self.model = CFGResNet(self.in_dim, self.out_dim, self.label_dim, model_dim=model_dim, dim_mult=dim_mult, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                           dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)

        elif model_type == 'Conv_1D':
            self.model = Conv1DCFGResNet(self.in_dim, self.out_dim, self.label_dim, model_dim=model_dim, dim_mult=dim_mult, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                           dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)


    def forward(self, x, sigma, class_labels=None, force_fp32=False,  **model_kwargs):

        x = x.to(torch.float32)  # the input noisy signal
        sigma = sigma.to(torch.float32).reshape(-1, 1) # the noisy intensity
        class_labels = None if (self.label_dim == 0 or class_labels is None) else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.model((x_in).to(dtype), class_labels, c_noise.flatten(), **model_kwargs)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)



class Aerofoil_Dataset(Dataset):
    def __init__(self, coordinates, cond_data):
        self.coordinates = coordinates
        self.cond_data = cond_data
    
    def __len__(self):
        return len(self.cond_data)

    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        cond_data = self.cond_data[idx]

        return torch.tensor(coordinates, dtype=torch.float32), torch.tensor(cond_data, dtype=torch.float32)



class EDMLoss:
    # 1D EDM Loss. Add a random noise level to the clean signal and ask the network to predict the clean signal. Calculate the weighted MSE as the loss function.
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        # net is the model, image is the coordaintes, labels are the conditions (re, ma, aoa, cl, cd)
        rnd_normal = torch.randn([images.shape[0], 1], device=images.device) # the pure noise
        sigma = (rnd_normal * self.P_std + self.P_mean).exp() # noise intensity is also a generated number
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma # the noise to be added to the clear signal
        D_yn = net(y + n, sigma, labels) # produce the denoised signal from the model
        loss = weight * ((D_yn - y) ** 2) # evaluate the mse error compared to the actual clear signal as the loss
        return loss



#  This is the model deployment function, which start with a pure nosie and apply the edm model repeatedly until sigma = 0
def edm_sampler(
    net,   # net is the model
    latents,# this is pure nosei
    class_labels=None, # the condition, re, ma, aoa, cl, cd
    randn_like=torch.randn_like,# 
    num_steps = 18, # number of denoising steps used, a hyper parameter
    sigma_min = 0.002, # smallest noise level
    sigma_max = 80, # largest noise level
    rho = 7, # a value to control how hoise levels are spaced
    S_churn = 0, 
    S_min = 0, 
    S_max = float('inf'), 
    S_noise = 0,
    deterministic=False # if true then the results will be fully reproducable given the same seeds
):  # rho is a hyperparam that can be used to tune the amount of noise added per step

    # Adjust noise levels based on what's supported by the network.
    # Clip to the range that the network is trained on such that it can be used to predict the noise level
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization. T_steps calculates the noisy levels that is going to be added to each step. It takes the value of a non-uniform noise distribution. 
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)

    # The schedule of the noise intensity along the timeline. Non-uniformity controlled by the hyperparameter rho.
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # Ensure the final step is always t_N = 0

    # The first step
    x_next = latents.to(torch.float64) * t_steps[0]

    whole_trajectory = torch.zeros((num_steps, *x_next.shape), dtype=torch.float64)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1

        x_cur = x_next
        if not deterministic:
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur # the noise we added at this particular step
            x_hat = x_cur # the noisy signal that is to be denoised
        # Euler step.
        denoised = net(x_hat, repeat(t_hat.reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, repeat(t_next.reshape(-1), 'w -> h w', h=x_next.shape[0]), class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        whole_trajectory[i] = x_next # record the entire denoisy steps

    return x_next, whole_trajectory




class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators) # assert is a checker that assess whether the two variable dimensinos are matched
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input): # generate the random number in the same shape (dimension) as the input tensor
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])





#======================== The Main Function ==========================





if __name__ == '__main__':


    # =================== Load the Dataset ====================
    df = pd.read_csv("aerofoil_data_normalised.csv")
    data_structure = 'raw_coordinates'   # pca / raw_coordiantes
    save_path = './mdl_weight/test.pth' # store the trained model weights

    combined_coordinates = []
    cond_data = []
    for _, row in df.iterrows():
        x_coords = np.fromstring(row["x"].strip("[]"), sep=" ")
        y_coords = np.fromstring(row["y"].strip("[]"), sep=" ")
        paired = np.column_stack((x_coords, y_coords)).flatten()
        combined_coordinates.append(paired)
    coordinates = np.array(combined_coordinates)

    name = df['name'].to_numpy()
    Ma = df['Ma'].to_numpy()
    Re = df['Re'].to_numpy()
    AOA = df['AOA'].to_numpy()
    CL = df['CL'].to_numpy()
    CD = df['CD'].to_numpy()
    for i in range(len(name)):
        cond_data.append([AOA[i], Ma[i], Re[i], CL[i], CD[i]])
    cond_size = len(cond_data[0])

    if data_structure == 'pca':
        num_components = 20
        pca = PCA(n_components=num_components)
        coordinates = pca.fit_transform(coordinates)
        model_type = 'MLP'
        print(f'Data structure is PCA with {num_components} PCs')

    elif data_structure == 'raw_coordinates':
        num_components = 600
        model_type = 'Conv_1D'
        print(f'Data structure is raw coordinates, neural network architecture is {model_type}')


    seed_everything(0)


    dataset = Aerofoil_Dataset(coordinates, cond_data)


    generator = torch.Generator().manual_seed(0)
    n = len(dataset)
    n_train = int(0.8 * n)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n - n_train], generator=generator)



    # =================== Hyperparameter Settings ====================


    learning_rate = 1E-4
    num_epochs = 200
    batch_size = 128
    cond_scale =1    # CFG guidance scale
    rescaled_phi = 0 # mixing ratio of the std_function
    device='cuda'
    Training = True
    


    # load the data set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # load the model
    model = EDM_CFG(num_components, num_components, cond_size=cond_size, model_dim=128,
                    dim_mult=[1,2,2], dim_mult_emb=4, num_blocks=10,
                    dropout=0, emb_type="sinusoidal", dim_mult_time=1, model_type=model_type,
                    dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=True)
    loss_fn = EDMLoss()



    if Training:

        model.train()
        model.to(device)

        # initialise the optimiser and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
        scheduler_iters = len(train_loader)

        # intiailise tracking variables
        loss_v = []
        loss_avg = []

        best_val_loss = float('inf')
        print('training ...')
        for epoch in trange(num_epochs): # use trange to create a process bar with tqdm
            model.train()
            train_loss = 0.
            num_items = 0

            for step, batch in enumerate(train_loader):
                x = batch[0]
                c = batch[1]
                x = x.to(device)
                c = c.to(device)

                tmp_loss = loss_fn(model, x, c)
                loss = tmp_loss.sum().mul(1/x.shape[0])
                

                # Compute gradients, update model weights and progress learning rates
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + step / scheduler_iters)
                # Store the training loss
                train_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                loss_v.append(loss.item())
                loss_avg.append(train_loss / num_items)
            
            # ============= evaluation =============
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_val = batch[0]
                    c_val = batch[1]
                    x_val = x_val.to(device)
                    c_val = c_val.to(device)

                    tmp_loss = loss_fn(model, x_val, c_val)
                    loss = tmp_loss.sum().mul(1/x.shape[0])
                    val_loss += loss.item() * x_val.size(0)

            val_loss /= len(val_loader.dataset)

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
            # Optionally print epoch statistics
            print(f'Epoch {epoch}: Training Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss:.4f}')


    else:
        model.load_state_dict(torch.load(save_path))
        model.to(device)
