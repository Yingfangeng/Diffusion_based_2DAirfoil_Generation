import os
from tqdm import trange, tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.decomposition import PCA
import yaml
import argparse
import matplotlib.pyplot as plt
import time

from .Conv_1D import *
from .MLP import *
from .Conv_1D_UNet import *
from .Conv_1D_ResNet_UNet import *
from .Conv_2D_ResNet_UNet import *


class EDM_CFG(torch.nn.Module):
    def __init__(self,
        in_dim, 
        out_dim, 
        cond_size,
        model_channel      = 128,
        channel_multiply   = [1,1,1,1],
        dim_mult_emb       = 4,
        num_blocks         = 4,
        dropout            = 0.,
        emb_type           = "sinusoidal",
        dim_mult_time      = 1,
        use_fp16           = False,
        sigma_min          = 0.02,
        sigma_max          = 80.,
        sigma_data         = 0.5,
        nn_structure       = 'MLP',
        **model_kwargs,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.label_dim = cond_size
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.nn_structure = nn_structure

        if nn_structure == 'MLP':
            self.model = CFGResNet(self.in_dim, self.out_dim, self.label_dim, model_channel=model_channel, channel_multiply=channel_multiply, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                           dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)

        elif nn_structure == 'Conv_1D':
            self.model = Conv1DCFGResNet(self.in_dim, self.out_dim, self.label_dim, model_channel=model_channel, channel_multiply=channel_multiply, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                           dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)

        elif nn_structure == 'Conv_1D_Unet':
            self.model = Conv1DUNetCFG(self.in_dim, self.out_dim, self.label_dim, model_channel=model_channel, channel_multiply=channel_multiply, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                       dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)
        
        elif nn_structure == 'ResNet_UNet':
            self.model = Conv1DResNetUNetCFG(self.in_dim, self.out_dim, self.label_dim, model_channel=model_channel, channel_multiply=channel_multiply, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                       dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)

        elif nn_structure == 'ResNet_UNet_2D':
            self.model = Conv2DResNetUNetCFG(self.in_dim, self.out_dim, self.label_dim, model_channel=model_channel, channel_multiply=channel_multiply, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                       dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False,  **model_kwargs):

        x = x.to(torch.float32)  # the input noisy signal
        # sigma = sigma.to(torch.float32).reshape(-1, 1) # the noisy intensity !!!!!

        sigma = sigma.to(torch.float32)
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
    def __init__(self, coordinates, cond_data, name, mode = None):
        self.coordinates = coordinates
        self.cond_data = cond_data
        self.mode = mode
        self.name = name

    def __len__(self):
        return len(self.cond_data)

    def __getitem__(self, idx):
        if self.mode == 'sdf':
            name_label = self.name[idx]
            # coordinates = self.coordinates[name_label]
            cond_data = self.cond_data[idx]
            sdf = self.coordinates[name_label]
            sdf = np.expand_dims(sdf, 0)

            return torch.FloatTensor(sdf), torch.FloatTensor(cond_data)

        elif self.mode == '3D_coordinates':
            name_label = self.name[idx]
            cond_data = self.cond_data[idx]
            coordinates = self.coordinates[name_label]

            return torch.tensor(coordinates, dtype=torch.float32), torch.tensor(cond_data, dtype=torch.float32)


        else:
            coordinates = self.coordinates[idx]
            cond_data = self.cond_data[idx]
            return torch.tensor(coordinates, dtype=torch.float32), torch.tensor(cond_data, dtype=torch.float32)



class EDMLoss:

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):

        # rnd_normal = torch.randn([images.shape[0], 1], device=images.device) 
        # sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # y = images
        # n = torch.randn_like(y) * sigma
        # D_yn = net(y + n, sigma, labels)
        # loss = weight * ((D_yn - y) ** 2)

        rnd = torch.randn([images.shape[0], 1], device=images.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        sigma = sigma.view([images.shape[0]] + [1] * (images.ndim - 1))

        weight = (sigma ** 2 + self.sigma_data ** 2) / ((sigma * self.sigma_data) ** 2 + 1e-8)

        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)

        return loss



# This is the model deployment function, which start with a pure nosie and apply the edm model repeatedly until sigma = 0
def edm_sampler(
    net,   # net is the model
    latents,# this is pure noise
    class_labels=None, # the condition, re, ma, aoa, cl, cd
    randn_like=torch.randn_like,
    num_steps = 18, # number of denoising steps used, a hyper parameter
    sigma_min = 0.002, # smallest noise level
    sigma_max = 80, # largest noise level
    rho = 7, # a value to control how hoise levels are spaced
    S_churn = 40, 
    S_min = 0.05, 
    S_max = 50, 
    S_noise = 1.003,
    deterministic=False
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
        # B = x_hat.shape[0]
        # x_hat = x_hat.view(B, 1, 256, 256)
        # denoised = net(x_hat, repeat(t_hat.reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        # denoised = denoised.view(B, -1)

        # d_cur = (x_hat - denoised) / t_hat
        # x_next = x_hat + (t_next - t_hat) * d_cur

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, repeat(t_next.reshape(-1), 'w -> h w', h=x_next.shape[0]), class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # ========new============
        # Euler step.
        B = x_hat.shape[0]
        sigma_hat = t_hat.expand(B, 1)
        denoised = net(x_hat, sigma_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            sigma_next = t_next.expand(B, 1)
            denoised = net(x_next, sigma_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        whole_trajectory[i] = x_next # record the entire denoisy steps

    return x_next, whole_trajectory




class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = []
        for _ in seeds:
            g = torch.Generator(device)
            g.seed()
            # g.manual_seed(1)
            self.generators.append(g)

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators) # assert is a checker that assess whether the two variable dimensinos are matched
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input): # generate the random number in the same shape (dimension) as the input tensor
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])



def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['best_val_loss']



def load_blade_curve(filename):
    profiles = []
    current_profile = []
    profile_has_error = False
    with open(filename, "r") as f:
        for index, line in enumerate(f):
            line = line.strip()

            # New profile marker
            if line.startswith("#Profile"):
                if current_profile:
                    profiles.append(np.array(current_profile))
                    current_profile = []
                continue

            # Skip comments / empty lines
            if not line or line.startswith("#"):
                continue

            # Read coordinates
            values = line.split()
            if len(values) >= 3:
                try:
                    x = float(values[0])
                    y = float(values[1])
                    z = float(values[2])
                except ValueError:
                    profile_has_error = True
                    # print("ValueError line:", repr(line))
                    continue

                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    profile_has_error = True
                    # print('There is invalid value in line', index, 'skipped the invalid values!')
                    pass

                else:
                    current_profile.append([x, y, z])
        # Append last profile
        if current_profile:
            profiles.append(np.array(current_profile))

    return profiles, profile_has_error







#======================== The Main Function ==========================





if __name__ == '__main__':


    # =================== Load the Dataset & Hyperparameters ====================

    parser = argparse.ArgumentParser(description="Run diffusion model with YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config_file = args.config

    # Load YAML config
    with open(config_file, "r") as f:
        model_config = yaml.safe_load(f)

    df = pd.read_csv(model_config['dataset_csv_path'])
    length = len(df)
    # df = df.sample(n=int(0.1*length), random_state=42)
    data_structure = model_config['data_structure']
    learning_rate = float(model_config['learning_rate'])
    num_epochs = model_config['num_epochs']
    batch_size = model_config['batch_size']
    model_channel = model_config['model_channel']
    model_layer = model_config['model_layer']
    model_channel_multiplication = model_config['model_channel_multiplication']
    cond_scale =model_config['cond_scale']    # CFG guidance scale
    rescaled_phi = model_config['rescaled_phi'] # mixing ratio of the std_function
    train_val_division = model_config['train_val_division'] # the percentage of training data in the dataset
    device=model_config['device']
    nn_structure=model_config['neural_network_sturcture']
    reduced_data_fraction=model_config['reduced_data_fraction']
    # condition = model_config['condition']
    model_code = f"./mdl_weight/{data_structure}_{nn_structure}_{model_channel}_{model_layer}_{len(model_channel_multiplication)}_with_{num_epochs}_epochs_{reduced_data_fraction}_data_resume"
    save_path = f"{model_code}_3D_debug.pth"
    check_point_path = f"{model_code}_check_point.pth"
    print(f'The model weight will be saved to path {save_path}')
    print(f'This training uses {int(reduced_data_fraction*100)}% of the entire dataset')
    combined_coordinates = []
    cond_data = []



    if data_structure != '1D_params' and data_structure != '3D_coordinates':
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
        mode = 'pca'
        num_components = int(model_config['component_number'])
        pca = PCA(n_components=num_components)
        coordinates = pca.fit_transform(coordinates)
        print(f'Data structure is PCA with {num_components} PCs, neural network architecture is {nn_structure}. Learning rate is {learning_rate}, Number of epochs {num_epochs}, Batch size {batch_size}, Model channel {model_channel}, Number of blocks {model_layer}, Model dimension multiplication {model_channel_multiplication}, Device is {device}')

    elif data_structure == 'raw_coordinates':
        mode = 'raw_coords'
        num_components = int(2*len(x_coords))
        print(f'Data structure is raw coordinates with {num_components} numbers, neural network architecture is {nn_structure}, Learning rate is {learning_rate}, Number of epochs {num_epochs}, Batch size {batch_size}, Model channel {model_channel}, Number of blocks {model_layer}, Model dimension multiplication {model_channel_multiplication}, Device is {device}')

    elif data_structure == 'sdf':
        num_components = int(model_config['component_number'])
        sdf_path = model_config['sdf_path']
        mode = 'sdf'
        coordinates = {}
        unique_names = df['name'].unique()
        for i in unique_names:
            sdf = np.load(f'{sdf_path}/{i}.npy')
            coordinates[i] = sdf

    elif data_structure == '1D_params':

        mode = '1D_params'
        
        name = 'trivial'

        num_components = int(model_config['component_number'])

        R_tip_1 = df['R_tip_1'].to_numpy()
        R_mean_1 = df['R_mean_1'].to_numpy()
        R_hub_1 = df['R_hub_1'].to_numpy()
        beta_b1_hub = df['beta_b1_hub'].to_numpy()
        beta_b1_tip = df['beta_b1_tip'].to_numpy()
        beta_b1_mean = df['beta_b1_mean'].to_numpy()
        beta_b2 = df['beta_b2'].to_numpy()
        R_mean_2 = df['R_mean_2'].to_numpy()
        b_2 = df['b_2'].to_numpy()
        L_z = df['L_z'].to_numpy()
        t = df['t'].to_numpy()
        nblades = df['nblades'].to_numpy()
        n_splitter_blades = df['n_splitter_blades'].to_numpy()
        b3 = df['b3'].to_numpy()
        r3 = df['r3'].to_numpy()
        slip_factor = df['slip_factor'].to_numpy()

        m_dot = df['m_dot'].to_numpy()
        omega = df['omega'].to_numpy()
        pressure_ratio = df['pressure_ratio'].to_numpy()
        efficiency = df['efficiency'].to_numpy()
        
        coordinates = []
        cond_data = []
        

        for i in range(len(df)):
            coordinates.append([R_tip_1[i], R_mean_1[i], R_hub_1[i], beta_b1_hub[i], beta_b1_tip[i], beta_b1_mean[i], 
                               beta_b2[i], R_mean_2[i], b_2[i], L_z[i],  t[i], nblades[i], n_splitter_blades[i], b3[i], r3[i], slip_factor[i]])
            
            cond_data.append([m_dot[i], omega[i], pressure_ratio[i], efficiency[i]])
        
        cond_size = len(cond_data[0])

    elif data_structure == '3D_coordinates':
        normalised_df = pd.read_csv('dataset/1D_compressor_geometry_normalised.csv')
        num_components = int(model_config['component_number'])
        curve_file = model_config['curve_file_path']
        mode = '3D_coordinates'

        coordinates = {}
        unique_names = df['geometry_index'].unique()

        name = []

        for i in unique_names:
            try: 
                profile, profile_has_error = load_blade_curve(f'{curve_file}/compressor_{i}.curve')
                
                if not profile_has_error:
                    coordinate = np.concatenate(profile).ravel()
                    compressor_name = f'compressor_{i}'
                    coordinates[compressor_name] = coordinate
                    idx = df[df['geometry_index'] == i].index
                    df_2 = normalised_df.loc[idx]
                    
                    for _, row in df_2.iterrows():
                        cond_data.append([row['m_dot'], row['omega'], row['pressure_ratio'], row['efficiency']])
                        name.append(compressor_name)
            
            except FileNotFoundError:
                continue

        cond_size = len(cond_data[0])

    else:
        raise NotImplementedError

    dataset = Aerofoil_Dataset(coordinates, cond_data, name, mode)
    print('passed dataset initialisation')

    # generator = torch.Generator().manual_seed(0)
    # n = len(dataset)
    # n_train = int(train_val_division * n)
    # train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n - n_train], generator=generator)
    # print('passed data division')
    # np.save("dataset/1D_val_indices.npy", val_set.indices)
    # np.save("dataset/1D_train_indices.npy", train_set.indices)



    seed = 0
    n = len(dataset)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(n)
    rng.shuffle(all_indices)

    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val  # ensures total == n even if n not divisible by 10

    train_indices = all_indices[:n_train]
    val_indices   = all_indices[n_train:n_train + n_val]
    test_indices  = all_indices[n_train + n_val:]

    train_frac = reduced_data_fraction
    val_frac = reduced_data_fraction
    test_frac = reduced_data_fraction

    rng = np.random.default_rng(seed)

    train_indices = rng.choice(train_indices, size=int(train_frac * len(train_indices)), replace=False)
    val_indices   = rng.choice(val_indices,   size=int(val_frac   * len(val_indices)),   replace=False)
    test_indices  = rng.choice(test_indices,  size=int(test_frac  * len(test_indices)),  replace=False)


    train_set = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_set   = torch.utils.data.Subset(dataset, val_indices.tolist())
    test_set  = torch.utils.data.Subset(dataset, test_indices.tolist())

    print("passed data division")
    print(len(train_set), len(val_set), len(test_set))

    # Save the dataset division indices
    # np.save("dataset/1D_train_indices_proper_division.npy", train_indices)
    # np.save("dataset/1D_val_indices_proper_division.npy", val_indices)
    # np.save("dataset/1D_test_indices_proper_division.npy", test_indices)


    Training = True
    

    # load the data set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    # test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print('passed data loader')
    # load the model
    model = EDM_CFG(num_components, num_components, cond_size=cond_size, model_channel=model_channel,
                    channel_multiply=model_channel_multiplication, dim_mult_emb=4, num_blocks=model_layer,
                    dropout=0, emb_type="sinusoidal", dim_mult_time=1, nn_structure=nn_structure,
                    dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=False, data_structure=data_structure, 
                    number_of_pc = num_components)
    loss_fn = EDMLoss()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print('passed model initialisation')

    if Training:

        model.train()
        model.to(device)

        # initialise the optimiser and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6, last_epoch=-1)
        scheduler_iters = len(train_loader)
        print('passed the optimiser initialisation')
        
        # intiailise tracking variables
        loss_v = []
        loss_avg = []

        best_val_loss = float('inf')
        
        if model_config['restart'] == True:
            last_epoch, best_val_loss = load_checkpoint(check_point_path, model, optimizer, scheduler, device)
            start_epoch = last_epoch + 1
            print(f"Resuming from epoch {start_epoch}, previous lowest loss is={best_val_loss}")
        else:
            start_epoch = 0
            print("Starting new training run.")

        print('training ...')
        for epoch in trange(start_epoch, num_epochs, initial=start_epoch): # use trange to create a process bar with tqdm
            model.train()
            train_loss = 0.
            num_items = 0

            # for step, batch in enumerate(train_loader):
            for step, batch in tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Batch (epoch {epoch})",
                            leave=False):
                x = batch[0]
                c = batch[1]
                x = x.to(device)
                c = c.to(device)

                tmp_loss = loss_fn(model, x, c)
                loss = tmp_loss.sum().mul(1/x.shape[0])
                
                # Compute gradients, update model weights and progress learning rates
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    loss = tmp_loss.sum().mul(1/x_val.shape[0])
                    val_loss += loss.item() * x_val.size(0)

            val_loss /= len(val_loader.dataset)
            
            

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
            # Optionally print epoch statistics
            save_checkpoint(check_point_path, epoch, model, optimizer, scheduler, best_val_loss)
            print(f'Epoch {epoch}: Training Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss:.4f}')


    else:
        model.load_state_dict(torch.load(save_path))
        model.to(device)
    print(f'Training completed, model weights written to {save_path}')