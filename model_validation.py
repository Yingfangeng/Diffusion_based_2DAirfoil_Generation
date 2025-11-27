import os
import numpy as np
import torch
from sklearn.decomposition import PCA
import pandas as pd
import subprocess as sp
import random
import yaml
from models.diffusion_model import EDM_CFG, edm_sampler, StackedRandomGenerator
import argparse
from tqdm import tqdm



def xfoil_calculation(x,y,AOA,Re,Ma,CL,CD):
    
    geom_file_name = f'generated_geom/profile_{AOA:.1f}_{Ma:.1f}_{int(Re)}_{CL:.1f}_{CD:.1f}'
    result_file_name = f'generated_data/profile_{AOA:.1f}_{Ma:.1f}_{int(Re)}_{CL:.1f}_{CD:.1f}'
    xfoilpath = '/home/yg1922/Desktop/Xfoil_2/bin/xfoil'

    with open(f'{geom_file_name}', "w") as f:
        f.write(f'generated_profile_{AOA:.1f}_{Ma:.1f}_{int(Re)}_{CL:.2f}_{CD:.2f}' + "\n")  # First line
    
        for xi, yi in zip(x, y):
            f.write(f"{xi} {yi}\n")

    # Delete the previous run results
    for f in [result_file_name]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    def Xfoil(Ma, Re):
        xfoil_commands = f"""
            PLOP
            G F

            LOAD {geom_file_name}

            OPER
            ITER 500
            MACH {Ma}
            VISC {Re}
            PACC
            {result_file_name}

            ALFA {AOA}

            QUIT
            """

        try:
            ps = sp.Popen([xfoilpath],
                        stdin=sp.PIPE,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)

            stdout, stderr = ps.communicate(input=xfoil_commands.encode(), timeout = 15)

            ps.stdin.close()
            ps.stdout.close()
            ps.stderr.close()

        except sp.TimeoutExpired:
            print('error in ', geom_file_name)
            ps.kill()
            print('Take too long for xfoil to run on this profile. Possibly because the profile generated is not feasible.')
            

        except Exception as e:
            print(f"Error running {geom_file_name}: {e}")
            print('Something unexpected happened')
            ps.kill()

    Xfoil(Ma=Ma, Re=Re)


    f = open(result_file_name, 'r', errors='ignore')
    flines = f.readlines()


    if len(flines) != 12: 
        for i in range(12, len(flines)):
            words = str.split(flines[i]) 
            CL_actual    = float(words[1])
            CD_actual    = float(words[2])
            LD_r  = float(words[1]) / float(words[2])
    else:
        print('There is no output from xfoil. Possibly because the profile generated is not feasible.')
        CL_actual = 999
        CD_actual = 999
    return CL_actual, CD_actual


def model_validation(model, data_structure, device, sample_percent = 0.1, pca = None, model_code = None, num_steps = None, manual_seed = None):

    model.eval()
    df = pd.read_csv("aerofoil_data_clean_normalised.csv")
    val_indices = np.load("val_indices_clean.npy")
    
    if manual_seed != None:
        random.seed(manual_seed)
    numbers = random.sample(range(0, len(val_indices)), int(sample_percent*len(val_indices)))

    output_file = f"mdl_validation/{model_code}.csv"
    min_max = pd.read_csv('min_max_clean.csv')
    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]
    
    if os.path.exists(output_file):
        print('This model has already be validated.')
        output_file = f"mdl_validation/{model_code}_v2.csv"

    results_df = pd.DataFrame(columns=["name", "Re", "AOA", "CL", "CD", "CL_actual", "CD_actual"])

    for idx in tqdm(numbers):
        
        i = val_indices[idx]

        name = df.loc[i,'name']
        Ma_normalised = df.loc[i, 'Ma']
        Re_normalised = df.loc[i, 'Re']
        AOA_normalised = df.loc[i, 'AOA']
        CL_normalised = df.loc[i, 'CL']
        CD_normalised = df.loc[i, 'CD']

        
        # denormalise the condition data
        Ma = Ma_normalised*(min_max['Ma_max'].loc[0]-min_max['Ma_min'].loc[0])+min_max['Ma_min'].loc[0]
        Re = Re_normalised*(min_max['Re_max'].loc[0]-min_max['Re_min'].loc[0])+min_max['Re_min'].loc[0]
        AOA = AOA_normalised*(min_max['AOA_max'].loc[0]-min_max['AOA_min'].loc[0])+min_max['AOA_min'].loc[0]
        CL = CL_normalised*(min_max['CL_max'].loc[0]-min_max['CL_min'].loc[0])+min_max['CL_min'].loc[0]
        CD = CD_normalised*(min_max['CD_max'].loc[0]-min_max['CD_min'].loc[0])+min_max['CD_min'].loc[0]

        # convert the condition data into a tensor
        cond = (torch.tensor([AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised]).to(device))

        # use the model to generate coordinates
        rnd = StackedRandomGenerator(device, range(1))
        latents = rnd.randn([1, model.in_dim], device=device)
        with torch.no_grad():
            samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=rnd.randn_like, deterministic=False, num_steps=num_steps) 
        samples = samples.float()
        sample = samples[0].cpu().numpy()
        
        if data_structure == 'pca':
            sample = pca.inverse_transform(sample)

        x_coords = sample[0::2]
        y_coords_normalised = sample[1::2]
        y_coords = y_coords_normalised*(y_max - y_min)+y_min

        # xfoil to calculate the data for the genrated design
        CL_actual, CD_actual = xfoil_calculation(x_coords, y_coords, AOA, Re, Ma, CL, CD)
        new_row = {"name": name, "Re": Re, "AOA": AOA, "CL": CL, "CD": CD, "CL_actual": CL_actual, "CD_actual": CD_actual}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    
    results_df.to_csv(output_file, index=False)



def model_deployment(model_config_path, sample_percent = None, num_steps = None, manual_seed = None):
    
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    data_structure = model_config['data_structure']
    num_epochs = model_config['num_epochs']
    model_channel = model_config['model_channel']
    model_layer = model_config['model_layer']
    model_channel_multiplication = model_config['model_channel_multiplication']
    device=model_config['device']
    nn_structure=model_config['neural_network_sturcture']
    model_code = f"{data_structure}_{nn_structure}_{model_channel}_{model_layer}_{len(model_channel_multiplication)}_with_{num_epochs}_epochs"
    save_path = f"mdl_weight/{model_code}.pth"
    print(f'Model Code is {model_code}')
    num_components=model_config['component_number']

    if data_structure == 'pca':
        
        df = pd.read_csv("aerofoil_data_normalised_256.csv")
        combined_coordinates = []
        for _, row in df.iterrows():
            x_coords = np.fromstring(row["x"].strip("[]"), sep=" ")
            y_coords = np.fromstring(row["y"].strip("[]"), sep=" ")
            paired = np.column_stack((x_coords, y_coords)).flatten()
            combined_coordinates.append(paired)
        coordinates = np.array(combined_coordinates)
        pca = PCA(n_components=num_components)
        coordinates = pca.fit_transform(coordinates)

        model = EDM_CFG(num_components, num_components, cond_size=5, model_channel=model_channel,
                channel_multiply=model_channel_multiplication, dim_mult_emb=4, num_blocks=model_layer,
                dropout=0, emb_type="sinusoidal", dim_mult_time=1, nn_structure=nn_structure,
                dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=False, data_structure=data_structure, 
                    number_of_pc = num_components)
        model.load_state_dict(torch.load(save_path))


    elif data_structure == 'raw_coordinates':

        model = EDM_CFG(num_components, num_components, cond_size=5, model_channel=model_channel,
                channel_multiply=model_channel_multiplication, dim_mult_emb=4, num_blocks=model_layer,
                dropout=0, emb_type="sinusoidal", dim_mult_time=1, nn_structure=nn_structure,
                dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=False, data_structure=data_structure, 
                    number_of_pc = num_components)
        pca = None
        model.load_state_dict(torch.load(save_path))


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model_validation(model, data_structure, device, sample_percent, pca, model_code, num_steps=num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion model with YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    config_file = args.config
    model_deployment(config_file, sample_percent=0.01, num_steps=30, manual_seed=123)