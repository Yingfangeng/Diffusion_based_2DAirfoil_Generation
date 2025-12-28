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
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

def sort_for_xfoil(x_list,y_list):
    coords = np.column_stack((x_list, y_list))

    # 1. start at trailing edge (max x)
    te_index = np.argmax(coords[:, 0])
    coords = np.roll(coords, -te_index, axis=0)

    # 2. find leading edge (min x)
    le_index = np.argmin(coords[:, 0])

    # 3. split into surfaces
    upper = coords[:le_index+1]   # TE → LE
    lower = coords[le_index:]     # LE → TE

    # 4. recombine into XFOIL order
    xfoil_coords = np.vstack((upper, lower))

    x_sorted = xfoil_coords[:, 0]
    y_sorted = xfoil_coords[:, 1]

    return x_sorted, y_sorted



def xfoil_calculation(x,y,AOA,Re,Ma,CL,CD, x_foil_timeout=5):
    
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

            stdout, stderr = ps.communicate(input=xfoil_commands.encode(), timeout = x_foil_timeout)

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

    try:
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
    except FileNotFoundError:
        CL_actual = 999
        CD_actual = 999
    return CL_actual, CD_actual



def model_validation(model, model_code, data_structure, device, sample_percent, pca = None, num_steps = None, 
                     manual_seed = None, x_foil_timeout = 20, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100,
                     mode = 'interpolation', name = None):

    model.eval()
    sample_size = 1
    design = 0
    
    dataframe = pd.read_csv('extrapolation.csv')
    
    # if mode =='interpolation':
    #     dataframe = pd.read_csv('interpolation.csv').sample(n=100, random_state=manual_seed)
    # elif mode == 'extrapolation':
    #     #.sample(n=100, random_state=manual_seed)
    # elif mode == 'pareto':
    #     dataframe = pd.read_csv('pareto.csv').sample(n=100, random_state=manual_seed)
    # elif mode == 'intermediate':
    #     dataframe = pd.read_csv('intermediate.csv').sample(n=100, random_state=manual_seed)
    
    min_max = pd.read_csv("dataset/min_max_clean.csv")
    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]

    
    output_file = f'mdl_validation/{model_code}_extrapolation.csv'
    print(f'The validation results will be stored at {output_file}')
    if os.path.exists(output_file):
        print(f"Existing file '{output_file}', a new validation csv with suffix v2 will be created.")
        output_file = f'mdl_validation/{model_code}_extrapolation_v2.csv'
    results_df = pd.DataFrame()
    idx = 1

    for _, row in tqdm(dataframe.iterrows()):
        
        Ma  = row['Ma']
        Re  = row['Re']
        AOA = row['AOA']
        CD  = row['CD']
        normalised_Ma = (Ma-min_max['Ma_min'].loc[0])/(min_max['Ma_max'].loc[0]-min_max['Ma_min'].loc[0])
        normalised_Re = (Re-min_max['Re_min'].loc[0])/(min_max['Re_max'].loc[0]-min_max['Re_min'].loc[0])
        normalised_AOA = (AOA-min_max['AOA_min'].loc[0])/(min_max['AOA_max'].loc[0]-min_max['AOA_min'].loc[0])
        normalised_CD = (CD-min_max['CD_min'].loc[0])/(min_max['CD_max'].loc[0]-min_max['CD_min'].loc[0])
        
        CL_start = row['CL']
        CL_end = CL_start * 1.05
        CD = CD * 0.95
        CL_extrapolation_list = np.linspace(CL_start, CL_end, 10)
        cannot_go_further = False
        

        for CL in CL_extrapolation_list:
            if not cannot_go_further: 
                normalised_CL = (CL-min_max['CL_min'].loc[0])/(min_max['CL_max'].loc[0]-min_max['CL_min'].loc[0])
                cond = (torch.tensor([normalised_AOA, normalised_Ma, normalised_Re, normalised_CL, normalised_CD]).to(device))
        
                iteration_count = 0
                unfeasible_design_count = 0
                max_iteration_count = max_iteration
                iteration_count  = 0
                next = 0
                CL_error_list = []
                CD_error_list = []
                CL_actual_list_2 = []
                CD_actual_list_2 = []

                while iteration_count < max_iteration_count and next == 0:
                    
                    if data_structure != 'sdf':
                        rnd = StackedRandomGenerator(device, range(sample_size))
                        latents = rnd.randn([sample_size, model.in_dim], device=device)
                        with torch.no_grad():
                            samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
                        
                        samples = samples.float()
                        sample = samples[0].cpu().numpy()
                        
                        if data_structure == 'pca':
                            sample = pca.inverse_transform(sample)

                        x_coords = sample[0::2]
                        y_coords_normalised = sample[1::2]
                        y_coords = y_coords_normalised*(y_max - y_min)+y_min
                    
                    else:
                        sdf_max = min_max['sdf_max'].loc[0]
                        sdf_min = min_max['sdf_min'].loc[0]
                        x_resolution = 128
                        y_resolution = 128
                        x_lower_lim = -0.01
                        x_upper_lim = 1.01
                        y_lower_lim = -0.17
                        y_upper_lim = 0.27
                        rnd = StackedRandomGenerator(device, range(sample_size))
                        latents = torch.randn(sample_size, 1, y_resolution, x_resolution, device=device)
                        with torch.no_grad():
                            samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=rnd.randn_like, num_steps=num_steps, deterministic=True) 
                        samples = samples.float()
                        sample_img = samples[0, 0].cpu().numpy()
                        
                        xs = np.linspace(x_lower_lim, x_upper_lim, x_resolution)
                        ys = np.linspace(y_lower_lim, y_upper_lim, y_resolution)

                        sdf= (sample_img + 1)*(sdf_max - sdf_min)/2 + sdf_min
                        # sdf_denormalised = sample_img

                        fig = plt.figure()
                        ax0 = fig.add_subplot(111)
                        edge = ax0.contour(xs, ys, sdf, levels=[0])
                        aerofoil_edge = edge.allsegs[0]
                        plt.close(fig) 

                        x_list = aerofoil_edge[0][:, 0]
                        y_list = aerofoil_edge[0][:, 1]
                        x_coords, y_coords = sort_for_xfoil(x_list, y_list)
                    
                    # xfoil to calculate the data for the genrated design
                    CL_actual, CD_actual = xfoil_calculation(x_coords, y_coords, AOA, Re, Ma, CL, CD, x_foil_timeout)
                    CL_error = abs((CL_actual - CL)/CL)
                    CD_error = abs((CD_actual - CD)/CD)
                    
                    if CL_error <= CL_tolerence and CD_error <= CD_tolerence:
                        CL_output = CL_actual
                        CD_output = CD_actual
                        next = 1


                    else:
                        CL_error_list.append(CL_error)
                        CD_error_list.append(CD_error)
                        CL_actual_list_2.append(CL_actual)
                        CD_actual_list_2.append(CD_actual)
                        
                    if CL_actual == 999:
                        unfeasible_design_count = unfeasible_design_count + 1

                    iteration_count = iteration_count + 1
                
                    

                if iteration_count == max_iteration_count and next == 0:
                    index = CL_error_list.index(min(CL_error_list))
                    CL_output = CL_actual_list_2[index]
                    CD_output = CD_actual_list_2[index]


                    print(f'Max {max_iteration_count} design iteration reached\n'
                            f'Using the best matching result with CL {CL_actual_list_2[index]} with error {CL_error_list[index]}, CD {CD_actual_list_2[index]} with error {CD_error_list[index]}.')
                    if CD_actual_list_2[index] == 999:
                        cannot_go_further = True
                else:
                    print(f'Took {iteration_count} design iteration(s)\n'
                        f'The design has CL {CL_actual} ({100*CL_error}%) and CD {CD_actual} ({100*CD_error}%).')

            new_row = {"idx": idx, "x": x_coords, "y": y_coords, "Re": Re, "Ma": Ma, "AOA": AOA, "CL": CL, "CD": CD, "CL_actual": CL_output, "CD_actual": CD_output, "design_iteration": iteration_count, "unfeasible_design": unfeasible_design_count}
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            
            results_df.to_csv(output_file, index=False)
        idx = idx +1



def model_deployment( model_config_path, sample_percent, num_steps = None,  manual_seed = None,
                     x_foil_timeout=20, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100, 
                     mode = 'interpolation', name = 'pareto_optimisation'):
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
    print('Model Code', model_code)
    num_components=model_config['component_number']
    
    if data_structure == 'pca':
        
        df = pd.read_csv("dataset/aerofoil_data_clean_normalised.csv")
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

    elif data_structure == 'sdf':
        
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

    model_validation(model, model_code, data_structure, device, sample_percent, pca = pca, num_steps = num_steps, 
                     manual_seed = manual_seed, x_foil_timeout = x_foil_timeout, CL_tolerence = CL_tolerence, 
                     CD_tolerence = CD_tolerence, max_iteration = max_iteration, mode = mode, name = name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion model with YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    parser.add_argument("mode", type=str, help="mode")
    args = parser.parse_args()
    config_file = args.config
    mode = args.mode
    model_deployment(config_file, sample_percent = 0.01, num_steps = 30,  manual_seed = 123,
                     x_foil_timeout=5, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100, mode=mode, 
                     name = 'pareto_optimisation_1.05_CL_0.95_CD')