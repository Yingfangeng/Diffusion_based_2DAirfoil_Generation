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
import signal
from meanline.meanline import MeanLine





class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def run_meanline(geometry, m_dot, omega, timeout=10):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        meanline = MeanLine(geometry)
        meanline.execution_impeller_inlet(m_dot, omega, 'Centrifugal')
        meanline.execution_impeller_outlet(m_dot, omega, 'Centrifugal')
        meanline.execution_vaneless_diffuser('Impeller', 'Centrifugal', m_dot)
        signal.alarm(0)
        return meanline.pressure_ratio, meanline.stage_eff

    except TimeoutException:
        print("Take too long for meanline to converge!")
        return 999, 999

    except Exception as e:
        signal.alarm(0)
        print('Meanline not converged')
        return 999, 999




def validation_1D(device, sample_percent, model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, model_code):
    
    model.eval()
    pr_tolerance = CL_tolerence
    eta_tolerance = CD_tolerence
    df = pd.read_csv('extrapolation_1D.csv')
    min_max = pd.read_csv('dataset/1D_compressor_geometry_minmax.csv')


    if ("min" in min_max.columns) and ("max" in min_max.columns):
        feature_col = min_max.columns[0]  # usually "Unnamed: 0"
        if feature_col not in ["min", "max"]:
            min_max = min_max.set_index(feature_col)

    # Clean any whitespace issues in feature names
    min_max.index = min_max.index.astype(str).str.strip()
    
    sample_number = 1

    output_file = f'mdl_validation/{model_code}_extrapolation.csv'
    print(f'The validation output will be stored at {output_file}')

    if os.path.exists(output_file):
        print(f"Existing file '{output_file}', a new validation csv with suffix v2 will be created.")
        output_file = f'mdl_validation/{model_code}_extrapolation_v2.csv'
    results_df = pd.DataFrame()


    for idx in range(len(df)):
        i = idx + 1
        design = 0
        
        while design < multiple_design:

            pr_original = df.loc[i, 'pr']
            eta_original = df.loc[i, 'eta'] + 0.02
            omega = df.loc[i, 'omega']
            m_dot = df.loc[i, 'm_dot']

            m_dot_normalised = (m_dot - min_max.loc['m_dot', 'min'])/(min_max.loc['m_dot', 'max'] - min_max.loc['m_dot', 'min'])
            omega_normalised = (omega - min_max.loc['omega', 'min'])/(min_max.loc['omega', 'max'] - min_max.loc['omega', 'min'])
            pr_normalised = (pr_original - min_max.loc['pressure_ratio', 'min'])/(min_max.loc['pressure_ratio', 'max'] - min_max.loc['pressure_ratio', 'min'])
            eta_normalised = (eta_original - min_max.loc['efficiency', 'min'])/(min_max.loc['efficiency', 'max'] - min_max.loc['efficiency', 'min'])
            

            number_of_trials = 1
            number_of_unfeasible_designs = 0
            success = False

            geometry_list = []
            pr_error_list = []
            eta_error_list = []
            pr_list = []
            eta_list = []

            while not success and number_of_trials < max_iteration:
                cond = (torch.tensor([m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised]).to(device))
                
                rnd = StackedRandomGenerator(device, range(sample_number))
                latents = rnd.randn([sample_number, model.in_dim], device=device)

                with torch.no_grad():
                    samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 

                samples = samples.float()
                sample = samples[0].cpu().numpy()

                geom_cols = ['R_tip_1', 'R_mean_1', 'R_hub_1', 'beta_b1_hub', 'beta_b1_tip', 'beta_b1_mean', 
                            'beta_b2', 'R_mean_2', 'b_2', 'L_z',  't', 'nblades', 'n_splitter_blades', 'b3', 'r3', 'slip_factor']


                mins = min_max.loc[geom_cols, "min"].to_numpy(dtype=np.float32)
                maxs = min_max.loc[geom_cols, "max"].to_numpy(dtype=np.float32)


                denormalised_geometry = sample * (maxs + 0.0000000000000001 - mins) + mins

                geometry  = {
                    'imp_type': 'Centrifugal', 
                    'P_01': 101325, 
                    'T_01': 288,
                    'R_tip_1': float(denormalised_geometry[0]), 
                    'R_mean_1': float(denormalised_geometry[1]), 
                    'R_hub_1': float(denormalised_geometry[2]),
                    'alpha_1': 0, 
                    'beta_b1_hub': float(denormalised_geometry[3]), 
                    'beta_b1_tip': float(denormalised_geometry[4]),
                    'beta_b1_mean': float(denormalised_geometry[5]),
                    'lambda_1': 1.0,
                    'beta_b2': float(denormalised_geometry[6]),
                    'R_mean_2': float(denormalised_geometry[7]), 
                    'lambda_2': 1.0, 
                    'b_2': float(denormalised_geometry[8]), 
                    'L_z': float(denormalised_geometry[9]), 
                    't': float(denormalised_geometry[10]), 
                    's': 0.0003,
                    'nblades': round(denormalised_geometry[11]),
                    'n_splitter_blades': round(denormalised_geometry[12]),
                    'b3': float(denormalised_geometry[13]), 
                    'r3': float(denormalised_geometry[14]),
                    'slip_factor': float(denormalised_geometry[15])}



                RPM = (omega*60)/(2*np.pi)

                pr, eta = run_meanline(geometry, m_dot, omega, x_foil_timeout)

                number_of_trials += 1


                pr_error = abs(pr - pr_original) / (pr_original)
                eta_error = abs(eta - eta_original) / (eta_original)
                
                pr_error_list.append(pr_error)
                eta_error_list.append(eta_error)
                pr_list.append(pr)
                eta_list.append(eta)
                geometry_list.append(geometry)



                if pr_error < pr_tolerance and eta_error < eta_tolerance:
                #     print(f'Number {design} design took {number_of_trials} trials.')
                #     print(f'Pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                #     print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')
                    success = True
                elif pr == 999 or eta == 999:
                    number_of_unfeasible_designs += 1
            
            design += 1

            if number_of_trials == max_iteration:
                
                index = pr_error_list.index(min(pr_error_list))

                geometry = geometry_list[index]
                pr = pr_list[index]
                eta = eta_list[index]
                pr_error = pr_error_list[index]
                eta_error = eta_error_list[index]
                print(f'Number {design} design cannot satisfy the tolerance after {number_of_trials} trails.')
                print(f'Using the best design: pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')
                print(f'The best geometry: {geometry}')

        new_row = {"omega": omega, "m_dot": m_dot, "pr_original": pr_original, "eta_original": eta_original, "pr_actual": pr, "eta_actual": eta, "design_iteration": number_of_trials, "unfeasible_design": number_of_unfeasible_designs}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        results_df.to_csv(output_file, index=False)






def model_deployment( model_config_path, sample_percent, num_steps = None,  manual_seed = None,
                     x_foil_timeout=20, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100, mode = 'validation'):
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

    elif data_structure == '1D_params':
        
        model = EDM_CFG(num_components, num_components, cond_size=4, model_channel=model_channel,
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

    if data_structure != '1D_params':
        model_validation(model, model_code, data_structure, device, sample_percent, pca = pca, num_steps = num_steps, 
                        manual_seed = manual_seed, x_foil_timeout = x_foil_timeout, CL_tolerence = CL_tolerence, 
                        CD_tolerence = CD_tolerence, max_iteration = max_iteration, mode = mode)
    elif data_structure == '1D_params':
        validation_1D(device, sample_percent, model, num_steps, manual_seed, 1, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, model_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion model with YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    config_file = args.config
    model_deployment( config_file, sample_percent = 0.01, num_steps = 100,  manual_seed = 123,
                     x_foil_timeout=5, CL_tolerence = 0.01, CD_tolerence = 0.01, max_iteration = 100, mode = 'trivial')