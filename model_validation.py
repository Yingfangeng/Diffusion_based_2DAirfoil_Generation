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
from scipy.signal import savgol_filter

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
    except FileNotFoundError or ValueError:
        CL_actual = 999
        CD_actual = 999
    return CL_actual, CD_actual



def model_validation(model, model_code, data_structure, device, sample_percent, pca = None, num_steps = None, 
                     manual_seed = None, x_foil_timeout = 20, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100, 
                     mode = 'validation'):

    model.eval()
    sample_size = 1
    design = 0

    if mode == 'intermediate':
        output_file = f'mdl_validation/{model_code}_intermediate_flow_condition.csv'
    
    else:
        output_file = f'mdl_validation/{model_code}.csv'
    print(f'The validation results will be stored at {output_file}')
    
    if os.path.exists(output_file):
        print(f"Existing file '{output_file}', a new validation csv with suffix v2 will be created.")
        if mode == 'intermediate':
            output_file = f'mdl_validation/{model_code}_intermediate_flow_condition_v2.csv'
        else:
            output_file = f'mdl_validation/{model_code}_v2.csv'
    results_df = pd.DataFrame()


    df = pd.read_csv("dataset/aerofoil_data_clean_normalised.csv")
    df2 = pd.read_csv("dataset/aerofoil_data_clean.csv")
    val_indices = np.load("dataset/val_indices_clean.npy")
    train_indices = np.load("dataset/train_indices_clean.npy")
    min_max = pd.read_csv('dataset/min_max_clean.csv')

    if manual_seed != None:
        random.seed(manual_seed)
    numbers = random.sample(range(0, len(val_indices)), int(len(val_indices)*sample_percent))

    
    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]
    
    
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


        # This is for intermediate test use only, comment out for normal model validation!!!!
        
        
        if mode == 'intermediate':
            AOA = AOA + random.uniform(-0.5, 0.5)
            Re = Re + random.uniform(-50000, 50000)
            Ma = Ma + random.uniform(-0.05, 0.05)
            AOA_normalised = (AOA - min_max['AOA_min'].loc[0])/(min_max['AOA_max'].loc[0] - min_max['AOA_min'].loc[0])
            Re_normalised = (Re - min_max['Re_min'].loc[0])/(min_max['Re_max'].loc[0] - min_max['Re_min'].loc[0])
            Ma_normalised = (Ma - min_max['Ma_min'].loc[0])/(min_max['Ma_max'].loc[0] - min_max['Ma_min'].loc[0])


        # convert the condition data into a tensor
        cond = (torch.tensor([AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised]).to(device))

        # the exact of CL, CD within the specific range


    
        unfeasible_design_count = 0
        max_iteration_count = max_iteration
        iteration_count  = 0
        valid_design = 0
        average_trial = 0
        CL_error_list = []
        CD_error_list = []
        CL_actual_list_2 = []
        CD_actual_list_2 = []
    
        while iteration_count < max_iteration_count: # and valid_design == 0:
            
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
            CL_error = 100*abs((CL_actual - CL)/CL)
            CD_error = 100*abs((CD_actual - CD)/CD)
            
            if CL_error <= CL_tolerence and CD_error <= CD_tolerence:

                CL_output = CL_actual
                CD_output = CD_actual
                valid_design += 1
                print('Found valid design')


            else:
                CL_error_list.append(CL_error)
                CD_error_list.append(CD_error)
                CL_actual_list_2.append(CL_actual)
                CD_actual_list_2.append(CD_actual)
                
            if CL_actual == 999:
                unfeasible_design_count = unfeasible_design_count + 1

            if valid_design == 1:
                average_trial = iteration_count

            iteration_count = iteration_count + 1
            
            if iteration_count == max_iteration_count and valid_design == 0:
                index = CL_error_list.index(min(CL_error_list))
                CL_output = CL_actual_list_2[index]
                CD_output = CD_actual_list_2[index]


                print(f'Max {max_iteration_count} design iteration reached for NO.{design+1} design.\n'
                        f'Using the best matching result with CL {CL_actual_list_2[index]} with error {CL_error_list[index]}, CD {CD_actual_list_2[index]} with error {CD_error_list[index]}.')

        # else:
            # print(f'The NO.{design+1} valid design took {iteration_count} design iteration(s)\n'
            #     f'The design has CL {CL_actual} ({100*CL_error}%) and CD {CD_actual} ({100*CD_error}%).')
            # print(f'The NO.{design+1} design case resulted in {valid_design} valid designs iteration(s)')

            
        
       
        new_row = {"name": name, "Re": Re, "AOA": AOA, "CL": CL, "CD": CD, "CL_actual": CL_output, "CD_actual": CD_output, "design_iteration": average_trial, "unfeasible_design": unfeasible_design_count, "valid_design": valid_design}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"{name} done.")
        design = design +1
        results_df.to_csv(output_file, index=False)
    




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
        return meanline.impeller_pressure_ratio, meanline.impeller_eff

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
    df = pd.read_csv('dataset/New_compressor_geometry/1D_compressor_geometry_normalised.csv')
    min_max = pd.read_csv('dataset/New_compressor_geometry/1D_compressor_geometry_minmax.csv')

    val_indices = np.load('dataset/New_compressor_geometry/1D_test_indices.npy')

    if ("min" in min_max.columns) and ("max" in min_max.columns):
        feature_col = min_max.columns[0]  # usually "Unnamed: 0"
        if feature_col not in ["min", "max"]:
            min_max = min_max.set_index(feature_col)

    # Clean any whitespace issues in feature names
    min_max.index = min_max.index.astype(str).str.strip()
    
    random.seed(manual_seed)
    numbers = random.sample(range(0, len(val_indices)), int(len(val_indices)*sample_percent))
   
    print('exe', manual_seed)

    sample_number = 1

    output_file = f'mdl_validation/{model_code}.csv'
    print(f'The validation output will be stored at {output_file}')

    if os.path.exists(output_file):
        print(f"Existing file '{output_file}', a new validation csv with suffix v2 will be created.")
        output_file = f'mdl_validation/{model_code}_v2.csv'
    results_df = pd.DataFrame()


    for idx in tqdm(numbers):
        i = val_indices[idx]
        design = 0
        
        while design < multiple_design:

            pr_normalised = df.loc[i, 'imp_pressure_ratio']
            eta_normalised = df.loc[i, 'imp_efficiency']
            omega_normalised = df.loc[i, 'omega']
            m_dot_normalised = df.loc[i, 'm_dot']

            number_of_trials = 0
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

                geom_cols = ['R_tip_1', 'R_hub_1', 'beta_b1_hub', 'beta_b1_tip', 'beta_b1_mean', 
                            'beta_b2', 'R_mean_2', 'b_2', 'L_z', 's', 't', 'nblades',  'b3', 'r3']


                mins = min_max.loc[geom_cols, "min"].to_numpy(dtype=np.float32)
                maxs = min_max.loc[geom_cols, "max"].to_numpy(dtype=np.float32)


                denormalised_geometry = sample * (maxs + 0.0000000000000001 - mins) + mins


                r_tip_1 = float(denormalised_geometry[0])
                
                r_hub_1 = float(denormalised_geometry[1])
                beta_b2 = float(denormalised_geometry[5])
                nblades = round(denormalised_geometry[11])


                r_mean_1 = (2/3) * (((r_tip_1**3) - (r_hub_1**3)) / ((r_tip_1**2) - (r_hub_1**2)))
                slip_factor = 1 - (((np.cos(beta_b2))**0.5) / (nblades**0.7))


                geometry  = {
                    'imp_type': 'Centrifugal', 
                    'P_01': 101325, 
                    'T_01': 288,
                    'R_tip_1': r_tip_1, 
                    'R_mean_1': r_mean_1, 
                    'R_hub_1': r_hub_1,
                    'alpha_1': 0, 
                    'beta_b1_hub': float(denormalised_geometry[2]), 
                    'beta_b1_tip': float(denormalised_geometry[3]),
                    'beta_b1_mean': float(denormalised_geometry[4]),
                    'lambda_1': 1.0,
                    'beta_b2': float(denormalised_geometry[5]),
                    'R_mean_2': float(denormalised_geometry[6]), 
                    'lambda_2': 1.0, 
                    'b_2': float(denormalised_geometry[7]), 
                    'L_z': float(denormalised_geometry[8]), 
                    's': float(denormalised_geometry[9]),
                    't': float(denormalised_geometry[10]), 
                    'nblades': round(denormalised_geometry[11]),
                    'n_splitter_blades': 0,
                    'b3': float(denormalised_geometry[12]), 
                    'r3': float(denormalised_geometry[13]),
                    'slip_factor': slip_factor}


                m_dot = m_dot_normalised*(min_max.loc['m_dot', 'max'] - min_max.loc['m_dot', 'min']) + min_max.loc['m_dot', 'min']
                omega = omega_normalised*(min_max.loc['omega', 'max']- min_max.loc['omega', 'min']) + min_max.loc['omega', 'min']

                pr_original = pr_normalised*(min_max.loc['imp_pressure_ratio', 'max']- min_max.loc['imp_pressure_ratio', 'min']) + min_max.loc['imp_pressure_ratio', 'min']
                eta_original = eta_normalised*(min_max.loc['imp_efficiency', 'max']- min_max.loc['imp_efficiency', 'min']) + min_max.loc['imp_efficiency', 'min']

                RPM = (omega*60)/(2*np.pi)

                pr, eta = run_meanline(geometry, m_dot, omega, x_foil_timeout)

                number_of_trials += 1


                pr_error = 100*abs(pr - pr_original) / (pr_original)
                eta_error = 100*abs(eta - eta_original) / (eta_original)
                
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



def load_blade_curve(filename):
    profiles = []
    current_profile = []
    profile_has_error = False
    with open(filename, "r") as f:
        for index, line in enumerate(f):
            line = line.strip()

            # New profile marker
            if line.startswith("#Profile") or line.startswith("# profile"):
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
                    continue

                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    profile_has_error = True
                
                else:
                    current_profile.append([x, y, z])
        # Append last profile
        if current_profile:
            profiles.append(np.array(current_profile))

    return profiles, profile_has_error



def load_1D_dataset():
    df = pd.read_csv('dataset/New_compressor_geometry/1D_compressor_geometry_normalised.csv')
    min_max = pd.read_csv('dataset/New_compressor_geometry/1D_compressor_geometry_minmax.csv')
    val_indices = np.load('dataset/New_compressor_geometry/3D_test_indices.npy')

    if ("min" in min_max.columns) and ("max" in min_max.columns):
        feature_col = min_max.columns[0]
    if feature_col not in ["min", "max"]:
        min_max = min_max.set_index(feature_col)
    min_max.index = min_max.index.astype(str).str.strip()

    return df, min_max, val_indices


def randomly_pick_1D_validation(manual_seed, sample_number):
    
    df, min_max, val_indices = load_1D_dataset()
    
    random.seed(manual_seed)
    print('exe', manual_seed)
    
    numbers = random.sample(range(0, len(val_indices)), sample_number)

    return numbers
    

def cyl_to_cart_about_x(x, r, theta):
    y = r * np.cos(theta)
    z = r * np.sin(theta)
    return x, y, z




def geometry_3D_to_1D_conversion(x,y,z,number_of_blades):
    



    x_both = np.concatenate((x[:512], x[-512:]))
    y_both = np.concatenate((y[:512], y[-512:]))
    z_both = np.concatenate((z[:512], z[-512:]))
 


    # Hub profile
    
    x = x_both[:512]
    y = y_both[:512]
    z = z_both[:512]

    x_max_idx, x_min_idx = np.argmax(x), np.argmin(x)
    inlet_hub = [x[x_min_idx], y[x_min_idx], z[x_min_idx]]
    outlet_hub = [x[x_max_idx], y[x_max_idx], z[x_max_idx]]

    # PS
    idx_1 = 249
    idx_2 = idx_1+1
    inlet_hub_1 = [x[idx_1], y[idx_1], z[idx_1]]
    inlet_hub_2 = [x[idx_2], y[idx_2], z[idx_2]]
    dx = abs(inlet_hub_2[0] - inlet_hub_1[0])
    dz = abs(inlet_hub_2[2] - inlet_hub_1[2])
    beta_b1_hub_1 = -np.atan2(dz, dx)

    # SS
    idx_1 = 261
    idx_2 = idx_1+1
    inlet_hub_3 = [x[idx_1], y[idx_1], z[idx_1]]
    inlet_hub_4 = [x[idx_2], y[idx_2], z[idx_2]]
    dx = abs(inlet_hub_4[0] - inlet_hub_3[0])
    dz = abs(inlet_hub_4[2] - inlet_hub_3[2])
    beta_b1_hub_2 = -np.atan2(dz, dx)
   

    # Calculate the average value of the two faces
    beta_b1_hub = (beta_b1_hub_1 + beta_b1_hub_2)/2



    r_hub_1 = (inlet_hub[1]**2 + inlet_hub[2]**2)**0.5
    r_hub_2 = (outlet_hub[1]**2 + outlet_hub[2]**2)**0.5
    

    
    suction_side = np.arange(0, 249)
    pressure_side = np.arange(262, 512)[::-1]

    thickness_hub = []
    for i, suction_side_idx in enumerate(suction_side):
        pressure_side_idx = pressure_side[i]
        suction_side_point = [x[suction_side_idx], y[suction_side_idx], z[suction_side_idx]]
        pressure_side_point = [x[pressure_side_idx], y[pressure_side_idx], z[pressure_side_idx]]
        local_thickness = ((suction_side_point[0]-pressure_side_point[0])**2 
                            +(suction_side_point[1]-pressure_side_point[1])**2 
                            +(suction_side_point[2]-pressure_side_point[2])**2 )**0.5
        thickness_hub.append(local_thickness)
    thickness_hub = np.average(thickness_hub)
        
        

    # Tip profile
    x = x_both[-512:]
    y = y_both[-512:]
    z = z_both[-512:]
    
    x_max_idx, x_min_idx = np.argmax(x), np.argmin(x)
    inlet_tip = [x[x_min_idx], y[x_min_idx], z[x_min_idx]]
    outlet_tip = [x[x_max_idx], y[x_max_idx], z[x_max_idx]]                    


    # The lower face
    idx_1 = 249
    idx_2 = idx_1+1
    inlet_tip_1 = [x[idx_1], y[idx_1], z[idx_1]]
    inlet_tip_2 = [x[idx_2], y[idx_2], z[idx_2]]
    dx = abs(inlet_tip_2[0] - inlet_tip_1[0])
    dz = abs(inlet_tip_2[2] - inlet_tip_1[2])
    beta_b1_tip_1 = -np.atan2(dz, dx)
    

    # The upper face
    idx_1 = 261
    idx_2 = idx_1+1
    inlet_tip_3 = [x[idx_1], y[idx_1], z[idx_1]]
    inlet_tip_4 = [x[idx_2], y[idx_2], z[idx_2]]
    dx = abs(inlet_tip_4[0] - inlet_tip_3[0])
    dz = abs(inlet_tip_4[2] - inlet_tip_3[2])
    beta_b1_tip_2 = -np.atan2(dz, dx)
    

    # Calculate the average value of the two faces
    beta_b1_tip = (beta_b1_tip_1 + beta_b1_tip_2)/2
    
    r_tip_1 = (inlet_tip[1]**2 + inlet_tip[2]**2)**0.5
    r_tip_2 = (outlet_tip[1]**2 + outlet_tip[2]**2)**0.5


    thickness_tip = []
    
    for i, suction_side_idx in enumerate(suction_side):
        pressure_side_idx = pressure_side[i]
        suction_side_point = [x[suction_side_idx], y[suction_side_idx], z[suction_side_idx]]
        pressure_side_point = [x[pressure_side_idx], y[pressure_side_idx], z[pressure_side_idx]]
        local_thickness = ((suction_side_point[0]-pressure_side_point[0])**2 + (suction_side_point[1]-pressure_side_point[1])**2 + (suction_side_point[2]-pressure_side_point[2])**2 )**0.5
        thickness_tip.append(local_thickness)
    thickness_tip = np.average(thickness_tip)



    L_z = (outlet_hub[0]-inlet_hub[0])/1000
    r_tip_1 = r_tip_1/1000
    r_hub_1 = r_hub_1 / 1000
    R_mean_2 = np.average([r_hub_2, r_tip_2])/1000
    b_2 = (outlet_hub[0]-outlet_tip[0])/1000
    
    thickness = np.average([thickness_hub, thickness_tip])
    thickness = thickness*1.057 / 1000
    L_z = L_z * 0.995
    
    # Derived values
    R_mean_1 = (2/3) * (((r_tip_1**3) - (r_hub_1**3)) / ((r_tip_1**2) - (r_hub_1**2)))
    beta_b1_mean = 0.4*beta_b1_hub + 0.6*beta_b1_tip
    s = 0.027 * b_2
    if s < 0.0003:
        s = 0.0003

    beta_b2 = -0.7854
    R_3 = 1.5 * R_mean_2
    b_3 = 0.8 * b_2


    geometry  = {
                'imp_type': 'Centrifugal', 
                'P_01': 101325, 
                'T_01': 288,
                'R_tip_1': r_tip_1, 
                'R_mean_1': R_mean_1, 
                'R_hub_1': r_hub_1,
                'alpha_1': 0, 
                'beta_b1_hub': beta_b1_hub, 
                'beta_b1_tip': beta_b1_tip,
                'beta_b1_mean': beta_b1_mean,
                'lambda_1': 1.0,
                'beta_b2': beta_b2,
                'R_mean_2': R_mean_2, 
                'lambda_2': 1.0, 
                'b_2': b_2, 
                'L_z': L_z, 
                't': thickness, 
                's': s,
                'nblades': int(number_of_blades),
                'n_splitter_blades': 0, # there is no splitter blades
                'b3': b_3, 
                'r3': R_3,
                'slip_factor': 0.8223}


    return geometry



def smoothening_3D(x,y,z, poly, window):
    
    x_smooth_list = []
    y_smooth_list = []
    z_smooth_list = []
    window = window      # must be odd; increase for more smoothing
    poly   = poly       # polynomial order

    points_per_profile = 512
    number_of_profiles = int(len(x) / points_per_profile)

    for profile_idx in range(number_of_profiles):
        
        ps_start_index = 0 + 512*profile_idx
        ps_end_index = 251 + 512*profile_idx
        ss_start_index = ps_end_index + 11
        ss_end_index = 511 + 512*profile_idx

        x_ps_fit = savgol_filter(x[ps_start_index:ps_end_index], window, poly)
        y_ps_fit = savgol_filter(y[ps_start_index:ps_end_index], window, poly)
        z_ps_fit = savgol_filter(z[ps_start_index:ps_end_index], window, poly)
        
        x_ss_fit = savgol_filter(x[ss_start_index:ss_end_index], window, poly)
        y_ss_fit = savgol_filter(y[ss_start_index:ss_end_index], window, poly)
        z_ss_fit = savgol_filter(z[ss_start_index:ss_end_index], window, poly)
        
        x_le = savgol_filter(x[ps_end_index:ss_start_index], 5, 3)
        y_le = savgol_filter(y[ps_end_index:ss_start_index], 5, 3)
        z_le = savgol_filter(z[ps_end_index:ss_start_index], 5, 3)
        

        x_smooth = np.concatenate([x_ps_fit, x_le, x_ss_fit])
        y_smooth = np.concatenate([y_ps_fit, y_le, y_ss_fit])
        z_smooth = np.concatenate([z_ps_fit, z_le, z_ss_fit])
        r_smooth = (y_smooth**2 + z_smooth**2)**0.5
        x_smooth_list.append(x_smooth)
        y_smooth_list.append(y_smooth)
        z_smooth_list.append(z_smooth)

    return x_smooth_list, y_smooth_list, z_smooth_list







def validation_3D(device, sample_percent, model, aux_model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, 
                  pca, data_structure, model_code, number_of_profiles = None, number_of_points = None):


    smoothening = False

    model.eval()
    aux_model.eval()

    df, min_max, val_indices = load_1D_dataset()


    df_2 = pd.read_csv('dataset/New_compressor_geometry/1D_compressor_geometry_filtered.csv')
    df_3 = pd.read_csv('dataset/New_compressor_geometry/polar_minmax_per_compressor_normalised.csv')
    df_4 = pd.read_csv('dataset/New_compressor_geometry/polar_minmax_secondary_normalisation.csv')


    r_min_min = df_4['r_min_min'].iloc[0]
    r_min_max = df_4['r_min_max'].iloc[0]
    r_max_min = df_4['r_max_min'].iloc[0]
    r_max_max = df_4['r_max_max'].iloc[0]
    theta_min_min = df_4['theta_min_min'].iloc[0]
    theta_min_max = df_4['theta_min_max'].iloc[0]
    theta_max_min = df_4['theta_max_min'].iloc[0]
    theta_max_max = df_4['theta_max_max'].iloc[0]
    x_min_min = df_4['x_min_min'].iloc[0]
    x_min_max = df_4['x_min_max'].iloc[0]
    x_max_min = df_4['x_max_min'].iloc[0]
    x_max_max = df_4['x_max_max'].iloc[0]

    pr_tolerance = CL_tolerence
    eta_tolerance = CD_tolerence


    random.seed(manual_seed)
    numbers = random.sample(range(0, len(val_indices)), int(len(val_indices)*sample_percent))
    print(len(val_indices))
    print('exe', manual_seed)

    sample_number = 1

    output_file = f'mdl_validation/{model_code}.csv'
    print(f'The validation output will be stored at {output_file}')

    if os.path.exists(output_file):
        print(f"Existing file '{output_file}', a new validation csv with suffix v2 will be created.")
        output_file = f'mdl_validation/{model_code}_v2.csv'
    results_df = pd.DataFrame()


    for idx in tqdm(numbers):
        i = val_indices[idx]


        pr_normalised = df.loc[i, 'imp_pressure_ratio']
        eta_normalised = df.loc[i, 'imp_efficiency']
        omega_normalised = df.loc[i, 'omega']
        m_dot_normalised = df.loc[i, 'm_dot']

        compressor_index = df_2.loc[i, 'geometry_index']
        number_of_unfeasible_designs = 0

        try: 
            row = df_3.loc[df_3['compressor_index'].astype(int) == int(compressor_index)]

            r_min = row.iloc[0]['r_min']
            r_max = row.iloc[0]['r_max']
            theta_min = row.iloc[0]['theta_min']
            theta_max = row.iloc[0]['theta_max']
            xc_min = row.iloc[0]['x_min']
            xc_max = row.iloc[0]['x_max']

                
            design = 0
            
            multiple_design_geometry = []
            multiple_design_blade_number = []
            multiple_design_1D_geometry = []
            while design < multiple_design:

                number_of_trials = 0
                success = False

                geometry_list = []
                geometry_1D_list = []
                blade_number_list = []
                pr_error_list = []
                eta_error_list = []
                pr_list = []
                eta_list = []

                while not success and number_of_trials < max_iteration:

                    # The first model, input 4 output 8
                    cond = (torch.tensor([m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised]).to(device))
                    # print(m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised)
                    rnd = StackedRandomGenerator(device, range(sample_number))
                    latents = rnd.randn([sample_number, aux_model.in_dim], device=device)
                    with torch.no_grad():
                        samples, _ = edm_sampler(aux_model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
                    
                    samples = samples.float()
                    sample = samples[0].cpu().numpy()
                    xc_min = sample[0]
                    xc_max = sample[1]
                    r_min = sample[2]
                    r_max = sample[3]
                    theta_min = sample[4]
                    theta_max = sample[5]
                    n_blades = sample[6]
                    n_splitter =  0

                    
                    # The main model
                    cond = (torch.tensor([m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised, xc_min, xc_max, r_min, r_max, theta_min, theta_max, n_blades, n_splitter]).to(device))
                    # print(m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised, xc_min, xc_max, r_min, r_max, theta_min, theta_max, n_blades, n_splitter)
                    rnd = StackedRandomGenerator(device, range(sample_number))
                    
                    if data_structure == '3D_PCA':
                        latents = rnd.randn([sample_number, model.in_dim], device=device)
                        with torch.no_grad():
                            samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
                        samples = samples.float()
                        sample = samples[0].cpu().numpy()
                        sample = pca.inverse_transform(sample)
                    

                    if data_structure == '3D_coordinates':
                        latents = torch.randn(1, 1, number_of_profiles, number_of_points*3, device=device)
                        with torch.no_grad():
                            samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
                        samples = samples.float()
                        sample = samples[0].cpu().numpy()
                        sample = sample.ravel()
                    

                    r_normalised = sample[1::3]
                    theta_normalised = sample[2::3]
                    x_normalised = sample[0::3]
                    

                    r_min_denormalised = r_min * (r_min_max - r_min_min) + r_min_min
                    r_max_denormalised = r_max * (r_max_max - r_max_min) + r_max_min
                    theta_min_denormalised = theta_min * (theta_min_max - theta_min_min) + theta_min_min
                    theta_max_denormalised = theta_max * (theta_max_max - theta_max_min) + theta_max_min
                    x_min_denormalised = xc_min * (x_min_max - x_min_min) + x_min_min
                    x_max_denormalised = xc_max * (x_max_max - x_max_min) + x_max_min


                    r = r_normalised * (r_max_denormalised - r_min_denormalised) + r_min_denormalised 
                    theta = theta_normalised * (theta_max_denormalised - theta_min_denormalised) + theta_min_denormalised 
                    x = x_normalised * (x_max_denormalised - x_min_denormalised) + x_min_denormalised 
    
                    x, y, z = cyl_to_cart_about_x(x, r, theta)
                    

                    ###############################################
                    # Temp fix for reversed leadign edge
                    n_profiles = 16
                    n_points = 512

                    x_2d = np.asarray(x).reshape(n_profiles, n_points)
                    y_2d = np.asarray(y).reshape(n_profiles, n_points)
                    z_2d = np.asarray(z).reshape(n_profiles, n_points)

                    # Points 251–264 inclusive, assuming zero-based indexing
                    x_2d[:, 251:261] = x_2d[:, 251:261][:, ::-1]
                    y_2d[:, 251:261] = y_2d[:, 251:261][:, ::-1]
                    z_2d[:, 251:261] = z_2d[:, 251:261][:, ::-1]

                    # Flatten back to the original form
                    x = x_2d.ravel()
                    y = y_2d.ravel()
                    z = z_2d.ravel()
                    #################################################





                    if smoothening:
                        x, y, z = smoothening_3D(x,y,z,3,11)


                    if round(n_blades) == 0:
                        number_of_blades =7
                    else:
                        number_of_blades =8

                    geometry_1D = geometry_3D_to_1D_conversion(x,y,z, number_of_blades)
                        
                        


                    m_dot = m_dot_normalised*(min_max.loc['m_dot', 'max']  - min_max.loc['m_dot', 'min']) + min_max.loc['m_dot', 'min']
                    omega = omega_normalised*(min_max.loc['omega', 'max']  - min_max.loc['omega', 'min']) + min_max.loc['omega', 'min']
                    pr_original = pr_normalised*(min_max.loc['imp_pressure_ratio', 'max']  - min_max.loc['imp_pressure_ratio', 'min']) + min_max.loc['imp_pressure_ratio', 'min']
                    eta_original = eta_normalised*(min_max.loc['imp_efficiency', 'max']  - min_max.loc['imp_efficiency', 'min']) + min_max.loc['imp_efficiency', 'min']

                    
                    RPM = (omega*60)/(2*np.pi)

                    pr, eta = run_meanline(geometry_1D, m_dot, omega, x_foil_timeout)

                    pr_error = 100*abs(pr - pr_original) / (pr_original)
                    eta_error = 100*abs(eta - eta_original) / (eta_original)
                    
                    pr_error_list.append(pr_error)
                    eta_error_list.append(eta_error)
                    pr_list.append(pr)
                    eta_list.append(eta)
                    geometry = [x,y,z]
                    geometry_list.append(geometry)
                    geometry_1D_list.append(geometry_1D)
                    blade_number_list.append(number_of_blades)

                    number_of_trials += 1
                    # print(number_of_trials)
                    # print(f'Pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                    # print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')
                    if pr_error < pr_tolerance and eta_error < eta_tolerance:
                        # print(f'Number {design+1} design took {number_of_trials} trials.')
                        # print(f'Pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                        # print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')
                        success = True
                    elif pr == 999 or eta == 999:
                        number_of_unfeasible_designs += 1
                design += 1

                if number_of_trials == max_iteration:

                    index = pr_error_list.index(min(pr_error_list))

                    geometry = geometry_list[index]
                    geometry_1D = geometry_1D_list[index]
                    number_of_blades = blade_number_list[index]
                    pr = pr_list[index]
                    eta = eta_list[index]
                    pr_error = pr_error_list[index]
                    eta_error = eta_error_list[index]
                    print(f'Number {design} design cannot satisfy the tolerance after {number_of_trials} trails.')
                    print(f'Using the best design: pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                    print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')

                
                multiple_design_geometry.append(geometry)
                multiple_design_1D_geometry.append(geometry_1D)
                multiple_design_blade_number.append(number_of_blades)

                
                # print(f'Number {design} design has blade number of {number_of_blades}.')

            new_row = {"omega": omega, "m_dot": m_dot, "pr_original": pr_original, "eta_original": eta_original, "pr_actual": pr, "eta_actual": eta, "design_iteration": number_of_trials, "unfeasible_design": number_of_unfeasible_designs}
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            results_df.to_csv(output_file, index=False)
        
        
        
        except Exception as e:
            print(e)
            pass












def model_deployment( model_config_path, sample_percent, aux_model_config_path = None, num_steps = None,  manual_seed = None,
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
    data_reduction_fraction = model_config['reduced_data_fraction']
    model_code = f"{data_structure}_{nn_structure}_{model_channel}_{model_layer}_{len(model_channel_multiplication)}_with_{num_epochs}_epochs_{data_reduction_fraction}_data"
    save_path = f"mdl_weight/{model_code}.pth"
    print('Model Code', model_code)
    num_components=model_config['component_number']
    


    # ============== Auxiliary model ===============
    if aux_model_config_path != None:
        with open(aux_model_config_path, "r") as f:
            aux_model_config = yaml.safe_load(f)

        aux_data_structure = aux_model_config['data_structure']
        aux_num_epochs = aux_model_config['num_epochs']
        aux_model_channel = aux_model_config['model_channel']
        aux_model_layer = aux_model_config['model_layer']
        aux_model_channel_multiplication = aux_model_config['model_channel_multiplication']
        aux_device = aux_model_config['device']
        aux_nn_structure = aux_model_config['neural_network_sturcture']
        aux_reduced_data_fraction = aux_model_config['reduced_data_fraction']
        aux_model_code = f"{aux_data_structure}_{aux_nn_structure}_{aux_model_channel}_{aux_model_layer}_{len(aux_model_channel_multiplication)}_with_{aux_num_epochs}_epochs_{aux_reduced_data_fraction}_data"
        aux_save_path = f"mdl_weight/{aux_model_code}.pth"
        print('Auxiliary Model Code', aux_model_code)
        aux_num_components = aux_model_config['component_number']


        aux_model = EDM_CFG(aux_num_components, aux_num_components, cond_size=4, model_channel=aux_model_channel,
        channel_multiply=aux_model_channel_multiplication, dim_mult_emb=4, num_blocks=aux_model_layer,
        dropout=0, emb_type="sinusoidal", dim_mult_time=1, nn_structure=aux_nn_structure,
        dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=False, data_structure=aux_data_structure, 
        number_of_pc = aux_num_components)
        aux_model.load_state_dict(torch.load(aux_save_path))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aux_model = aux_model.to(device)






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


    elif data_structure == '3D_PCA':
        
        curve_file = 'dataset/3D_compressor_polar_normalised'
        coordinates = []

        for i in range(1512):
            try:
                profile, profile_has_error = load_blade_curve(f'{curve_file}/compressor_{i}.curve')
                if not profile_has_error:
                    coordinate = np.concatenate(profile).ravel()
                coordinates.append(coordinate)
                
            except FileNotFoundError:
                pass
        
        coordinates = np.array(coordinates)
        pca = PCA(n_components=num_components)
        pca_scores = pca.fit_transform(coordinates)
        

        model = EDM_CFG(num_components, num_components, cond_size=12, model_channel=model_channel,
                channel_multiply=model_channel_multiplication, dim_mult_emb=4, num_blocks=model_layer,
                dropout=0, emb_type="sinusoidal", dim_mult_time=1, nn_structure=nn_structure,
                dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=False, data_structure=data_structure, 
                    number_of_pc = num_components)
        model.load_state_dict(torch.load(save_path))


        aux_model = EDM_CFG(8, 8, cond_size=4, model_channel=model_channel,
                channel_multiply=[1,2,4], dim_mult_emb=4, num_blocks=model_layer,
                dropout=0, emb_type="sinusoidal", dim_mult_time=1, nn_structure=nn_structure,
                dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=False, data_structure='3D_aux', 
                    number_of_pc = 8)
        aux_model.load_state_dict(torch.load('mdl_weight/3D_aux_ResNet_UNet_64_5_3_with_100_epochs_1_data.pth'))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aux_model = aux_model.to(device)


    elif data_structure == '3D_coordinates':

        number_of_profiles = model_config['number_of_profiles']
        number_of_points = model_config['number_of_points']
        model = EDM_CFG(number_of_profiles, number_of_points, cond_size=12, model_channel=model_channel,
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

    
        
    if data_structure == '1D_params':
        validation_1D(device, sample_percent, model, num_steps, manual_seed, 1, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, model_code)

    elif data_structure == '3D_PCA':
        validation_3D(device, sample_percent, model, aux_model, num_steps, manual_seed, 1, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,
                       pca, data_structure, model_code)
    elif data_structure == '3D_coordinates':
        validation_3D(device, sample_percent, model, aux_model, num_steps, manual_seed, 1, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,
                       pca, data_structure, model_code, number_of_profiles, number_of_points)
    else:
        model_validation(model, model_code, data_structure, device, sample_percent, pca = pca, num_steps = num_steps, 
                        manual_seed = manual_seed, x_foil_timeout = x_foil_timeout, CL_tolerence = CL_tolerence, 
                        CD_tolerence = CD_tolerence, max_iteration = max_iteration, mode = mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion model with YAML config.")
    parser.add_argument("main_model_config", type=str, help="Path to the YAML config file")
    parser.add_argument("aux_model_config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    config_file_main = args.main_model_config
    config_file_aux = args.aux_model_config
    model_deployment(config_file_main, sample_percent = 0.1, aux_model_config_path=config_file_aux, num_steps = 30,  manual_seed = 123,
                     x_foil_timeout=5, CL_tolerence = 1, CD_tolerence = 1, max_iteration = 100, mode = 'intermediate')