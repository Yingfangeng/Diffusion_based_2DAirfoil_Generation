import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
import pandas as pd
import subprocess as sp
import re
import random
import yaml
from scipy.stats import gaussian_kde
import imageio.v2 as imageio
import os
from IPython.display import Image
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tqdm import tqdm
import time
from meanline.meanline import *
import signal
import contextlib
from contextlib import redirect_stdout
from scipy.signal import savgol_filter
import logging
from torchmetrics.functional.image.ssim import structural_similarity_index_measure


from models.diffusion_model import EDM_CFG, edm_sampler, StackedRandomGenerator
from meanline.impeller import Blade_Forming_3D

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
text_font = 'Liberation Sans'
math_font = 'stix'

plt.rcParams['mathtext.fontset'] = math_font
rcParams['font.family'] = text_font




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


def save_fig_custom(fig, file_path='', file_name='fig', 
                    format_list=['.eps', '.png'], overwrite=False, dpi = 500):
    
    if (file_path != '') and (file_path[-1] != '/'): 
        file_path = file_path+'/'
        
    if (file_path != '') and (os.path.isdir(file_path) == False):
        os.makedirs(file_path)
        print('Save directory %s is created'%(file_path))
    
    for save_format in format_list:
        if save_format[0] != '.': save_format = '.' + save_format

        file_name_now = file_path+file_name+save_format
        if not overwrite:
            i = 1
            while os.path.exists(file_name_now):
                file_name_now = file_path+file_name + '%i'%(i)+save_format
                i = i+1
        
        if save_format == '.png':
            fig.savefig(file_name_now, facecolor='white', bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(file_name_now, facecolor='white', bbox_inches="tight")

        
        print(file_name_now, 'is saved.')
    
    return


def string_coords_to_list(x_string, y_string):

    x_string = re.sub(r'[\[\]]', '', x_string)   # remove brackets
    x_list = [float(v) for v in x_string.split()]
    y_string = re.sub(r'[\[\]]', '', y_string)
    y_list = np.array([float(v) for v in y_string.split()])

    return x_list, y_list


def xfoil_calculation(x,y,AOA,Re,Ma,CL,CD, x_foil_timeout):
    
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
        print('There is no file written from xfoil!!!')
        CL_actual = 999
        CD_actual = 999
    return CL_actual, CD_actual


def load_data_set(data_structure):

    df = pd.read_csv("dataset/aerofoil_data_clean_normalised.csv")
    df2 = pd.read_csv("dataset/aerofoil_data_clean.csv")
    val_indices = np.load("dataset/val_indices_clean.npy")
    train_indices = np.load("dataset/train_indices_clean.npy")
    min_max = pd.read_csv('dataset/min_max_clean.csv')

    return df, df2, val_indices, train_indices, min_max


def random_pick_validation(manual_seed, val_indices, sample_size):
    if manual_seed != None:
        random.seed(manual_seed)
        print('exe', manual_seed)
    numbers = random.sample(range(0, len(val_indices)), sample_size)
    return numbers


def load_validation_condition(idx, data_structure, device):

    df, df2, val_indices, train_indices, min_max = load_data_set(data_structure)

    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]
    

    i = val_indices[idx]
    x_str = df.loc[i, 'x']
    y_str = df.loc[i, 'y']
    
    x_coords_original, y_coords_original = string_coords_to_list(x_str, y_str)
    y_coords_original = y_coords_original * (y_max - y_min) + y_min


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
    
    return name, x_coords_original, y_coords_original, Ma, Re, AOA, CL, CD, AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised


def model_generation(model, mode, data_structure,  Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original = None, y_coords_original = None, pca = None, num_steps = None, 
                    multiple_design = 1, x_res = 256, y_res = 256, x_foil_timeout = 20, CL_tolerence = 0.01, 
                    CD_tolerence = 0.05, max_iteration = 100, distribution_plot = False):
    
    sample_size = 1
    # the exact of CL, CD within the specific range
    CL_actual_list = []
    CD_actual_list = []
    x_coords_list = []
    y_coords_list = []
    sdf_list = []

    valid_design = 0
    iteration_count = 0
    max_iteration_count = max_iteration
    print(f"{num_steps} sampling steps, {multiple_design} design(s)")

    
    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]
    sdf_min = min_max['sdf_min'].loc[0]
    sdf_max = min_max['sdf_max'].loc[0]

    while valid_design < multiple_design:
        iteration_count  = 0
        next = 0
        CL_error_list = []
        CD_error_list = []
        CL_actual_list_2 = []
        CD_actual_list_2 = []
        x_coords_list_2 = []
        y_coords_list_2 = []
        sdf_list_2 = []
        while iteration_count < max_iteration_count and next == 0:
            
            if data_structure != 'sdf':
                if mode == 'optimisation':
                    clean_sample = np.column_stack((x_coords_original, y_coords_original)).flatten()
                    clean_sample = torch.tensor(clean_sample, device=device).float()
                    sigma0 = 0.0
                    noisy_sample = clean_sample + torch.randn_like(clean_sample) * sigma0
                    latents = noisy_sample.unsqueeze(0).repeat(sample_size, 1)

                else:
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
                x_resolution = x_res
                y_resolution = y_res
                x_lower_lim = -0.01
                x_upper_lim = 1.01
                y_lower_lim = -0.17
                y_upper_lim = 0.27
                rnd = StackedRandomGenerator(device, range(sample_size))
                latents = torch.randn(sample_size, 1, y_resolution, x_resolution, device=device)
                with torch.no_grad():
                    samples, _ = edm_sampler(model, latents=latents, class_labels=cond, randn_like=rnd.randn_like, num_steps=num_steps, deterministic=False) 
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
            CL_error = abs((CL_actual - CL)/CL)
            CD_error = abs((CD_actual - CD)/CD)
            

            if CL_error <= CL_tolerence and CD_error <= CD_tolerence:
                x_coords_list.append(x_coords)
                y_coords_list.append(y_coords)
                CL_actual_list.append(CL_actual)
                CD_actual_list.append(CD_actual)
                if not distribution_plot:
                    next = 1
                    valid_design = valid_design + 1
                x_coords_list_2.append(x_coords)
                y_coords_list_2.append(y_coords)
                CL_actual_list_2.append(CL_actual)
                CD_actual_list_2.append(CD_actual)
                if data_structure == 'sdf':
                    sdf_list.append(sdf)
                    sdf_list_2.append(sdf)
            # elif CL_actual == 999:
            #     print('The design is not feasible, regenerating')
            
    
            else:
                x_coords_list_2.append(x_coords)
                y_coords_list_2.append(y_coords)
                CL_error_list.append(CL_error)
                CD_error_list.append(CD_error)
                CL_actual_list_2.append(CL_actual)
                CD_actual_list_2.append(CD_actual)
                if data_structure == 'sdf':
                    sdf_list_2.append(sdf)

            iteration_count = iteration_count + 1
        
        
        if iteration_count == max_iteration_count and next == 0:
            
            index = CL_error_list.index(min(CL_error_list))
            x_coords_list.append(x_coords_list_2[index])
            y_coords_list.append(y_coords_list_2[index])
            if data_structure == 'sdf':
                sdf_list.append(sdf_list_2[index])
            CL_actual_list.append(CL_actual_list_2[index])
            CD_actual_list.append(CD_actual_list_2[index])

            valid_design = valid_design + 1
            print(f'Max {max_iteration_count} design iteration reached for NO.{valid_design} design.\n'
                    f'Using the best matching result with CL {CL_actual_list_2[index]} with error {100*CL_error_list[index]}%, CD {CD_actual_list_2[index]} with error {100*CD_error_list[index]}%.')

        else:
            print(f'The NO.{valid_design} valid design took {iteration_count} design iteration(s)\n'
                f'The design has CL {CL_actual} ({100*CL_error}%) and CD {CD_actual} ({100*CD_error}%).')

    if data_structure == 'sdf':
        return x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2, sdf_list, sdf_list_2

    else:
        return x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2


def plot_off_design(x_coords_list, y_coords_list, Re, Ma, AOA, CL, CD, x_foil_timeout):
    
    design_point_AOA = AOA
    CL_list_2 = []
    AOA_list_2 = []
    CD_list_2 = []
    AOA_start = 0
    AOA_end = 20
    AOA_off_design = np.arange(AOA_start, AOA_end+1, 1)
    
    for idx, _ in enumerate(x_coords_list):
        AOA_list = []
        CL_list = []
        CD_list = []
        for AOA in AOA_off_design:
            CL_actual, CD_actual = xfoil_calculation(x_coords_list[idx], y_coords_list[idx], AOA, Re, Ma, CL, CD, x_foil_timeout)
            if CL_actual != 999:
                CL_list.append(CL_actual)
                CD_list.append(CD_actual)
                AOA_list.append(AOA)
        CL_list_2.append(CL_list)
        CD_list_2.append(CD_list)
        AOA_list_2.append(AOA_list)
    
    fig, ax1 = plt.subplots()
    for idx, _ in enumerate(CL_list_2):
        ax1.plot(AOA_list_2[idx], CL_list_2[idx], zorder = 1)
    ax1.scatter(design_point_AOA, CL, marker='*', color = 'r', s = 60, label = 'Design Target', zorder = 5)

    ax1.set_xlabel('Angle of Attack (deg)', fontsize = 14)
    ax1.set_ylabel('Lift Coefficient ($C_L$)', fontsize = 14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, ls=':')
    ax1.legend(loc = 'upper left', fontsize = 15)
    save_fig_custom(fig, file_path='fig', file_name=f'multi_target_CL_off_design', 
                    format_list=['.eps', '.png'], overwrite=True, dpi = 500)

    fig, ax2 = plt.subplots()
    
    for idx, _ in enumerate(CD_list_2):
        ax2.plot(AOA_list_2[idx], CD_list_2[idx], zorder = 1)
    ax2.scatter(design_point_AOA, CD, marker='*', color = 'r', s = 60, label = 'Design Target', zorder = 5)
    ax2.set_xlabel('Angle of Attack (deg)', fontsize = 14)
    ax2.set_ylabel('Drag Coefficient ($C_D$)', fontsize = 14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True, ls=':')
    ax2.legend(loc = 'upper left', fontsize = 15)
    save_fig_custom(fig, file_path='fig', file_name=f'multi_target_CD_off_design', 
                    format_list=['.eps', '.png'], overwrite=True, dpi = 500)




def find_same_condition_in_dataset(AOA, Re, Ma, CL, CD, df2, val_indices, train_indices):
    same_intended_cond_val_idx = []
    same_intended_cond_train_idx = []
    # same_actual_cond_val_idx = []
    # same_actual_cond_train_idx = []
    for idx in range(len(df2)):
        if int(df2['AOA'].iloc[idx]) == int(AOA):
            if int(df2['Re'].iloc[idx]) == int(Re):
                if round(df2['Ma'].iloc[idx], 1) == round(Ma, 1):
                    if abs((df2['CD'].iloc[idx] - CD)/df2['CD'].iloc[idx]) <=0.01:
                        if abs((df2['CL'].iloc[idx] - CL)/df2['CL'].iloc[idx]) <=0.01:
                            if idx in val_indices:
                                same_intended_cond_val_idx.append(idx)
                                print('Same intended conditions in validation set', df2['name'].iloc[idx])
                            elif idx in train_indices:
                                same_intended_cond_train_idx.append(idx)
                                print('Same intended conditions in training set', df2['name'].iloc[idx])
                    # elif abs((df2['CD'].iloc[idx] - CD_actual)/df2['CD'].iloc[idx]) <=0.01:
                    #     if abs((df2['CL'].iloc[idx] - CL_actual)/df2['CL'].iloc[idx]) <=0.01:
                    #         if idx in val_indices:
                    #             same_actual_cond_val_idx.append(idx)
                    #             print('Same actual conditions in validation set', df2['name'].iloc[idx], 'in the validation set.')
                    #         elif idx in train_indices:
                    #             same_actual_cond_train_idx.append(idx)
                    #             print('Same actual conditions in training set', df2['name'].iloc[idx], 'in the training set.')
    return same_intended_cond_val_idx, same_intended_cond_train_idx #, same_actual_cond_val_idx, same_actual_cond_train_idx



def distribution_graph(CL_actual_list, CD_actual_list, CL, CD):
    fig, ax2 = plt.subplots()
    ax2.scatter(CD_actual_list, CL_actual_list, s = 2, c = 'b', label = 'generated')
    ax2.scatter([CD], [CL], s=7, marker='x', c = 'r', label = 'intended')
    ax2.set_xlim(0, 0.05)
    ax2.set_ylim(0, 2)
    ax2.set_xlabel('CD')
    ax2.set_ylabel('CL')
    ax2.set_title(f'Intended CL {CL:.2f}, intended CD {CD:.2f}')
    ax2.legend()
    plt.show()



def plotting_function(data_structure, x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, Ma, Re, AOA, CL, CD, 
                      same_intended_cond_val_idx, same_intended_cond_train_idx, df, min_max, distribution_plot, off_design_plot, 
                      x_coords_list_2 = None, y_coords_list_2 = None, CL_actual_list_2 = None, CD_actual_list_2 = None, sdf_list = None, sdf_list_2 = None):
    
    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]

    if distribution_plot:
        x_coords_list = x_coords_list_2
        y_coords_list = y_coords_list_2
        CL_actual_list = CL_actual_list_2
        CD_actual_list = CD_actual_list_2


    fig, ax1 = plt.subplots()
    count = 0
    ax1.grid(True, ls=':')
    
    
    if data_structure == 'sdf':
        x_lower_lim = -0.01
        x_upper_lim = 1.01
        y_lower_lim = -0.17
        y_upper_lim = 0.27

        xs = np.linspace(x_lower_lim, x_upper_lim, 128)
        ys = np.linspace(y_lower_lim, y_upper_lim, 128)


        colors = [
            (0.0, "#ff0000"),  # blue-ish for negative values
            (0.5, "#FFFFFF"),    # exact zero
            (1.0, "#0113CD")   # red-ish for positive values
        ]

        my_cmap = LinearSegmentedColormap.from_list("my_diverging", colors)
        limit = 0.3
        factor1 = 0.35
        factor2 = 0.15
        cmap=my_cmap # change here

        norm = TwoSlopeNorm(vmin=-0.1*factor1, vcenter=0, vmax=0.3*factor2)

        for idx, _ in enumerate(x_coords_list):
            ax1.contourf(xs, ys, sdf_list[idx], levels=500, cmap=cmap, norm = norm)
            ax1.plot(x_coords_list[idx], y_coords_list[idx], color = 'k', linewidth = 3, ls = '--')
        ax1.set_ylim(-0.17, 0.2)


    else: 
        for idx, _ in enumerate(x_coords_list):
            if count == 0:
                ax1.plot(x_coords_list[idx],y_coords_list[idx])
            else:
                ax1.plot(x_coords_list[idx],y_coords_list[idx])
            # if data_structure =='raw_coordinates':
            #     ax1.scatter(x_coords_list[idx][::4],y_coords_list[idx][::4], color = 'b')
            count = count +1
        count1 = 0
        count2 = 0
        for i in same_intended_cond_val_idx:
            name = df.loc[i, 'name']
            x_str = df.loc[i, 'x']
            y_str = df.loc[i, 'y']
            x_coords_original, y_coords_original = string_coords_to_list(x_str, y_str)
            y_coords_original = y_coords_original * (y_max - y_min) + y_min
            # if count1 == 0:
            #     ax1.plot(x_coords_original,y_coords_original, c = 'r', ls = ':', label = f'validation set')
            # else:
            #     ax1.plot(x_coords_original,y_coords_original, c = 'r',ls = ':')
            count1 = count1 + 1

        for i in same_intended_cond_train_idx:
            name = df.loc[i, 'name']
            x_str = df.loc[i, 'x']
            y_str = df.loc[i, 'y']
            x_coords_original, y_coords_original = string_coords_to_list(x_str, y_str)
            y_coords_original = y_coords_original * (y_max - y_min) + y_min
            
            # if count2 == 0:
            #     ax1.plot(x_coords_original,y_coords_original, c = 'g', ls = ':', label = f'training set')
            # else:
            #     ax1.plot(x_coords_original,y_coords_original, c = 'g', ls = ':')
            count2 = count2 + 1
        ax1.set_xlim(-0.01, 1.01)
        ax1.set_ylim(-0.2,0.2)
        
    ax1.set_aspect('equal', 'box')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(bottom=False, top=False, left=False, right=False,
               labelbottom=False, labelleft=False)
    
    # ax1.set_title(
    #         f"Ma: {Ma:.1f} "
    #         f"Re: {int(Re)} "
    #         f"AOA: {int(AOA)} deg \n"
    #         f"CD: {CD:.2f} "
    #         f"CL: {CL:.2f} \n"
    #         , fontsize = 14)
    ax1.text(0.695, 0.04, f"Ma: {Ma:.1f} "
            f"Re: {int(Re)} "
            f"AOA: {int(AOA)} deg \n"
            f"$C_L$: {CL:.2f} "
            f"$C_D$: {CD:.2f}",
            transform=ax1.transAxes,
            fontsize=14,
            verticalalignment='bottom',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.65))


    # ax1.legend()
    plt.show()
    print(f"Actual CD: {min(CD_actual_list):.2f} - {max(CD_actual_list):.2f}"
          f"Actual CL: {min(CL_actual_list):.2f} - {max(CL_actual_list):.2f} ")
    save_fig_custom(fig, file_path='fig', file_name=f'demo_plot_{int(Re)}_{Ma:.2f}_{int(AOA)}_{CL:.2f}_{CD:.2f}_{data_structure}', 
                    format_list=['.eps', '.png'], overwrite=True, dpi = 500)

    if off_design_plot:
        plot_off_design(x_coords_list, y_coords_list, Re, Ma, AOA, CL, CD, 5)

    if distribution_plot:
        distribution_graph(CL_actual_list, CD_actual_list, CL, CD)




def test_plot(AOA, Ma, Re, CL, CD, data_structure, device):

    df, df2, val_indices, train_indices, min_max = load_data_set(data_structure)
    
    normalised_Ma = (Ma-min_max['Ma_min'].loc[0])/(min_max['Ma_max'].loc[0]-min_max['Ma_min'].loc[0])
    normalised_Re = (Re-min_max['Re_min'].loc[0])/(min_max['Re_max'].loc[0]-min_max['Re_min'].loc[0])
    normalised_AOA = (AOA-min_max['AOA_min'].loc[0])/(min_max['AOA_max'].loc[0]-min_max['AOA_min'].loc[0])
    normalised_CL = (CL-min_max['CL_min'].loc[0])/(min_max['CL_max'].loc[0]-min_max['CL_min'].loc[0])
    normalised_CD = (CD-min_max['CD_min'].loc[0])/(min_max['CD_max'].loc[0]-min_max['CD_min'].loc[0])

    cond = (torch.tensor([normalised_AOA, normalised_Ma, normalised_Re, normalised_CL, normalised_CD]).to(device))
    
    return Ma, Re, AOA, CL, CD, cond




def validation_plot(model, sample_size, data_structure, device, mode = 'validation', pca = None, num_steps = None, manual_seed = None, 
                    lim = 0.5, multiple_design = 1, x_res = 256, y_res = 256, x_foil_timeout = 20, CL_tolerence = 0.01, 
                    CD_tolerence = 0.05, max_iteration = 100, distribution_plot = False, off_design_plot = False, 
                    manual_Re = None, manual_Ma = None, manual_AOA = None, manual_CL = None, manual_CD = None):

    # load the dataset
    df, df2, val_indices, train_indices, min_max = load_data_set(data_structure)

    
    
    if mode != 'test':
        numbers = random_pick_validation(manual_seed, val_indices, sample_size)
        
        for index in numbers:
            name, x_coords_original, y_coords_original, Ma, Re, AOA, CL, CD, AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised = load_validation_condition(index, data_structure, device)
            print(f'Target: CL:{CL}, CD: {CD}')
            
            # randomly pick one validation results
            if mode == 'optimisation':
                fig, ax = plt.subplots()
                ax.grid(True, ls=':')
                ax.plot(x_coords_original,y_coords_original, color = 'r', label = 'original')
                ax.set_ylim(-0.4,0.4)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.legend()
                ax.set_title(f"{name}\n"
                            f"Ma: {Ma:.1f} "
                            f"Re: {int(Re)} "
                            f"AOA: {int(AOA)} deg \n"
                            f"CD: {CD:.2f} "
                            f"CL: {CL:.2f} \n")
                plt.show(block=True)

                tmp = input(f"Current design CL is {CL:.4f}. Press ENTER to keep it, or enter new target CL: ")
                if tmp.strip() == "":
                    CL_optimised = CL
                else:
                    CL_optimised = float(tmp)

                tmp = input(f"Current design CD is {CD:.4f}. Press ENTER to keep it, or enter new target CD: ")
                if tmp.strip() == "":
                    CD_optimised = CD
                else:
                    CD_optimised = float(tmp)
                
                CL_optimised_normalised = (CL_optimised - min_max['CL_min'].loc[0])/(min_max['CL_max'].loc[0] - min_max['CL_min'].loc[0])
                CD_optimised_normalised = (CD_optimised - min_max['CD_min'].loc[0])/(min_max['CD_max'].loc[0] - min_max['CD_min'].loc[0])
                cond = (torch.tensor([AOA_normalised, Ma_normalised, Re_normalised, CL_optimised_normalised, CD_optimised_normalised]).to(device))
                
                x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, _, _, _, _ = model_generation(model, mode,  data_structure, 
                Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original, y_coords_original, pca, num_steps, multiple_design, x_res, y_res, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, distribution_plot)
                
                x_coords_list_scratch, y_coords_list_scratch, CL_actual_list_scratch, CD_actual_list_scratch, _, _, _, _ = model_generation(model, 'validation',  data_structure, 
                Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original, y_coords_original, pca, num_steps, multiple_design, x_res, y_res, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, distribution_plot)

                fig, axes = plt.subplots(1,2)
                ax1 = axes[0]
                ax2 = axes[1]
                count = 0
                ax1.grid(True, ls=':')
                ax2.grid(True, ls=':')
                for idx, _ in enumerate(x_coords_list):
                    if count == 0:
                        ax1.plot(x_coords_list[idx],y_coords_list[idx], color = 'b', label = 'optimised')
                        ax1.plot(x_coords_original, y_coords_original, color = 'r', label = 'original')
                        ax2.plot(x_coords_list_scratch[idx],  y_coords_list_scratch[idx], color = 'y', label = 'scratch')
                        ax2.plot(x_coords_original, y_coords_original, color = 'r', label = 'original')
                    else:
                        ax1.plot(x_coords_list[idx],y_coords_list[idx], color = 'b')
                        ax1.plot(x_coords_original, y_coords_original, color = 'r')
                        ax2.plot(x_coords_list_scratch[idx],  y_coords_list_scratch[idx], color = 'y')
                        ax2.plot(x_coords_original, y_coords_original, color = 'r')
                    count = count +1

                ax1.set_ylim(-0.4,0.4)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.legend()

                
                ax2.set_ylim(-0.4,0.4)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.legend()
                fig.suptitle(
                            f"Ma: {Ma:.1f} "
                            f"Re: {int(Re)} "
                            f"AOA: {int(AOA)} deg \n"
                            f"Original CD: {CD:.2f} "
                            f"Original CL: {CL:.2f} \n"
                            f"Intended optimised CD: {CD_optimised:.2f} "
                            f"Intended optimised CL: {CL_optimised:.2f} \n"
                            f"Actual optimised CD: {min(CD_actual_list):.2f} - {max(CD_actual_list):.2f} "
                            f"Actual optimised CL: {min(CL_actual_list):.2f} - {max(CL_actual_list):.2f} \n"
                            f"From scratch CD: {min(CD_actual_list_scratch):.2f} - {max(CD_actual_list_scratch):.2f} "
                            f"From scratch CL: {min(CL_actual_list_scratch):.2f} - {max(CL_actual_list_scratch):.2f} \n"
                            )
                plt.tight_layout()
                plt.subplots_adjust(top=0.7)  
                plt.show()

                distribution_graph(CL_actual_list, CD_actual_list, CL_optimised, CD_optimised)
                distribution_graph(CL_actual_list_scratch, CD_actual_list_scratch, CL_optimised, CD_optimised)



            else:    
                cond = (torch.tensor([AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised]).to(device))
                
                if data_structure == 'sdf':
                    x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2, sdf_list, sdf_list_2 = model_generation(model, mode, data_structure, 
                    Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original, y_coords_original, pca, num_steps, multiple_design, x_res, y_res, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,  distribution_plot)
                else:
                    x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2 = model_generation(model, mode, data_structure, 
                    Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original, y_coords_original, pca, num_steps, multiple_design, x_res, y_res, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,  distribution_plot)
                    sdf_list = None
                    sdf_list_2 = None
                # same intended condition in training set
                same_intended_cond_val_idx, same_intended_cond_train_idx = find_same_condition_in_dataset(AOA, Re, Ma, CL, CD, df2, val_indices, train_indices)
                


                plotting_function(data_structure, x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, Ma, Re, AOA, CL, CD, 
                            same_intended_cond_val_idx, same_intended_cond_train_idx, df, min_max, distribution_plot, off_design_plot, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2, sdf_list, sdf_list_2)

    else:
        Ma, Re, AOA, CL, CD, cond = test_plot(manual_AOA, manual_Ma, manual_Re, manual_CL, manual_CD, data_structure, device)
       
        if data_structure == 'sdf':
            x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2, sdf_list, sdf_list_2 = model_generation(model, mode, data_structure, 
            Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original = None, y_coords_original = None, pca = pca, num_steps = num_steps, multiple_design = multiple_design, x_res = x_res, y_res = y_res, x_foil_timeout = x_foil_timeout, CL_tolerence = CL_tolerence, CD_tolerence = CD_tolerence, max_iteration = max_iteration,  distribution_plot = distribution_plot)
        
        else:

            x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2 = model_generation(model, 'validation',  data_structure,
            Ma, Re, AOA, CL, CD, cond, device, min_max, x_coords_original=None, y_coords_original=None, pca = pca, num_steps = num_steps, 
            multiple_design=multiple_design, x_res=x_res, y_res=x_res, x_foil_timeout=x_foil_timeout, CL_tolerence=CL_tolerence, CD_tolerence=CD_tolerence, 
            max_iteration=max_iteration, distribution_plot=distribution_plot)
            sdf_list = None
            sdf_list_2 = None
        
        same_intended_cond_val_idx, same_intended_cond_train_idx = find_same_condition_in_dataset(AOA, Re, Ma, CL, CD, df2, val_indices, train_indices)    
        plotting_function(data_structure, x_coords_list, y_coords_list, CL_actual_list, CD_actual_list, Ma, Re, AOA, CL, CD, 
                            same_intended_cond_val_idx, same_intended_cond_train_idx, df, min_max, distribution_plot, off_design_plot, x_coords_list_2, y_coords_list_2, CL_actual_list_2, CD_actual_list_2, sdf_list, sdf_list_2)



def validation_accuracy(model_config_path, file_name, band = None, mode = None, CL_axis_min = None, CL_axis_max = None, CD_axis_min = None, CD_axis_max = None):
    
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    data_structure = model_config['data_structure']
    # num_epochs = model_config['num_epochs']
    # model_channel = model_config['model_channel']
    # model_layer = model_config['model_layer']
    # model_channel_multiplication = model_config['model_channel_multiplication']
    # nn_structure=model_config['neural_network_sturcture']
    # model_code = f"{data_structure}_{nn_structure}_{model_channel}_{model_layer}_{len(model_channel_multiplication)}_with_{num_epochs}_epochs"

    results = pd.read_csv(f"{file_name}")

    CL = []
    CD = []
    CL_actual = []
    CD_actual = []
    unfeasible = 0
    unfeasible_design_number = []
    total_design_number = []
    
    if data_structure == '1D_params' or data_structure == '3D_coordinates' or data_structure == '3D_PCA':

        first_variable_name = 'PR'
        second_variable_name = 'Eta'
        first_variable_name_latex = '$PR$'
        second_variable_name_latex = '$\eta$'

        for idx in range(len(results)):
            unfeasible_design_number.append(results['unfeasible_design'].loc[idx])
            total_design_number.append(results['design_iteration'].loc[idx])
            if results['pr_actual'].loc[idx] == 999:
                unfeasible = unfeasible + 1
            else:
                CL.append(results['pr_original'].loc[idx])
                CD.append(results['eta_original'].loc[idx])
                CL_actual.append(results['pr_actual'].loc[idx])
                CD_actual.append(results['eta_actual'].loc[idx]) 
        
    else:
        first_variable_name = 'CL'
        second_variable_name = 'CD'
        first_variable_name_latex = '$C_L$'
        second_variable_name_latex = '$C_D$'
        for idx in range(len(results)):
            unfeasible_design_number.append(results['unfeasible_design'].loc[idx])
            total_design_number.append(results['design_iteration'].loc[idx])
            
            if results['CL_actual'].loc[idx] == 999:
                unfeasible = unfeasible + 1
            else:
                CL.append(results['CL'].loc[idx])
                CD.append(results['CD'].loc[idx])
                CL_actual.append(results['CL_actual'].loc[idx])
                CD_actual.append(results['CD_actual'].loc[idx])

    CL = np.array(CL)
    CD = np.array(CD)
    
    CL_actual = np.array(CL_actual)
    CD_actual = np.array(CD_actual)


    CL_error = (CL - CL_actual)/CL
    CD_error = (CD - CD_actual)/CD
    

    CL_sum_of_square = 0
    CD_sum_of_square = 0
    
    CL_in_bound = 0
    CD_in_bound = 0

    band_1 = band # change here to change the error band
    upper_band_grad = 1 + band_1
    lower_band_grad = 1 - band_1


    for idx, i in enumerate(CL_error):
        if CL[idx] >= CL_axis_min and CL[idx] <= CL_axis_max: 
            if abs(i) <= upper_band_grad-1:
                CL_in_bound += 1
            
            if abs(i) < 1:
                CL_sum_of_square = CL_sum_of_square + i**2
            else:
                print(f'Outlier in {first_variable_name}')
    CL_rmse = (CL_sum_of_square/len(CL_error))**0.5
        

    for idx, i in enumerate(CD_error):
        if CD[idx] >= CD_axis_min and CD[idx] <= CD_axis_max: 
            if abs(i) <= upper_band_grad-1:
                CD_in_bound += 1
            
            if abs(i) < 1:
                CD_sum_of_square = CD_sum_of_square + i**2
            else:
                print(f'Outlier in {second_variable_name}')
    CD_rmse = (CD_sum_of_square/len(CD_error))**0.5
    
    final_unfeasible_percent = 100*(unfeasible/len(results))
    average_design_trials = np.average(total_design_number)
    total_unfeasible_design_percent = 100*(np.sum(unfeasible_design_number))/(np.sum(total_design_number))
    print(f'The validation takes {len(results)} samples, among which {unfeasible} ({final_unfeasible_percent:.2f}%) designs are unfeasible after 100 trials.')
    print(f'The accuracy information of the {len(CL)} feasible designs are shown below:')
    print(f'{first_variable_name} RMSE is:', CL_rmse)
    print(f'{second_variable_name} RMSE is:', CD_rmse)
    print(f'{int(100*(CL_in_bound)/(len(CL)))}% samples have {first_variable_name} within {int(band_1*100)}% relative error.')
    print(f'{int(100*(CD_in_bound)/(len(CL)))}% samples have {second_variable_name} within {int(band_1*100)}% relative error.')
    print(f'Averaged number of trials {average_design_trials:.2f}, averaged percent of unfeasible designs {total_unfeasible_design_percent:.2f}%.')

    fig, axes = plt.subplots(1,2, figsize=(12,5))

    ax1 = axes[0]
    ax2 = axes[1]

    # CL_padding = 0.1
    # CD_padding = 0.01
    # CL_min = min(CL) - CL_padding
    # CL_max = max(CL) + CL_padding
    # CD_min = min(CD) - CD_padding
    # CD_max = max(CD) + CD_padding
    
    # CL_actual_min = min(CL_actual) - CL_padding
    # CL_actual_max = max(CL_actual) + CL_padding
    # CD_actual_min = min(CD_actual) - CD_padding
    # CD_actual_max = max(CD_actual) + CD_padding

    # CL_axis_lower_limit = min(CL_actual_min, CL_min)
    # CL_axis_upper_limit = max(CL_actual_max, CL_max)
    # CD_axis_lower_limit = min(CD_actual_min, CD_min)
    # CD_axis_upper_limit = max(CD_actual_max, CD_max)
    CL_axis_lower_limit = CL_axis_min
    CL_axis_upper_limit = CL_axis_max
    CD_axis_lower_limit = CD_axis_min
    CD_axis_upper_limit = CD_axis_max

    # ================================= KDE
    x = np.array(CL)
    y = np.array(CL_actual)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xx, yy = np.mgrid[CL_axis_min:CL_axis_max:200j, CL_axis_min:CL_axis_max:200j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_coords).reshape(xx.shape)
    # =================================

    if mode == 'scatter':
        ax1.scatter(CL, CL_actual, color = 'b', s=5, label = 'Generated Point')
    elif mode == 'heatmap':
        ax1.imshow(density.T, origin='lower', extent=[CL_axis_min, CL_axis_max, CL_axis_min, CL_axis_max], cmap='Reds', aspect='auto')
    
    m_cl, b_cl = np.polyfit(CL, CL_actual, 1)
    # ax1.plot(np.linspace(CL_min, CL_max, 10), m_cl*np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10) + b_cl, c='g', ls = ':') # line of best fit
    ax1.plot(np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10), np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10), c='r', ls='--', label = '100% Accuracy') # line with grad = 1
    # ax1.plot(np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10), upper_band_grad*np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10), c='r', ls=':', label = f'{int(band_1*100)}% band') # upperbound of 10% error
    # ax1.plot(np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10), lower_band_grad*np.linspace(CL_axis_lower_limit, CL_axis_upper_limit, 10), c='r', ls=':') # lowerbound of 10% error
    ax1.set_xlabel(f'Target {first_variable_name_latex}', fontsize = 14)
    ax1.set_ylabel(f'Generated {first_variable_name_latex}', fontsize = 14)
    ax1.set_xlim(CL_axis_lower_limit, CL_axis_upper_limit)
    ax1.set_ylim(CL_axis_lower_limit, CL_axis_upper_limit)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(linestyle = ':')
    ax1.legend(fontsize = 14)
#     ax1.text(0.97, 0.03, f'$C_L$ RMSE = {CL_rmse:.4f}',
#     transform=ax1.transAxes,
#     fontsize=14,
#     verticalalignment='bottom',
#     horizontalalignment='right',
#     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
# )
    

    # ================================= KDE
    x = np.array(CD)
    y = np.array(CD_actual)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xx, yy = np.mgrid[CD_axis_min:CD_axis_max:200j, CD_axis_min:CD_axis_max:200j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_coords).reshape(xx.shape)
    # =================================

    if mode == 'scatter':
        ax2.scatter(CD, CD_actual,  color = 'b',  s=5)
    elif mode == 'heatmap':
        ax2.imshow(density.T, origin='lower', extent=[CD_axis_min, CD_axis_max, CD_axis_min, CD_axis_max], cmap='Reds', aspect='auto')
    
    m_cd, b_cd = np.polyfit(CD, CD_actual, 1)
    # ax2.plot(np.linspace(CD_min, CD_max, 10), m_cd*np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10) + b_cd, c='g', ls = ':') # line of best fit
    ax2.plot(np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10), np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10), c='r', ls='--') # line with grad = 1
    # ax2.plot(np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10), upper_band_grad*np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10), c='r', ls=':') # upperbound of 10% error
    # ax2.plot(np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10), lower_band_grad*np.linspace(CD_axis_lower_limit, CD_axis_upper_limit, 10), c='r', ls=':') # lowerbound of 10% error
    ax2.set_xlabel(f'Target {second_variable_name_latex}', fontsize = 14)
    ax2.set_ylabel(f'Generated {second_variable_name_latex}', fontsize = 14)
    ax2.set_xlim(CD_axis_lower_limit, CD_axis_upper_limit)
    ax2.set_ylim(CD_axis_lower_limit, CD_axis_upper_limit)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(linestyle = ':')
    # ax2.text(0.97, 0.03, f'$C_D$ RMSE = {CD_rmse:.4f}',
    # transform=ax2.transAxes,
    # fontsize=14,
    # verticalalignment='bottom',
    # horizontalalignment='right',
    # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.show()

    print(f"{first_variable_name} best-fit-line gradient is:", m_cl)
    print(f"{second_variable_name} best-fit-line gradient is:", m_cd)
    save_fig_custom(fig, file_path='fig', file_name=f'single_target_accuracy_plot_{mode}_{data_structure}', 
                    format_list=['.eps', '.png'], overwrite=True, dpi = 500)

    return CL_rmse, CD_rmse, final_unfeasible_percent, average_design_trials, total_unfeasible_design_percent



def training_data_reduction_plot():
    
    pr_rsme_list = []
    eta_rsme_list = []
    final_unfeasible_percent_list = []
    average_design_trials_list = []
    total_unfeasible_design_percent_list = []
    total_training_data = []
    data_percentage_list = [100, 75, 50, 25]
    
    for data_percentage in data_percentage_list:
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                pr_rsme, eta_rsme, final_unfeasible_percent, average_design_trials, total_unfeasible_design_percent = validation_accuracy(model_config_path=f'mdl_hyperparams/3D_coords_conv_2d_unet_{data_percentage}.yaml', 
                                file_name = f'mdl_validation/3D_coordinates_ResNet_UNet_2D_64_5_4_with_300_epochs_{data_percentage/100:g}_data.csv',
                                band = 0.1, mode = 'scatter', CL_axis_min=1, CL_axis_max=4, CD_axis_min=0.75, CD_axis_max=0.9)
        pr_rsme_list.append(pr_rsme*100)
        eta_rsme_list.append(eta_rsme*100)
        final_unfeasible_percent_list.append(final_unfeasible_percent)
        average_design_trials_list.append(average_design_trials)
        total_unfeasible_design_percent_list.append(total_unfeasible_design_percent)
        total_training_data.append(data_percentage*34442*0.8/100)
    
    fig, ax = plt.subplots()
    ax.plot(data_percentage_list, pr_rsme_list, 'b', marker = 'o')
    ax.set_xlabel('Training Data Reduction %', fontsize = 14)
    ax.set_ylabel('Pressure Ratio RMSE %', fontsize = 14)
    ax.grid(True, ls = ':')


    fig_2, ax_2 = plt.subplots()
    ax_2.plot(data_percentage_list, eta_rsme_list, 'b', marker = 'o')
    ax_2.set_xlabel('Training Data Reduction %', fontsize = 14)
    ax_2.set_ylabel('Total Efficiency RMSE %', fontsize = 14)
    ax_2.grid(True, ls = ':')

    fig_3, ax_3 = plt.subplots()
    ax_3.plot(data_percentage_list, average_design_trials_list, 'b', marker = 'o')
    ax_3.set_xlabel('Training Data Reduction %', fontsize = 14)
    ax_3.set_ylabel('Average Design Trials', fontsize = 14)
    ax_3.grid(True, ls = ':')

    fig_4, ax_4 = plt.subplots()
    ax_4.plot(data_percentage_list, total_unfeasible_design_percent_list, 'b', marker = 'o')
    ax_4.set_xlabel('Training Data Reduction %', fontsize = 14)
    ax_4.set_ylabel('Total Unfeasible Design %', fontsize = 14)
    ax_4.grid(True, ls = ':')

    

    plt.show()



def save_denoising_gif(gif_path, frames, frame_duration=0.1, pause_time=2.0):
    images, shapes = [], []

    for f in frames:
        im = imageio.imread(f) if isinstance(f, (str, os.PathLike)) else np.asarray(f)
        if im.ndim == 3 and im.shape[2] == 4:
            im = im[:, :, :3]
        images.append(im)
        shapes.append(im.shape)

    max_h = max(s[0] for s in shapes)
    max_w = max(s[1] for s in shapes)

    padded = []
    for im in images:
        h, w = im.shape[:2]
        padded.append(np.pad(im, ((0, max_h-h), (0, max_w-w), (0, 0)),
                             mode="constant", constant_values=255))

    # per-frame durations (seconds)
    durations = [frame_duration] * len(padded)
    durations[-1] = pause_time

    imageio.mimsave(gif_path, padded, duration=durations, loop=0)
    print("GIF saved as:", gif_path)






def denoise_process_plot(model, data_structure, device, sample_size, num_steps, pca, fig_size, manual_seed = None, x_res = 128, y_res = 128):

    model.eval()
    text_font = 'Liberation Sans'
    math_font = 'stix'

    plt.rcParams['mathtext.fontset'] = math_font
    rcParams['font.family'] = text_font

    df, df2, val_indices, train_indices, min_max = load_data_set(data_structure)

    if manual_seed != None:
        random.seed(manual_seed)
    numbers = random.sample(range(0, len(val_indices)), sample_size)


    y_max = min_max['y_max'].loc[0]
    y_min = min_max['y_min'].loc[0]

    for index in numbers:
        # name, x_coords_original, y_coords_original, Ma, Re, AOA, CL, CD, AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised = load_validation_condition(index, data_structure, device)
        # cond = (torch.tensor([AOA_normalised, Ma_normalised, Re_normalised, CL_normalised, CD_normalised]).to(device))

        Ma, Re, AOA, CL, CD, cond = test_plot(3, 0.2, 350000, 0.2960999999999999, 0.012509999999999985, data_structure, device)
        
        
        # use the model to generate coordinates
        rnd = StackedRandomGenerator(device, range(sample_size))
        
        if data_structure !='sdf':
            latents = rnd.randn([sample_size, model.in_dim], device=device)
        else: 
            x_resolution = x_res
            y_resolution = y_res
            latents = torch.randn(sample_size, 1, x_resolution, y_resolution, device=device)
        with torch.no_grad():
            samples, trajectory = edm_sampler(model, latents=latents, class_labels=cond, randn_like=rnd.randn_like, num_steps=num_steps, deterministic=False) 
        samples = samples.float()
        sample = samples[0].cpu().numpy()
        print(f"{num_steps} sampling steps")
        
        if data_structure == 'pca':
            sample = pca.inverse_transform(sample)

        # denormalise the generated coordinates
        if data_structure !='sdf':
            x_coords = sample[0::2]
            y_coords_normalised = sample[1::2]
            y_coords = y_coords_normalised*(y_max - y_min)+y_min

            # xfoil to calculate the data for the genrated design
            # CL_actual, CD_actual = xfoil_calculation(x_coords, y_coords, AOA, Re, Ma, CL, CD, x_foil_timeout=5)

        frames = []  # store frame paths
        output_dir = "frames"
        os.makedirs(output_dir, exist_ok=True)

        # Plotting function
        for i in range(num_steps+1):
            if i <= num_steps-1: 
                samples = trajectory[i]
            else:
                samples = trajectory[-1]
            samples = samples.float()
            sample = samples[0].cpu().numpy()
            

            if data_structure == 'pca':
                sample = pca.inverse_transform(sample)


            if data_structure!='sdf':
                # denormalise the generated coordinates
                x_coords = sample[0::2]
                y_coords_normalised = sample[1::2]
                y_coords = y_coords_normalised*(y_max - y_min)+y_min
                
                fig, ax = plt.subplots(figsize=fig_size, dpi = 300)
                ax.grid(True, ls=':')
                # ax.plot(x_coords,y_coords, color = 'b', label = 'generated')
                # ax.plot(x_coords_original,y_coords_original, color = 'r', label = 'original')
                if data_structure == 'pca':
                    ax.plot(x_coords,y_coords, color = 'b', linewidth = 5)
                    
                else:
                    ax.scatter(x_coords[::4], y_coords[::4], color='b')
                    if i == num_steps:
                        ax.plot(x_coords,y_coords, color = 'b')
                # ax.plot(x_coords_original,y_coords_original, color = 'r')
                ax.set_xlim(-0.01, 1.01)
                # ax.set_ylim(-0.2,0.2)
                # if i == num_steps-1:
                #     ax.set_title(' \n'
                #                 f'Denoising Sampling Step {i+1} \n'
                #                 f"Ma: {Ma:.1f} "
                #                 f"Re: {int(Re)} "
                #                 f"AOA: {int(AOA)} deg \n"
                #                 f"Intended CD: {CD:.2f} "
                #                 f"Intended CL: {CL:.2f} \n"
                #                 f"Actual CD: {CD_actual:.2f} "
                #                 f"Actual CL: {CL_actual:.2f} ")
                # else:
                #     ax.set_title(f'Denoising Sampling Step {i+1}')
                # ax.legend()
            
                
            else:
                
                x_lower_lim = -0.01
                x_upper_lim = 1.01
                y_lower_lim = -0.17
                y_upper_lim = 0.27
                sdf_max = min_max['sdf_max'].loc[0]
                sdf_min = min_max['sdf_min'].loc[0]
                xs = np.linspace(x_lower_lim, x_upper_lim, x_resolution)
                ys = np.linspace(y_lower_lim, y_upper_lim, y_resolution)

                sample_img = sample.reshape(y_resolution, x_resolution)
                sdf_denormalised = (sample_img + 1)*(sdf_max - sdf_min)/2 + sdf_min

                colors = [
                    (0.0, "#ff0000"),  # blue-ish for negative values
                    (0.5, "#FFFFFF"),    # exact zero
                    (1.0, "#0113CD")   # red-ish for positive values
                ]
                my_cmap = LinearSegmentedColormap.from_list("my_diverging", colors)
                limit = 0.3
                factor1 = 0.35
                factor2 = 0.15
                cmap=my_cmap # change here

                fig = plt.figure()
                ax0 = fig.add_subplot(111)
                edge = ax0.contour(xs, ys, sdf_denormalised, levels=[0])
                aerofoil_edge = edge.allsegs[0]
                plt.close(fig)

                x_list = aerofoil_edge[0][:, 0]
                y_list = aerofoil_edge[0][:, 1]
                x_coords, y_coords = sort_for_xfoil(x_list, y_list)

                fig, ax = plt.subplots()

                
                norm = TwoSlopeNorm(vmin=-0.1*factor1, vcenter=0, vmax=0.3*factor2)
                contour = ax.contourf(xs, ys, sdf_denormalised, levels=500, cmap=cmap, norm = norm)
                ax.grid(True, ls=':')
                # contour = ax.contourf(xs, ys, sdf_denormalised, vmin = limit, vmax = -limit, levels=100, cmap=color_map)
                
                # norm = TwoSlopeNorm(vmin=limit, vcenter=0, vmax=-limit)

                # contour = ax.contourf(xs, ys, sdf_denormalised, levels=100,
                    #   cmap='Spectral', norm=norm)
                
                
                # ax.contourf(xs, ys, sdf_denormalised, levels=100, cmap=color_map)
                if i == num_steps:
                    # pass
                    # ax.contour(xs, ys, sdf_denormalised, levels=[0], colors='b')
                    ax.plot(x_coords, y_coords, color = 'k', linewidth = 3, ls = '--')
                    # fig.colorbar(contour, ax=ax, label="Signed Distance")
            ax.set_ylim(-0.17, 0.2)
            
            ax.set_aspect('equal', 'box')
            
            if data_structure== 'sdf' and i == num_steps:
                pass
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="4%", pad=0.1)
                # vmin, vmax = limit, -limit
                # cbar = fig.colorbar(contour, cax=cax)
                # ticks = np.arange(vmin, vmax + 0.001, 0.06)
                # cbar.set_ticks(ticks)
                # cbar.ax.tick_params(labelsize=14)
                # cbar.set_label("Signed Distance", fontsize = 18)
            
            ax.tick_params(bottom=False, top=False, left=False, right=False,
               labelbottom=False, labelleft=False)
            # ax.tick_params(axis='both', which='major', labelsize=18)

            frame_path = f"{output_dir}/{data_structure}_frame_{i}.png"
            plt.savefig(frame_path)
            plt.close(fig)
            save_fig_custom(fig, file_path='fig', file_name=f'{data_structure}_frame_{i}', overwrite=True, dpi = 500)
            frames.append(frame_path)
        
        print(
            f'Denoising Sampling Step {i+1} \n'
            f"Ma: {Ma:.1f} "
            f"Re: {int(Re)} "
            f"AOA: {int(AOA)} deg \n"
            f"Intended CD: {CD} "
            f"Intended CL: {CL} \n"
            )
        
        gif_path = f"denoising_gif/denoising_process_{data_structure}_{num_steps}.gif"
        with imageio.get_writer(gif_path, mode="I", duration=2.0) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))

        print("GIF saved as:", gif_path)
    Image(filename=gif_path)



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





def load_1D_dataset():
    df = pd.read_csv('dataset/1D_compressor_geometry_normalised.csv')
    min_max = pd.read_csv('dataset/1D_compressor_geometry_minmax.csv')
    val_indices = np.load('dataset/3D_test_indices.npy')

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
    


def convert_1D_to_3D(multiple_design_geometry, compressor_code, convert_to_3D):
    
    
    design_number = 1
    for geometry in multiple_design_geometry:
        compressor_path = f'/home/yg1922/Desktop/Yingfan_FYP_Code/Diffusion_based_2DAirfoil_Generation/generated_compressor_3D_geometry/compressor_{compressor_code}_design_{design_number}'
        
        vaneless_diff_existence = True
        splitter_existence = True
        target_rake_angle = 10
        pinching_ratio = 0.4

        three_d_blade = Blade_Forming_3D(splitter_existence, target_rake_angle, compressor_path, geometry, vaneless_diff_existence, pinching_ratio, extreme_value_for_imp_inlet=-int(geometry['R_tip_1'] * 1000 * 2))
        
        with contextlib.redirect_stdout(open(os.devnull, 'w')): # to suppress all the print outs
            three_d_blade.execute(convert_to_3D)
        design_number += 1




def test_condition_1D(m_dot, RPM, pr, eta):

    omega = RPM * 2*np.pi / 60
    df, min_max, val_indices = load_1D_dataset()

    pr_normalised = (pr - min_max.loc['pressure_ratio', 'min'])/(min_max.loc['pressure_ratio', 'max'] - min_max.loc['pressure_ratio', 'min'])
    m_dot_normalised = (m_dot - min_max.loc['m_dot', 'min'])/(min_max.loc['m_dot', 'max'] - min_max.loc['m_dot', 'min'])
    eta_normalised = (eta - min_max.loc['efficiency', 'min'])/(min_max.loc['efficiency', 'max'] - min_max.loc['efficiency', 'min'])
    omega_normalised = (omega - min_max.loc['omega', 'min'])/(min_max.loc['omega', 'max'] - min_max.loc['omega', 'min'])
    
    return pr_normalised, m_dot_normalised, eta_normalised, omega_normalised




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





def visualise_3D(compressor_code, m_dot, RPM, pr, eta, convert_to_3D, geometry):
    
    
    if convert_to_3D:
        main_blade_profiles = load_blade_curve(f"generated_compressor_3D_geometry/compressor_{compressor_code}/3D_blades_0.40/BladeMain.curve")
        main_blade_profiles = [main_blade_profiles[0], main_blade_profiles[-1]]
        splitter_blade_profiles = load_blade_curve(f"generated_compressor_3D_geometry/compressor_{compressor_code}/3D_blades_0.40/BladeSplitter.curve")
    
        all_points_main = np.vstack(main_blade_profiles)
        x_main, y_main, z_main = all_points_main[:, 0], all_points_main[:, 1], all_points_main[:, 2]

        all_points_splitter = np.vstack(main_blade_profiles)
        x_splitter, y_splitter, z_splitter = all_points_splitter[:, 0], all_points_main[:, 1], all_points_main[:, 2]


    # 3D view plotting
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")

    # for profile in main_blade_profiles:
    #     ax.scatter(profile[:, 0], profile[:, 1], profile[:, 2], color="k", s=1)
        
    # ax.set_xlabel("X (mm)")
    # ax.set_ylabel("Y (mm)")
    # ax.set_zlabel("Z (mm)")
    # ax.view_init(elev=25, azim=135) 
    # plt.tight_layout()

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    cmap = plt.get_cmap("tab10")

    for design_number in range(len(geometry)):

        case = f'generated_compressor_3D_geometry/compressor_{compressor_code}_design_{design_number+1}/Main_Blades/casing.dat'
        hub = f'generated_compressor_3D_geometry/compressor_{compressor_code}_design_{design_number+1}/Main_Blades/hub.dat'
        thickness = f'generated_compressor_3D_geometry/compressor_{compressor_code}_design_{design_number+1}/Main_Blades/thickness.dat'
        beta = f'generated_compressor_3D_geometry/compressor_{compressor_code}_design_{design_number+1}/Main_Blades/beta.dat'

        case = np.loadtxt(case)
        meridional_tip_x, meridional_tip_r = case[:, 0], case[:, 1]

        hub = np.loadtxt(hub)
        meridional_hub_x, meridional_hub_r = hub[:, 0], hub[:, 1]

        thickness = np.loadtxt(thickness, skiprows=2)
        thickness_meridional, thickness_hub, thickness_tip = thickness[:, 0], thickness[:, 1], thickness[:, 2]

        beta = np.loadtxt(beta, skiprows=2)
        beta_meridional, beta_hub, beta_tip = beta[:, 0], beta[:, 1], beta[:, 2]



        # meridional view plotting
        
        color = cmap((design_number+1) % 10)

        ax.plot(meridional_tip_x, meridional_tip_r, color = color)
        ax.plot(meridional_hub_x, meridional_hub_r, color = color) 
        ax.set_xlabel('Axial (mm)')
        ax.set_ylabel('Radial (mm)')
        ax.grid(True, ls = ':')
        ax.text(0.27, 0.90, fr'$\dot{{m}}$: {m_dot:.2f} Kg/s   '
                f"RPM: {int(RPM)} \n"
                f"PR: {pr:.2f}   "
                f"$\eta$: {eta:.2f}",
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.65))
        ax.axis('equal')
        
        
        
        ax2.plot(thickness_meridional, thickness_hub, color = color)
        ax2.plot(thickness_meridional, thickness_tip, color = color) 
        ax2.set_xlabel('Axial (mm)')
        ax2.set_ylabel('Thickness (mm)')
        ax2.grid(True, ls = ':')

        
        ax3.plot(beta_meridional, beta_hub, color = color)
        ax3.plot(beta_meridional, beta_tip, color = color) 
        ax3.set_xlabel('Axial (mm)')
        ax3.set_ylabel('Metal Angle (deg)')
        ax3.grid(True, ls = ':')
        
    plt.show()



def off_design_plot_1D(multi_design_geometry, m_dot_design_point, omega, pr_design, eta_design):

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    cmap = plt.get_cmap("tab10")
    ax.scatter(m_dot_design_point, pr_design, marker='*', s= 50, c = 'r', zorder = 5, label = 'Design Point')
    ax2.scatter(m_dot_design_point, eta_design, marker='*', s= 50, c = 'r', zorder = 5, label = 'Design Point')
    
    for design_number, geometry in enumerate(multi_design_geometry):
        
        color = cmap((design_number+1) % 10)
        pr_list = []
        eta_list = []
        m_dot_list_2 = [] # to ensure same length as pr and eta lists
        m_dot_list = np.linspace(0.07, 0.12, 20)


        for m_dot in m_dot_list:
            pr, eta = run_meanline(geometry, m_dot, omega, 5)

            if pr != 999:
                m_dot_list_2.append(m_dot)
                pr_list.append(pr)
                eta_list.append(eta)

        
        
        ax.plot(m_dot_list_2, pr_list, color = color, zorder = 1)
        ax.set_xlabel('Mass Flow Rate (kg/s)')
        ax.set_ylabel('Pressure Ratio')
        ax.set_xlim(0.07, 0.12)
        ax.set_ylim(1,4.5)
        ax.grid(True, ls = ':')
        ax.legend()


        
        ax2.plot(m_dot_list_2, eta_list, color = color, zorder = 1)
        ax2.set_xlabel('Mass Flow Rate (kg/s)')
        ax2.set_ylabel('Total Efficiency')
        ax2.set_xlim(0.07, 0.12)
        ax2.set_ylim(0.7, 0.9)
        ax2.grid(True, ls = ':')
        ax2.legend()

        save_fig_custom(fig, file_path='fig', file_name=f'off_design_pr_plot', overwrite=True, dpi = 500)
        save_fig_custom(fig2, file_path='fig', file_name=f'off_design_eta_plot', overwrite=True, dpi = 500)
    plt.show()



def validation_1D(mode, device, sample_number, model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, 
                  m_dot, RPM, eta, pr, convert_to_3D, plot_blade_distribution):
    
    model.eval()
    df, min_max, val_indices = load_1D_dataset()
    


    pr_tolerance = CL_tolerence
    eta_tolerance = CD_tolerence

    if mode == 'validation':
        numbers = randomly_pick_1D_validation(manual_seed, sample_number)
    else:
        numbers = [0]

    for idx in numbers:
        
        multiple_design_geometry = []
        if mode == 'validation':
            i = val_indices[idx]
            pr_normalised = df.loc[i, 'pressure_ratio']
            eta_normalised = df.loc[i, 'efficiency']
            omega_normalised = df.loc[i, 'omega']
            m_dot_normalised = df.loc[i, 'm_dot']
            
            
        else:
            pr_normalised, m_dot_normalised, eta_normalised, omega_normalised = test_condition_1D(m_dot, RPM, pr, eta)

        design = 0
        
        while design < multiple_design:

            number_of_trials = 1
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

                deviation =  0.0000000000000001

                mins = min_max.loc[geom_cols, "min"].to_numpy(dtype=np.float32)
                maxs = min_max.loc[geom_cols, "max"].to_numpy(dtype=np.float32)


                denormalised_geometry = sample * (maxs + deviation - mins) + mins

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
                    'nblades': math.ceil(denormalised_geometry[11]),
                    'n_splitter_blades': math.ceil(denormalised_geometry[11])/2, # ensure equal number of main and splitter blades
                    'b3': float(denormalised_geometry[13]), 
                    'r3': float(denormalised_geometry[14]),
                    'slip_factor': float(denormalised_geometry[15])}

                m_dot = m_dot_normalised*(min_max.loc['m_dot', 'max'] + deviation - min_max.loc['m_dot', 'min']) + min_max.loc['m_dot', 'min']
                omega = omega_normalised*(min_max.loc['omega', 'max'] + deviation - min_max.loc['omega', 'min']) + min_max.loc['omega', 'min']

                pr_original = pr_normalised*(min_max.loc['pressure_ratio', 'max'] + deviation - min_max.loc['pressure_ratio', 'min']) + min_max.loc['pressure_ratio', 'min']
                eta_original = eta_normalised*(min_max.loc['efficiency', 'max'] + deviation - min_max.loc['efficiency', 'min']) + min_max.loc['efficiency', 'min']

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
                    print(f'Number {design+1} design took {number_of_trials} trials.')
                    print(f'Pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                    print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')
                    success = True
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
            
            multiple_design_geometry.append(geometry)
            print(geometry)
            print(m_dot, omega)

        
        if plot_blade_distribution:
            
            compressor_code = f'{m_dot:.2f}_{int(RPM)}_{pr_original:.2f}_{eta_original:.2f}'
            print(f'Plotting the blade distribution. Compressor code: {compressor_code}')
            if convert_1D_to_3D:
                print('Converting from 1D to 3D.')
            convert_1D_to_3D(multiple_design_geometry, compressor_code, convert_to_3D)
            visualise_3D(compressor_code, m_dot, RPM, pr_original, eta_original, convert_to_3D, multiple_design_geometry)
            off_design_plot_1D(multiple_design_geometry, m_dot, omega, pr, eta)




def cyl_to_cart_about_x(x, r, theta):
    y = r * np.cos(theta)
    z = r * np.sin(theta)
    return x, y, z




def geometry_3D_to_1D_conversion(x,y,z,number_of_blades, ax_3D):
    


    debug = False

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

    if debug:
        ax_3D.scatter(inlet_hub_1[0], inlet_hub_1[1], inlet_hub_1[2], color = 'g', s=10)
        ax_3D.scatter(inlet_hub_2[0], inlet_hub_2[1], inlet_hub_2[2], color = 'g', s=10)

    # SS
    idx_1 = 261
    idx_2 = idx_1+1
    inlet_hub_3 = [x[idx_1], y[idx_1], z[idx_1]]
    inlet_hub_4 = [x[idx_2], y[idx_2], z[idx_2]]
    dx = abs(inlet_hub_4[0] - inlet_hub_3[0])
    dz = abs(inlet_hub_4[2] - inlet_hub_3[2])
    beta_b1_hub_2 = -np.atan2(dz, dx)
    
    if debug:    
        ax_3D.scatter(inlet_hub_3[0], inlet_hub_3[1], inlet_hub_3[2], color = 'g', s=10)
        ax_3D.scatter(inlet_hub_4[0], inlet_hub_4[1], inlet_hub_4[2], color = 'g', s=10)


    # Calculate the average value of the two faces
    beta_b1_hub = (beta_b1_hub_1 + beta_b1_hub_2)/2

    if debug:
        print(beta_b1_hub_1, beta_b1_hub_2, beta_b1_hub, 'hub angles')
    

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
    
    if debug:
        ax_3D.scatter(inlet_tip_1[0], inlet_tip_1[1], inlet_tip_1[2], color = 'g', s=10)
        ax_3D.scatter(inlet_tip_2[0], inlet_tip_2[1], inlet_tip_2[2], color = 'g', s=10)


    # The upper face
    idx_1 = 261
    idx_2 = idx_1+1
    inlet_tip_3 = [x[idx_1], y[idx_1], z[idx_1]]
    inlet_tip_4 = [x[idx_2], y[idx_2], z[idx_2]]
    dx = abs(inlet_tip_4[0] - inlet_tip_3[0])
    dz = abs(inlet_tip_4[2] - inlet_tip_3[2])
    beta_b1_tip_2 = -np.atan2(dz, dx)
    
    if debug:
        ax_3D.scatter(inlet_tip_3[0], inlet_tip_3[1], inlet_tip_3[2], color = 'g', s=10)
        ax_3D.scatter(inlet_tip_4[0], inlet_tip_4[1], inlet_tip_4[2], color = 'g', s=10)


    # Calculate the average value of the two faces
    beta_b1_tip = (beta_b1_tip_1 + beta_b1_tip_2)/2
    
    if debug:
        print(beta_b1_tip_1, beta_b1_tip_2, beta_b1_tip, 'tip angles')

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
                'n_splitter_blades': int(number_of_blades)/2, # ensure equal number of main and splitter blades
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
        ps_end_index = 249 + 512*profile_idx
        ss_start_index = ps_end_index + 13
        ss_end_index = 512 + 512*profile_idx


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
        
    x_smooth_list = np.concatenate(x_smooth_list)
    y_smooth_list = np.concatenate(y_smooth_list)
    z_smooth_list = np.concatenate(z_smooth_list)

    return x_smooth_list, y_smooth_list, z_smooth_list





def create_hub_curve_file(x_hub, r_hub, r_1_tip, r_2, compressor_code, vaneless_existence = True, pinching = True):

    r_3 = r_2 * 1.5 * 1000
    extreme_value = - int(r_1_tip * 2 * 1000)
    curve_path = f'generated_compressor_3D_geometry/{compressor_code}/Hub.curve'



    # Process lines to add a zero column
    x = x_hub
    r = r_hub
    

    all_lines = []


    # Use the second and third columns of the first processed line for the extreme values
    
    for i in np.arange(extreme_value, 1, 1):
        line = f'{i}, {r[0]}, {0}\n'
        all_lines.append(line)
    
    for idx, _ in enumerate(x):
        
        line = f'{x[idx]} {r[idx]} {0}\n'
        all_lines.append(line)



    num_extension_points = 50

    if vaneless_existence and r[-1] < r_3:
        for r_extended in np.linspace(r[-1], r_3, num_extension_points):
            line = f"{x[-1]} {r_extended} {0}\n"
            all_lines.append(line)
        
        
    elif not vaneless_existence:
        extension_percentage = 1.35
        r_3 = r[-1] * extension_percentage
        for r_extended in np.linspace(r[-1], r_3, num_extension_points):
            line = f"{x[-1]} {r_extended} {0}\n"
            all_lines.append(line)
            

    else:
        print('The geometry has problem as the impeller outlet radius is greater than the vaneless diffuser. ')

   
    if pinching:
        num_extension_points_pinching = 50
        pinching_extension_percentage = 1.211

        for r_extended in np.linspace(r_3, r_3 * pinching_extension_percentage, num_extension_points_pinching):

            line = f"{x[-1]} {r_extended} {0}\n"
            all_lines.append(line)
        
        print(f"Hub curve file created at {curve_path} (included pinching-related extension).")
    else:
        print(f"Hub curve file created at {curve_path} (pinching-related extension skipped).")


    with open(curve_path, 'w') as file:
        file.writelines(all_lines)

    



def create_shroud_curve_file(x_shroud, r_shroud, compressor_code, r_1_tip, r_2, b_2, vaneless_existence = True, pinching = True):
    
    r_2 = r_2 * 1000
    b_2 = b_2 * 1000
    r_3 = r_2 * 1.5
    b_3 = b_2 * 0.8

    extreme_value = - int(1000*r_1_tip * 2)

    NUM_TAPER_POINTS   = 50      # pts for taper zone  (Δr = b₂)
    NUM_CONST_POINTS   = 50      # pts for constant-width zone  (r₂+b₂ → r₃)
    NUM_PAR_WALL_PTS   = 50      # pts for parallel wall when *no* vaneless
    NUM_PINCH_PTS      = 50      # pts *excluding* the first node (so +1 later)


    curve_path = f'generated_compressor_3D_geometry/{compressor_code}/Shroud.curve'

    
    hub_curve_path = f'generated_compressor_3D_geometry/{compressor_code}/Hub.curve'
    
    with open(hub_curve_path, 'r') as hf:
        last_hub_x = float(hf.readlines()[-1].split()[0])

    x = x_shroud
    r = r_shroud
    
    all_lines = []


    for i in np.arange(extreme_value, 1, 1):
        line = f'{i}, {r[0]}, {0}\n'
        all_lines.append(line)


    for idx, _ in enumerate(x):
        line = f'{x[idx]} {r[idx]} {0}\n'
        all_lines.append(line)



    if vaneless_existence and r[-1] < r_3:

        taper_end_radius = r[-1] + b_2
        taper_end_radius = min(taper_end_radius, r_3)

        Δr_taper  = taper_end_radius - r[-1]
        Δx_total  = b_2 - b_3

        Δx_total  = Δx_total * (Δr_taper / b_2) # ???
        Δr_step   = Δr_taper / NUM_TAPER_POINTS
        Δx_step   = Δx_total / NUM_TAPER_POINTS

        
        for i in range(1, NUM_TAPER_POINTS + 1):
            new_r = r[-1] + i * Δr_step
            new_x = x[-1] + i * Δx_step
            all_lines.append(f"{new_x} {new_r} {0}\n")

        # update “current” position after taper
        x_s_current = x[-1] + Δx_total
        r_s_current  = taper_end_radius

       
        if r_s_current < r_3:
            Δr_const  = r_3 - r_s_current
            Δr_step_c = Δr_const / NUM_CONST_POINTS

            
            for i in range(1, NUM_CONST_POINTS + 1):
                new_r = r_s_current + i * Δr_step_c
                all_lines.append(f"{x_s_current} {new_r} {0}\n")
            r_s_current = new_r
            

    else:

        extended_r1 = r_s_current * 1.35
        Δr_pw = (extended_r1 - r_s_current) / NUM_PAR_WALL_PTS

        for i in range(1, NUM_PAR_WALL_PTS + 1):
            new_r = r_s_current + i * Δr_pw
            all_lines.append(f"{x_s_current} {new_r} {0}\n")

        r_s_current = extended_r1   # for pinching later



    if pinching:
        r_s_end = r_s_current * 1.211
        area_ratio = 0.4

        x_h = last_hub_x
        x_s_end = x_h + (r_s_current * (x_s_current - x_h) * area_ratio) / r_s_end

        num_total = NUM_PINCH_PTS + 1

        for i in range(num_total + 1):
            t = i / num_total
            r = r_s_current + (r_s_end - r_s_current) * t
            x = x_s_current + (x_s_end - x_s_current) * (1 - math.cos(math.pi * t)) / 2
            all_lines.append(f"{x} {r} {0}\n")
        print(f"Shroud curve file created at {curve_path} (included pinching phase).")
    else:
        print(f"Shroud curve file created at {curve_path} (pinching phase skipped).")


    with open(curve_path, 'w') as f_out:
        f_out.writelines(all_lines)





def validation_3D(mode, device, sample_number, model, aux_model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration, 
                  m_dot, RPM, eta, pr, pca, data_structure, off_design_plot, number_of_profiles = None, number_of_points = None, smoothening = False):


    model.eval()
    aux_model.eval()

    df, min_max, val_indices = load_1D_dataset()



    df_2 = pd.read_csv('dataset/1D_compressor_geometry.csv')
    df_3 = pd.read_csv('dataset/polar_minmax_per_compressor_normalised.csv')
    df_4 = pd.read_csv('dataset/polar_secondary_minmax.csv')



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


    fig_3D = plt.figure(figsize=(10, 8))
    ax_3D = fig_3D.add_subplot(111, projection="3d")
    fig, ax = plt.subplots()

    if mode == 'validation':
        numbers = randomly_pick_1D_validation(manual_seed, sample_number)
    else:
        numbers = [0]

    for idx in numbers:


        if mode == 'validation':
            i = val_indices[idx]

            pr_normalised = df.loc[i, 'pressure_ratio']
            eta_normalised = df.loc[i, 'efficiency']
            omega_normalised = df.loc[i, 'omega']
            m_dot_normalised = df.loc[i, 'm_dot']
            n_blades = df.loc[i, 'nblades']
            n_splitter = df.loc[i, 'n_splitter_blades']


            compressor_index = df_2.loc[i, 'geometry_index']

            
            row = df_3.loc[df_3['compressor_index'].astype(int) == int(compressor_index)]
            df_3 = row.iloc[0]

            r_min = df_3['r_min']
            r_max = df_3['r_max']
            theta_min = df_3['theta_min']
            theta_max = df_3['theta_max']
            xc_min = df_3['x_min']
            xc_max = df_3['x_max']


            print('This is compressor', compressor_index)

            original_profiles, profile_has_error = load_blade_curve(f'dataset/3D_compressor_polar_normalised/compressor_{compressor_index}.curve')
            if profile_has_error:
                print('This profile has error!')
            
            for profile in original_profiles:
                r_min_denormalised = r_min * (r_min_max - r_min_min) + r_min_min
                r_max_denormalised = r_max * (r_max_max - r_max_min) + r_max_min
                theta_min_denormalised = theta_min * (theta_min_max - theta_min_min) + theta_min_min
                theta_max_denormalised = theta_max * (theta_max_max - theta_max_min) + theta_max_min
                x_min_denormalised = xc_min * (x_min_max - x_min_min) + x_min_min
                x_max_denormalised = xc_max * (x_max_max - x_max_min) + x_max_min
                

                x_normalised = profile[:, 0]
                r_normalised = profile[:, 1]
                theta_normalised = profile[:, 2]
                r = r_normalised * (r_max_denormalised - r_min_denormalised) + r_min_denormalised 
                theta = theta_normalised * (theta_max_denormalised - theta_min_denormalised) + theta_min_denormalised 
                x = x_normalised * (x_max_denormalised - x_min_denormalised) + x_min_denormalised 
                
                
                x, y, z = cyl_to_cart_about_x(x, r, theta)
                
                
                ax_3D.plot(x, y, z, color='r')
                ax.scatter(x, (y**2 + z**2)**0.5, s = 1, color = 'r')
            x_1 = np.concatenate(original_profiles).ravel()
            x_1 = np.expand_dims(x_1, 0)
            x_1 = torch.tensor(x_1, dtype=torch.float32)
            x_1 = x_1.view(1, 16, 512, 3)
            x_1 = x_1.permute(0, 3, 1, 2)
            
        
        else:
            pr_normalised, m_dot_normalised, eta_normalised, omega_normalised = test_condition_1D(m_dot, RPM, pr, eta)


        
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

                # The auxiliary model, input 4 output 8
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
                n_splitter =  sample[7]

                
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
                    x_2 = samples[0].cpu()
                    sample = sample.ravel()
                
                x_2 = x_2.view(1, 16, 512, 3)
                x_2 = x_2.permute(0, 3, 1, 2)

                r_normalised = sample[1::3]
                theta_normalised = sample[2::3]
                x_normalised = sample[0::3]
                

                r_min_denormalised = r_min * (r_min_max - r_min_min) + r_min_min
                r_max_denormalised = r_max * (r_max_max - r_max_min) + r_max_min
                theta_min_denormalised = theta_min * (theta_min_max - theta_min_min) + theta_min_min
                theta_max_denormalised = theta_max * (theta_max_max - theta_max_min) + theta_max_min
                x_min_denormalised = xc_min * (x_min_max - x_min_min) + x_min_min
                x_max_denormalised = xc_max * (x_max_max - x_max_min) + x_max_min

                print(r_min_denormalised, r_max_denormalised, x_min_denormalised, x_max_denormalised)


                r = r_normalised * (r_max_denormalised - r_min_denormalised) + r_min_denormalised 
                theta = theta_normalised * (theta_max_denormalised - theta_min_denormalised) + theta_min_denormalised 
                x = x_normalised * (x_max_denormalised - x_min_denormalised) + x_min_denormalised 

                x, y, z = cyl_to_cart_about_x(x, r, theta)
                
                if smoothening:
                    print('Smoothening is enabled')
                    x, y, z = smoothening_3D(x,y,z, 3, 11) # third order polynomial fit with window of 11

                if round(n_blades) == 0:
                    number_of_blades =10
                else:
                    number_of_blades =12

                geometry_1D = geometry_3D_to_1D_conversion(x,y,z, number_of_blades, ax_3D)
                    
                    


                m_dot = m_dot_normalised*(min_max.loc['m_dot', 'max']  - min_max.loc['m_dot', 'min']) + min_max.loc['m_dot', 'min']
                omega = omega_normalised*(min_max.loc['omega', 'max']  - min_max.loc['omega', 'min']) + min_max.loc['omega', 'min']
                pr_original = pr_normalised*(min_max.loc['pressure_ratio', 'max']  - min_max.loc['pressure_ratio', 'min']) + min_max.loc['pressure_ratio', 'min']
                eta_original = eta_normalised*(min_max.loc['efficiency', 'max']  - min_max.loc['efficiency', 'min']) + min_max.loc['efficiency', 'min']
                print('m_dot', m_dot, 'omega', omega)
                
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

                if pr_error < pr_tolerance and eta_error < eta_tolerance:
                    print(f'Number {design+1} design took {number_of_trials} trials.')
                    print(f'Pressure ratio {pr} has relative error {pr_error}% compared to {pr_original}.')
                    print(f'Efficiency {eta} has relative error {eta_error}% compared to {eta_original}. ')
                    success = True
            
            design += 1

            if number_of_trials == max_iteration and not success:

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

            compressor_code = f'{m_dot:.2f}_{int(RPM)}_{pr_original:.2f}_{eta_original:.2f}_design_{design}'
            out_path = f'generated_compressor_3D_geometry/{m_dot:.2f}_{int(RPM)}_{pr_original:.2f}_{eta_original:.2f}_design_{design}'
            os.makedirs(out_path, exist_ok=True)
            out_path = f'{out_path}/Main_blade.curve'
            print(f'Generated geometry saved to {out_path}')
            n_profiles = len(geometry[0]) // 512
            print(n_profiles, 'profiles')
            
            print(x_1.shape, x_2.shape, 'The two tensor shapes')
            print('The similarity score:', structural_similarity_index_measure(x_1, x_2).item())
            
            
            
            # write the blade profile curve file
            with open(out_path, "w") as f:
                for i in range(n_profiles): 
                    x_to_store = geometry[0][i*512:(i+1)*512]
                    y_to_store = geometry[1][i*512:(i+1)*512]
                    z_to_store = geometry[2][i*512:(i+1)*512]


                    # ax_3D.plot(x_to_store, y_to_store, z_to_store, markersize=1)
                    # ax.plot(x_to_store, (y_to_store**2 + z_to_store**2)**0.5, markersize=1)



                    if i == 0:
                        x_hub = x_to_store[:250][::-1]
                        r_hub = ((y_to_store[:250][::-1])**2 + (z_to_store[:250][::-1])**2)**0.5

                    elif i == n_profiles-1:
                        
                        x_tip = x_to_store[:247][::-1]
                        r_tip = ((y_to_store[:247][::-1])**2 + (z_to_store[:247][::-1])**2)**0.5

                    profile = np.column_stack((x_to_store, y_to_store, z_to_store))

                    f.write(f"# profile {i+1}\n") 
                    np.savetxt(f, profile, fmt="%.12f")
                    if i != n_profiles- 1:
                        f.write("\n")
            
            # write the hub and shroud curve files
            r_1_tip = geometry_1D['R_tip_1']
            r_2 = geometry_1D['R_mean_2']
            b_2 = geometry_1D['b_2']
            print(geometry_1D['beta_b1_hub'], geometry_1D['beta_b1_tip'])
            print(geometry_1D['R_tip_1'], geometry_1D['b_2'])
            create_hub_curve_file(x_hub, r_hub, r_1_tip, r_2, compressor_code, vaneless_existence = True, pinching = True)
            create_shroud_curve_file(x_tip, r_tip, compressor_code, r_1_tip, r_2, b_2, vaneless_existence = True, pinching = True)


            ax_3D.scatter(geometry[0], geometry[1], geometry[2], s=1)
            ax.scatter(geometry[0], ((geometry[1])**2 + (geometry[2])**2)**0.5, s=1)
            print(f'Number {design} design has blade number of {number_of_blades}.')
        

        if off_design_plot:
                off_design_plot_1D(multiple_design_1D_geometry, m_dot, omega, pr_original, eta_original)
        ax.axis('equal')
        ax.grid(True, ls=':')
        ax.text(0.27, 1.08, fr'$\dot{{m}}$: {m_dot:.2f} Kg/s   '
                f"RPM: {int(RPM)} \n"
                f"PR: {pr_original:.2f}      "
                f"$\eta$: {eta_original:.2f}",
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.65))
        ax.set_xlabel('Axial (mm)')
        ax.set_ylabel('Radial (mm)')
        ax_3D.set_xlabel('X (mm)')
        ax_3D.set_ylabel('Y (mm)')
        ax_3D.set_zlabel('Z (mm)')
        


        save_fig_custom(fig_3D, file_path='fig', file_name=f'3D_view_{m_dot:.2f}_{int(RPM)}_{pr_original:.2f}_{eta_original:.2f}', overwrite=True, dpi = 500)
        save_fig_custom(fig, file_path='fig', file_name=f'Meridional_view_{m_dot:.2f}_{int(RPM)}_{pr_original:.2f}_{eta_original:.2f}', overwrite=True, dpi = 500)
        plt.show()
        



def denoising_plot_3D(mode, device, sample_number, model, aux_model, num_steps, manual_seed, 
                  m_dot, RPM, eta, pr, pca, data_structure, number_of_profiles = None, number_of_points = None):


    model.eval()
    aux_model.eval()

    df, min_max, val_indices = load_1D_dataset()


    df_2 = pd.read_csv('dataset/1D_compressor_geometry.csv')
    df_3 = pd.read_csv('dataset/polar_minmax_per_compressor_normalised.csv')
    df_4 = pd.read_csv('dataset/polar_secondary_minmax.csv')
    

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



    if mode == 'denoise_process_plot_3D':
        numbers = randomly_pick_1D_validation(manual_seed, sample_number)
    else:
        numbers = [0]

    for idx in numbers:


        if mode == 'denoise_process_plot_3D':
            i = val_indices[idx]

            pr_normalised = df.loc[i, 'pressure_ratio']
            eta_normalised = df.loc[i, 'efficiency']
            omega_normalised = df.loc[i, 'omega']
            m_dot_normalised = df.loc[i, 'm_dot']
            compressor_index = df_2.loc[i, 'geometry_index']

            
            row = df_3.loc[df_3['compressor_index'].astype(int) == int(compressor_index)]
            df_3 = row.iloc[0]

            r_min = df_3['r_min']
            r_max = df_3['r_max']
            theta_min = df_3['theta_min']
            theta_max = df_3['theta_max']
            xc_min = df_3['x_min']
            xc_max = df_3['x_max']

            
        else:
            pr_normalised, m_dot_normalised, eta_normalised, omega_normalised = test_condition_1D(m_dot, RPM, pr, eta)

        output_dir = "fig"
        os.makedirs(output_dir, exist_ok=True)


        # The auxiliary model, input 4 output 8
        cond = (torch.tensor([m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised]).to(device))
        # print(m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised)
        rnd = StackedRandomGenerator(device, range(sample_number))
        
        latents = rnd.randn([sample_number, aux_model.in_dim], device=device)
        with torch.no_grad():
            samples, trajectory = edm_sampler(aux_model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
        
        samples = samples.float()
        sample = samples[0].cpu().numpy()
        xc_min = sample[0]
        xc_max = sample[1]
        r_min = sample[2]
        r_max = sample[3]
        theta_min = sample[4]
        theta_max = sample[5]
        n_blades = sample[6]
        n_splitter =  sample[7]

        r_min_denormalised = r_min * (r_min_max - r_min_min) + r_min_min
        r_max_denormalised = r_max * (r_max_max - r_max_min) + r_max_min
        theta_min_denormalised = theta_min * (theta_min_max - theta_min_min) + theta_min_min
        theta_max_denormalised = theta_max * (theta_max_max - theta_max_min) + theta_max_min
        x_min_denormalised = xc_min * (x_min_max - x_min_min) + x_min_min
        x_max_denormalised = xc_max * (x_max_max - x_max_min) + x_max_min



        headers = ["X_min", "X_max", "R_min", "R_max", "Theta_min", "Theta_max", "Main Blade Number", "Splitter Blade Number"]
        frames_number = []
        for i in range(num_steps+1):
            if i <= num_steps-1: 
                samples = trajectory[i]
            else:
                samples = trajectory[-1]
            
            samples = samples.float()
            sample = samples[0].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(23, 2))
            
            ax.axis('off')
            row = [sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7]]
            table = ax.table(cellText=[row], colLabels=headers, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(15)
            table.scale(1, 2.5)

            frame_path_number = f"{output_dir}/{data_structure}_frame_{i}_number.png"
            plt.close(fig)
            save_fig_custom(fig, file_path='fig', file_name=f'{data_structure}_frame_{i}_number', overwrite=True, dpi = 500)
            frames_number.append(frame_path_number)


        gif_path = f"denoising_gif/denoising_process_{data_structure}_{num_steps}_number.gif"
        save_denoising_gif(gif_path, frames_number)




        # The main model
        cond = (torch.tensor([m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised, xc_min, xc_max, r_min, r_max, theta_min, theta_max, n_blades, n_splitter]).to(device))
        # print(m_dot_normalised, omega_normalised,  pr_normalised, eta_normalised, xc_min, xc_max, r_min, r_max, theta_min, theta_max, n_blades, n_splitter)
        rnd = StackedRandomGenerator(device, range(sample_number))
        
        frames = []  # store frame paths
        frames_3D = []
        frames_3D_norm = []
        

        if data_structure == '3D_PCA':
            latents = rnd.randn([sample_number, model.in_dim], device=device)
            with torch.no_grad():
                samples, trajectory = edm_sampler(model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
            
            for i in range(num_steps+1):
                if i <= num_steps-1: 
                    samples = trajectory[i]
                else:
                    samples = trajectory[-1]
                samples = samples.float()
                sample = samples[0].cpu().numpy()
                sample = pca.inverse_transform(sample)
                
                
                r_normalised = sample[1::3]
                theta_normalised = sample[2::3]
                x_normalised = sample[0::3]
            

                r = r_normalised * (r_max_denormalised - r_min_denormalised) + r_min_denormalised 
                theta = theta_normalised * (theta_max_denormalised - theta_min_denormalised) + theta_min_denormalised 
                x = x_normalised * (x_max_denormalised - x_min_denormalised) + x_min_denormalised 

                x, y, z = cyl_to_cart_about_x(x, r, theta)
                r = (y**2 + z**2)**0.5

                fig, ax = plt.subplots()

                ax.scatter(x_normalised, r_normalised, color = 'b', s = 1)
                ax.set_xlabel('Axial')
                ax.set_ylabel('Radial')
                ax.set_xlim(-0.1,1.1)
                ax.set_ylim(-0.1,1.1)
                ax.grid(True, ls=':')
                frame_path = f"{output_dir}/{data_structure}_frame_{i}.png"

                plt.close(fig)
                save_fig_custom(fig, file_path='fig', file_name=f'{data_structure}_frame_{i}', overwrite=True, dpi = 500)
                frames.append(frame_path)
                

                
                fig_3D = plt.figure(figsize=(10, 8))
                ax_3D = fig_3D.add_subplot(111, projection="3d")

                ax_3D.scatter(x, y, z, color = 'b', s = 1)
                ax_3D.set_xlabel('X (mm)')
                ax_3D.set_ylabel('Y (mm)')
                ax_3D.set_zlabel('Z (mm)')
                
                frame_path_3D = f"{output_dir}/{data_structure}_frame_{i}_3D.png"
                plt.close(fig_3D)
                save_fig_custom(fig_3D, file_path='fig', file_name=f'{data_structure}_frame_{i}_3D', overwrite=True, dpi = 500)
                frames_3D.append(frame_path_3D)


                fig_3D_norm = plt.figure(figsize=(10, 8))
                ax_3D_norm = fig_3D_norm.add_subplot(111, projection="3d")

                ax_3D_norm.scatter(x_normalised, r_normalised, theta_normalised, color = 'b', s = 1)
                ax_3D_norm.set_xlabel('X')
                ax_3D_norm.set_ylabel('R')
                ax_3D_norm.set_zlabel('Theta')
                ax_3D_norm.set_xlim(0,1)
                ax_3D_norm.set_ylim(0,1)
                ax_3D_norm.set_zlim(0,1)


                frame_path_3D_norm = f"{output_dir}/{data_structure}_frame_{i}_3D_norm.png"
                plt.close(fig_3D_norm)
                save_fig_custom(fig_3D_norm, file_path='fig', file_name=f'{data_structure}_frame_{i}_3D_norm', overwrite=True, dpi = 500)
                frames_3D_norm.append(frame_path_3D_norm)


        if data_structure == '3D_coordinates':
            latents = torch.randn(1, 1, number_of_profiles, number_of_points*3, device=device)
            with torch.no_grad():
                samples, trajectory = edm_sampler(model, latents=latents, class_labels=cond, randn_like=torch.randn_like, num_steps=num_steps, deterministic=False) 
            
            for i in range(num_steps+1):
                if i <= num_steps-1: 
                    samples = trajectory[i]
                else:
                    samples = trajectory[-1]
                samples = samples.float()
                sample = samples[0].cpu().numpy()
                sample = sample.ravel()
                
                r_normalised = sample[1::3]
                theta_normalised = sample[2::3]
                x_normalised = sample[0::3]
            

                r = r_normalised * (r_max_denormalised - r_min_denormalised) + r_min_denormalised 
                theta = theta_normalised * (theta_max_denormalised - theta_min_denormalised) + theta_min_denormalised 
                x = x_normalised * (x_max_denormalised - x_min_denormalised) + x_min_denormalised 

                x, y, z = cyl_to_cart_about_x(x, r, theta)
                r = (y**2 + z**2)**0.5

                fig, ax = plt.subplots()

                ax.scatter(x_normalised, r_normalised, color = 'b', s = 1)
                ax.set_xlabel('Axial')
                ax.set_ylabel('Radial')
                ax.set_xlim(-0.1,1.1)
                ax.set_ylim(-0.1,1.1)

                ax.grid(True, ls=':')
                frame_path = f"{output_dir}/{data_structure}_frame_{i}.png"

                plt.close(fig)
                save_fig_custom(fig, file_path='fig', file_name=f'{data_structure}_frame_{i}', overwrite=True, dpi = 200)
                frames.append(frame_path)
                

                

                fig_3D = plt.figure(figsize=(10, 8))
                ax_3D = fig_3D.add_subplot(111, projection="3d")
                ax_3D.scatter(x, y, z, color = 'b', s = 1)
                ax_3D.set_xlabel('X (mm)')
                ax_3D.set_ylabel('Y (mm)')
                ax_3D.set_zlabel('Z (mm)')

                frame_path_3D = f"{output_dir}/{data_structure}_frame_{i}_3D.png"

                plt.close(fig_3D)
                
                save_fig_custom(fig_3D, file_path='fig', file_name=f'{data_structure}_frame_{i}_3D', overwrite=True, dpi = 200)
                frames_3D.append(frame_path_3D)
                

                fig_3D_norm = plt.figure(figsize=(10, 8))
                ax_3D_norm = fig_3D_norm.add_subplot(111, projection="3d")

                ax_3D_norm.scatter(x_normalised, r_normalised, theta_normalised, color = 'b', s = 1)
                ax_3D_norm.set_xlabel('X')
                ax_3D_norm.set_ylabel('R')
                ax_3D_norm.set_zlabel('Theta')
                ax_3D_norm.set_xlim(0,1)
                ax_3D_norm.set_ylim(0,1)
                ax_3D_norm.set_zlim(0,1)


                frame_path_3D_norm = f"{output_dir}/{data_structure}_frame_{i}_3D_norm.png"
                plt.close(fig_3D_norm)
                save_fig_custom(fig_3D_norm, file_path='fig', file_name=f'{data_structure}_frame_{i}_3D_norm', overwrite=True, dpi = 200)
                frames_3D_norm.append(frame_path_3D_norm)


    gif_path = f"denoising_gif/denoising_process_{data_structure}_{num_steps}.gif"
    save_denoising_gif(gif_path, frames)

    gif_path = f"denoising_gif/denoising_process_{data_structure}_{num_steps}_3D.gif"
    save_denoising_gif(gif_path, frames_3D)


    gif_path = f"denoising_gif/denoising_process_{data_structure}_{num_steps}_3D_norm.gif"
    save_denoising_gif(gif_path, frames_3D_norm)



def model_deployment(mode, model_config_path, aux_model_config_path = None, sample_number=1, AOA=None, Ma=None, Re=None, CL=None, CD=None, 
                     num_steps = None, fig_size = None, manual_seed = None, lim = 0.5, multiple_design = 1, 
                     x_foil_timeout=20, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100,
                     distribution_plot_switch = False, off_design_plot_switch = False, m_dot = None, RPM = None, 
                     pr = None, eta = None, convert_to_3D = False, plot_blade_distribution = True, smoothening = False):
    
    # ============== Main model ===============
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    data_structure = model_config['data_structure']
    num_epochs = model_config['num_epochs']
    model_channel = model_config['model_channel']
    model_layer = model_config['model_layer']
    model_channel_multiplication = model_config['model_channel_multiplication']
    device=model_config['device']
    nn_structure=model_config['neural_network_sturcture']
    reduced_data_fraction=model_config['reduced_data_fraction']
    model_code = f"{data_structure}_{nn_structure}_{model_channel}_{model_layer}_{len(model_channel_multiplication)}_with_{num_epochs}_epochs_{reduced_data_fraction}_data"
    save_path = f"mdl_weight/{model_code}.pth"
    print('Main Model Code', model_code)
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



    total_params = sum(p.numel() for p in model.parameters())
    print(f"Main model parameters: {total_params:,}")

    total_params_aux = sum(p.numel() for p in aux_model.parameters())
    print(f"Auxiliary model parameters: {total_params_aux:,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)



    if mode == 'validation' or mode == 'test':
        if data_structure == '1D_params':
            validation_1D(mode, device, sample_number, model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,
                        m_dot, RPM, eta, pr, convert_to_3D, plot_blade_distribution)
            
        elif data_structure == '3D_coordinates':
            validation_3D(mode, device, sample_number, model, aux_model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,
                        m_dot, RPM, eta, pr, pca, data_structure, off_design_plot_switch, number_of_profiles, number_of_points, smoothening)

        elif data_structure == '3D_PCA':
            validation_3D(mode, device, sample_number, model, aux_model, num_steps, manual_seed, multiple_design, x_foil_timeout, CL_tolerence, CD_tolerence, max_iteration,
                      m_dot, RPM, eta, pr, pca, data_structure, off_design_plot_switch)

    else:
        if mode == 'denoise_process_plot':
            denoise_process_plot(model, data_structure, device, sample_number, num_steps, pca, fig_size, manual_seed, num_components, num_components)

        elif mode == 'denoise_process_plot_3D':
            
            denoising_plot_3D(mode, device, sample_number, model, aux_model, num_steps, manual_seed, 
                  m_dot, RPM, eta, pr, pca, data_structure, 16, 512)

        else: 
            validation_plot(model, sample_number, data_structure, device, mode, pca, num_steps, manual_seed, 
                        lim, multiple_design, num_components, num_components, x_foil_timeout, CL_tolerence, 
                        CD_tolerence, max_iteration, distribution_plot_switch, off_design_plot_switch, Re, Ma, AOA, CL, CD)






def extrapolation_test_set_generation():
    
    df = pd.read_csv("dataset/aerofoil_data_clean.csv")
    val_indices = np.load("dataset/val_indices_clean.npy")
    train_indices = np.load("dataset/train_indices_clean.npy")

    # df_val = df.loc[val_indices]
    df_train = df
    

    AOA = 3
    Re = 250000
    Ma = 0.2

    df_train = df_train[(df_train['AOA'].astype(int) == int(AOA)) &
                        (df_train['Re'].astype(int) == int(Re)) &
                        (df_train['Ma'].round(1) == round(Ma, 1))]

    # df_val = df_val[(df_val['AOA'].astype(int) == AOA) &
    #                     (df_val['Re'].astype(int) == Re) &
    #                     (df_val['Ma'].round(1) == Ma)]

    interpolation = []
    pareto = []
    extrapolation = []
    intermediate = []


    for _, row in tqdm(df_train.iterrows()):
        CL = row['CL']
        CD = row['CD']

        if CD > 0.01:
            df_cd = df_train[abs(df_train['CD'] - CD) <= 0.001]
        else:
            df_cd = df_train[abs(df_train['CD'] - CD) <= 0.0001]
        
        max_cl = df_cd['CL'].max()

        row = {'AOA': AOA, 'Re': Re, 'Ma': Ma, 'CL': CL, 'CD': CD}
        
        if max_cl <= CL:
            extrapolation.append(row)
        
        # elif max_cl - CL <= 0.03:
        #     pareto.append(row)
        
        # else:
        #     interpolation.append(row)
        #     intermediate.append({'AOA': AOA + random.uniform(-0.5, 0.5), 
        #                         'Re': Re + random.uniform(-50000, 50000), 
        #                         'Ma': Ma + random.uniform(-0.05,  0.05), 
        #                         'CL': CL, 'CD': CD})

    columns = ['AOA', 'Re', 'Ma', 'CL', 'CD']

    # pd.DataFrame(interpolation, columns=columns).to_csv('interpolation_sketch.csv', index=False)

    pd.DataFrame(extrapolation, columns=columns).to_csv('extrapolation_sketch.csv', index=False)

    # pd.DataFrame(pareto, columns=columns).to_csv('pareto_sketch.csv', index=False)

    # pd.DataFrame(intermediate, columns=columns).to_csv('intermediate.csv', index=False)


def visualise_the_extrapolation_testing_set():
    train_indices = np.load("dataset/train_indices_clean.npy")

    extrapolation = pd.read_csv('extrapolation.csv')#.sample(n=100, random_state=123)
    all = pd.read_csv('dataset/aerofoil_data_clean.csv').loc[train_indices]

    # change this line 
    coordinate_results = pd.read_csv('mdl_validation/coordinates_pareto_extrapolation.csv')
    sdf_results = pd.read_csv('mdl_validation/pareto_optimisation_1.05_CL_0.95_CD.csv')
    pca_results = pd.read_csv('mdl_validation/pca_pareto_extrapolation.csv')


    AOA = 3
    Re = 250000
    Ma = 0.2
    all = all[(all['AOA'].astype(int) == AOA) &
                        (all['Re'].astype(int) == Re) &
                        (all['Ma'].round(1) == Ma)]
    

    fig, ax = plt.subplots()

    
    ax.scatter(all['CD'], all['CL'], s=1, color = 'grey', label = 'training data')
    # ax.scatter(interpolation['CD'], interpolation['CL'], s=10, color = 'g', label = 'interpolation')
    ax.scatter(extrapolation['CD'], extrapolation['CL'], s=10, color = 'b', label = 'Pareto front')


    for i in range(10):
        selected_coordinates = coordinate_results[coordinate_results['idx']==i+1]
        

        idx_min_cd = selected_coordinates['CD_actual'].idxmin()
        selected_coordinates = selected_coordinates.drop(idx_min_cd)

        max_row_coordinates = selected_coordinates.loc[selected_coordinates['CL_actual'].idxmax()]

        # max_row = selected.loc[(selected['CL_actual'] / selected['CD_actual']).idxmax()]
        ax.scatter(max_row_coordinates['CD_actual'], max_row_coordinates['CL_actual'], marker = 'o', s=20, color = 'r', label='optimised (coordinates)' if i == 0 else None)
    

        selected_pca = pca_results[pca_results['idx']==i+1]
        max_row_pca = selected_pca.loc[selected_pca['CL_actual'].idxmax()]
        # max_row = selected.loc[(selected['CL_actual'] / selected['CD_actual']).idxmax()]
        ax.scatter(max_row_pca['CD_actual'], max_row_pca['CL_actual'], marker = 'x', s=20, color = 'g', label='optimised (PCA)' if i == 0 else None)
    
        selected_sdf = sdf_results[sdf_results['idx']==i+1]
        max_row_sdf = selected_sdf.loc[selected_sdf['CL_actual'].idxmax()]
        # max_row = selected.loc[(selected['CL_actual'] / selected['CD_actual']).idxmax()]
        ax.scatter(max_row_sdf['CD_actual'], max_row_sdf['CL_actual'], marker = '*', s=20, color = 'orange', label='optimised (SDF)' if i == 0 else None)

    
    # ax.scatter(pareto['CD'], pareto['CL'], s=10, color = 'b', label = 'Pareto front')
    # ax.scatter(intermediate['CD'], intermediate['CL'], s=4, color = 'purple', label = 'intermediate')
    ax.set_xlabel("Drag Coefficient ($C_D$)", fontsize = 14)
    ax.set_ylabel("Lift Coefficient ($C_L$)", fontsize = 14)
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 2)
    ax.legend(loc = 'upper right', fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    save_fig_custom(fig, file_path='fig', file_name=f'aerodynamic_extrapolation_results', overwrite=True, dpi = 500)



def find_the_most_popular_flow_condition():
    all = pd.read_csv('dataset/aerofoil_data_clean.csv')
    AOA_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Ma_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Re_list = [200000, 250000, 300000, 350000]

    count = {}

    for Re in Re_list:
        for Ma in Ma_list:
            for AOA in AOA_list:
                sub_df = all[(all['AOA'].astype(int) == AOA) &
                             (all['Re'].astype(int) == Re) &
                             (all['Ma'].round(1) == Ma)]
                count[f'{Re}_{Ma}_{AOA}'] = len(sub_df)
    # print(count)
    max_item = max(count.items(), key=lambda x: x[1])
    print(max_item)


def find_the_most_popular_flow_condition_1D():
    all = pd.read_csv('dataset/1D_compressor_geometry.csv')
    m_dot_list = np.linspace(0.07, 0.12, 5)
    RPM_list = np.linspace(70000, 90000, 5)

    count = {}

    all['m_dot'] = all['m_dot'].round(2)
    all['omega'] = all['omega'].astype(int)

    for m_dot in m_dot_list:
        for RPM in RPM_list:
            omega = RPM * 2 * np.pi/60
            sub_df = all[(all['m_dot'] == round(m_dot, 2)) &
                         (all['omega'] == int(omega))]
            count[f'{round(m_dot, 2)}_{int(omega)}'] = len(sub_df)

    max_item = max(count.items(), key=lambda x: x[1])
    print(max_item)



def extrapolation_test_set_generation_1D():
    
    df = pd.read_csv("dataset/1D_compressor_geometry.csv")
    val_indices = np.load("dataset/1D_val_indices.npy")
    train_indices = np.load("dataset/1D_train_indices.npy")
    test_indices = np.load("dataset/1D_train_indices.npy")

    df_train = df
    

    m_dot = 0.07
    omega = 7330

    df_train = df_train[(df_train['m_dot'] == round(m_dot, 2)) &
                        (df_train['omega'].astype(int) == int(omega))]

    extrapolation = []

    for _, row in tqdm(df_train.iterrows()):
        pr = row['pressure_ratio']
        eta = row['efficiency']

        df_pr = df_train[abs(df_train['pressure_ratio'] - pr) <= 0.03]
        
        max_eta = df_pr['efficiency'].max()

        row = {'m_dot': m_dot, 'omega': omega, 'pr': pr, 'eta': eta}
        
        if max_eta <= eta:
            extrapolation.append(row)

    columns = ['m_dot', 'omega', 'pr', 'eta']

    pd.DataFrame(extrapolation, columns=columns).to_csv('extrapolation_1D.csv', index=False)



def visualise_the_extrapolation_testing_set_1D():
    train_indices = np.load("dataset/1D_train_indices.npy")

    extrapolation = pd.read_csv('extrapolation_1D.csv')#.sample(n=100, random_state=123)
    all = pd.read_csv('dataset/1D_compressor_geometry.csv').loc[train_indices]

    # change this line 
    param_results = pd.read_csv('mdl_validation/1D_params_ResNet_UNet_64_5_4_with_1000_epochs_extrapolation.csv')


    m_dot = 0.07
    omega = 7330


    all = all[(all['m_dot'] == round(m_dot, 2)) &
              (all['omega'].astype(int) == int(omega))]


    fig, ax = plt.subplots()
    print(len(extrapolation['pr']))
    ax.scatter(all['pressure_ratio'], all['efficiency'], s=1, color = 'grey', label = 'training data')
    ax.scatter(extrapolation['pr'], extrapolation['eta'], s=10, color = 'b', label = 'Pareto front')
    ax.scatter(param_results['pr_actual'], param_results['eta_actual'], marker = 'o', color = 'r', s = 10, label = 'Optimised')
    
  
    ax.set_xlabel("Pressure Ratio", fontsize = 14)
    ax.set_ylabel("Efficiency ($\eta$)", fontsize = 14)
    ax.set_xlim(1, 3)
    ax.set_ylim(0.6, 0.9)
    ax.legend(loc = 'upper right', fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    save_fig_custom(fig, file_path='fig', file_name=f'aerodynamic_extrapolation_results', overwrite=True, dpi = 500)






def pca_accuracy_plot_for_3D(n_components_range):
    
    curve_file = 'dataset/3D_compressor_polar_global_normalised'
    coordinates = []
    explained_variance = []
    for i in range(1512):
        
        try:
            profile, profile_has_error = load_blade_curve(f'{curve_file}/compressor_{i}.curve')
            if not profile_has_error:
                coordinate = np.concatenate(profile).ravel()
            coordinates.append(coordinate)
            
        except FileNotFoundError:
            pass
    coordinates = np.array(coordinates)
    for n_components in n_components_range:
        pca = PCA(n_components=n_components)
        pca_scores = pca.fit_transform(coordinates)
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        explained_variance.append(explained_var)

    fig, ax = plt.subplots()
    ax.plot(n_components_range, explained_variance, color = 'b')
    ax.scatter(n_components_range, explained_variance, color = 'b')
    print(explained_variance)
    ax.set_ylabel('Explained Variance (Accuracy) %', fontsize = 14)
    ax.set_xlabel('Number of Principle Components (N)', fontsize = 14)
    ax.set_xlim(0, 50)
    ax.set_ylim(75, 100)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, ls=':')
    plt.show()





def pca_analysis(n_components=2):
    curve_file = 'dataset/3D_compressor_polar_global_normalised'

    data_matrix = []
    keys = []
    explained_variance = []
    coordinates = []
    for i in range(1513):
        try:
            profile, profile_has_error = load_blade_curve(f'{curve_file}/compressor_{i}.curve')
            if not profile_has_error:
                coordinate = np.concatenate(profile).ravel()
                coordinates.append(coordinate)
                key = f'compressor_{i}'
                keys.append(key)
        except FileNotFoundError:
            pass

    coordinates = np.array(coordinates)

    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(coordinates)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100

    print(f"Explained Variance with {n_components} PCs: {explained_var:.5f}%")

    reconstructed = pca.inverse_transform(pca_scores)

    reconstructed_profile = {}

    for vec, key in zip(reconstructed, keys):
        reconstructed_profile[key] = vec


    return {
        "pca": pca,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance": pca.explained_variance_ratio_,
        "keys": keys,
        'reconstructed': reconstructed_profile,
        'data_matrix': data_matrix,
        'reconstruct': reconstructed
    }





def pca_reconstruct_aerofoil_comparison(component_number):
    
    curve_file = 'dataset/3D_compressor_polar_global_normalised'
    key = 'compressor_111'
    
    original_profile, profile_has_error = load_blade_curve(f'{curve_file}/{key}.curve')

    if profile_has_error:
        print('This profile contains error!')
    coordinate = np.concatenate(original_profile).ravel()
    x_original = coordinate[0::3]
    y_original = coordinate[1::3]
    z_original = coordinate[2::3]

    reconstructed = pca_analysis(n_components=component_number)
    reconstructed = reconstructed['reconstructed'][f'{key}']

    x_reconstructed = reconstructed[0::3]
    y_reconstructed = reconstructed[1::3]
    z_reconstructed = reconstructed[2::3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x_original, y_original, z_original, color = 'b', label = 'Original Profile')
    ax.scatter(x_reconstructed, y_reconstructed , z_reconstructed, color = 'r', linestyle = ':', label = 'PCA Reconstructed Profile')

    ax.grid(ls=':')
    ax.legend(fontsize = 12)
    plt.show()


def swd_pooled_points(blades_real, blades_gen, n_points=50000, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    A = np.asarray(blades_real).reshape(-1, 3).astype(np.float64)
    B = np.asarray(blades_gen).reshape(-1, 3).astype(np.float64)

    if n_points is not None:
        A = A[rng.choice(len(A), size=min(n_points, len(A)), replace=False)]
        B = B[rng.choice(len(B), size=min(n_points, len(B)), replace=False)]

    n = min(len(A), len(B))
    if len(A) != n:
        A = A[rng.choice(len(A), size=n, replace=False)]
    if len(B) != n:
        B = B[rng.choice(len(B), size=n, replace=False)]

    dirs = rng.normal(size=(n_proj, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    total = 0.0
    for v in dirs:
        a = A @ v
        b = B @ v
        a.sort()
        b.sort()
        total += np.mean(np.abs(a - b))

    return float(total / n_proj)




def compare_3D_distribution(mode):

    # fig_3D = plt.figure(figsize=(10, 8))
    # ax_3D = fig_3D.add_subplot(111, projection="3d")
    fig, ax = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    fig_3, ax_3 = plt.subplots()
    cmap = plt.get_cmap("tab10")
    
    if mode == 'individual':
        

        # Model generated distribution
        for compressor_index in range(100):
            color = cmap(compressor_index % 10)
            profiles, error = load_blade_curve(f'generated_compressor_3D_geometry/0.08_85000_3.59_0.80_design_{compressor_index+1}/Main_blade.curve')
            profiles = [profiles[0], profiles[-1]]
            for profile in profiles:
                x = profile[:, 0]
                y = profile[:, 1]
                z = profile[:, 2]

                ax.scatter(x,(y**2+z**2)**0.5, s=1, color = color)
                ax.axis('equal')
                ax.set_ylim(5,50)
        

        # Training data distribution
        df = pd.read_csv('dataset/OLD_NO_USE/1D_compressor_geometry.csv')
        
        target_efficiency = 0.8004911799854328
        target_pressure_ratio =  3.591969416336344

        compressor_index_list =[]

        for _, row in df.iterrows():
            if row['m_dot'] == 0.0825:
                if row['omega'] == 8901.17918517108:
                    if 100*(abs((row['efficiency'] - target_efficiency)/target_efficiency)) < 1:
                        if 100*(abs((row['pressure_ratio'] - target_pressure_ratio)/target_pressure_ratio)) < 1:
                            compressor_index_list.append(row['geometry_index'])
                            


        print(f'There are {len(compressor_index_list)} that can satisfy the condition in the training set.')
        for compressor_index in compressor_index_list:
            profiles, _ = load_blade_curve(f'dataset/OLD_NO_USE/3D_compressor_16_profiles/compressor_{compressor_index}.curve')
            color = cmap(compressor_index % 10)
            profiles = [profiles[0], profiles[-1]]
            for profile in profiles:
                x = profile[:, 0]
                y = profile[:, 1]
                z = profile[:, 2]

                ax_2.scatter(x,(y**2 + z**2)**0.5, s=1, color = color)
                ax_2.axis('equal')
                ax_2.set_ylim(5,50)
        
        
        # Physical distribution
        for compressor_index in range(445):
            color = cmap(compressor_index % 10)
            profiles, error = load_blade_curve(f'dataset/physical_distribution_blades/compressor_{compressor_index+1}.curve')
            profiles = [profiles[0], profiles[-1]]
            for profile in profiles:
                x = profile[:, 0]
                y = profile[:, 1]
                z = profile[:, 2]

                ax_3.scatter(x,(y**2+z**2)**0.5, s=1, color = color)
                ax_3.axis('equal')
                ax_3.set_ylim(5,50)
    
    elif mode == 'heatmap':
        
        def plot_kde(ax, X, R):


            values = np.vstack([X, R])
            kde = gaussian_kde(values)

            xmin, xmax = min(X), max(X)
            rmin, rmax = min(R), max(R)
            print('X min', xmin, 'X max', xmax)
            print('R min', rmin, 'R max', rmax)

            xi, ri = np.mgrid[xmin:xmax:200j, rmin:rmax:200j]
            zi = kde(np.vstack([xi.ravel(), ri.ravel()])).reshape(xi.shape)

            ax.contourf(xi, ri, zi, levels=50, cmap = 'Reds')
            ax.axis('equal')
            ax.set_ylim(5, 50)


        # Model generated
        X1, R1 = [], []
        for compressor_index in range(100):
            profiles, _ = load_blade_curve(f'generated_compressor_3D_geometry/0.08_85000_3.59_0.80_design_{compressor_index+1}/Main_blade.curve')
            profiles = [profiles[0], profiles[-1]]
            for profile in profiles:
                x = profile[:, 0]
                r = np.sqrt(profile[:,1]**2 + profile[:,2]**2)
                X1.append(x); R1.append(r)

        plot_kde(ax, np.concatenate(X1), np.concatenate(R1))


        # Training data
        df = pd.read_csv('dataset/OLD_NO_USE/1D_compressor_geometry.csv')
        
        target_efficiency = 0.8004911799854328
        target_pressure_ratio =  3.591969416336344

        compressor_index_list =[]

        for _, row in df.iterrows():
            if row['m_dot'] == 0.0825:
                if row['omega'] == 8901.17918517108:
                    if 100*(abs((row['efficiency'] - target_efficiency)/target_efficiency)) < 1:
                        if 100*(abs((row['pressure_ratio'] - target_pressure_ratio)/target_pressure_ratio)) < 1:
                            compressor_index_list.append(row['geometry_index'])
                            

        X2, R2 = [], []
        for compressor_index in compressor_index_list:
            profiles, _ = load_blade_curve(f'dataset/OLD_NO_USE/3D_compressor_16_profiles/compressor_{compressor_index}.curve')
            profiles = [profiles[0], profiles[-1]]
            for profile in profiles:
                x = profile[:, 0]
                r = np.sqrt(profile[:,1]**2 + profile[:,2]**2)
                X2.append(x); R2.append(r)

        plot_kde(ax_2, np.concatenate(X2), np.concatenate(R2))


        # Physical
        X3, R3 = [], []
        for compressor_index in range(445):
            profiles, _ = load_blade_curve(f'dataset/physical_distribution_blades/compressor_{compressor_index+1}.curve')
            profiles = [profiles[0], profiles[-1]]
            for profile in profiles:
                x = profile[:, 0]
                r = np.sqrt(profile[:,1]**2 + profile[:,2]**2)
                X3.append(x); R3.append(r)

        plot_kde(ax_3, np.concatenate(X3), np.concatenate(R3))

    
    
    plt.show()


