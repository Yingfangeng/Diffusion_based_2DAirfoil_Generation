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
import imageio
import os
from IPython.display import Image
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tqdm import tqdm
import time
from meanline.meanline import *
import signal


from models.diffusion_model import EDM_CFG, edm_sampler, StackedRandomGenerator



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
    # if data_structure != 'pca':
    #     df = pd.read_csv("dataset/aerofoil_data_normalised_256.csv")
    #     df2 = pd.read_csv("dataset/aerofoil_data_256.csv")
    #     val_indices = np.load("dataset/val_indices_256.npy")
    #     train_indices = np.load("dataset/train_indices_256.npy")
    #     min_max = pd.read_csv('dataset/min_max_256.csv')


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

    # CL_error = (CL - CL_actual)/CL
    # CD_error = (CD - CD_actual)/CD

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
                print('Outlier in CL')
    CL_rmse = (CL_sum_of_square/len(CL_error))**0.5
        

    for idx, i in enumerate(CD_error):
        if CD[idx] >= CD_axis_min and CD[idx] <= CD_axis_max: 
            if abs(i) <= upper_band_grad-1:
                CD_in_bound += 1
            
            if abs(i) < 1:
                CD_sum_of_square = CD_sum_of_square + i**2
            else:
                print('Outlier in CD')
    CD_rmse = (CD_sum_of_square/len(CD_error))**0.5
    

    print(f'The validation takes {len(results)} samples, among which {unfeasible} ({100*(unfeasible/len(results)):.2f}%) designs are unfeasible after 10 trials.')
    print(f'The accuracy information of the {len(CL)} feasible designs are shown below:')
    print('CL RMSE is:', CL_rmse)
    print('CD RMSE is:', CD_rmse)
    print(f'{int(100*(CL_in_bound)/(len(CL)))}% samples have CL within {int(band_1*100)}% relative error.')
    print(f'{int(100*(CD_in_bound)/(len(CL)))}% samples have CD within {int(band_1*100)}% relative error.')

    print(f'Averaged number of trials {np.average(total_design_number):.2f}, averaged percent of unfeasible designs {int(100*(np.sum(unfeasible_design_number))/(np.sum(total_design_number)))}%.')

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
    ax1.set_xlabel('Target $C_L$', fontsize = 14)
    ax1.set_ylabel('Generated $C_L$', fontsize = 14)
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
    ax2.set_xlabel('Target $C_D$', fontsize = 14)
    ax2.set_ylabel('Generated $C_D$', fontsize = 14)
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

    print("CL best-fit-line gradient is:", m_cl)
    print("CD best-fit-line gradient is:", m_cd)
    save_fig_custom(fig, file_path='fig', file_name=f'single_target_accuracy_plot_{mode}_{data_structure}', 
                    format_list=['.eps', '.png'], overwrite=True, dpi = 500)


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
        return meanline, True

    except TimeoutException:
        print("Take too long for meanline to converge!")
        return None, False

    except Exception as e:
        signal.alarm(0)
        print('Meanline not converged')
        return None, False








def validation_1D(device, sample_size, model, num_steps):
    
    
    df = pd.read_csv('dataset/1D_compressor_geometry_normalised.csv')
    min_max = pd.read_csv('dataset/1D_compressor_geometry_minmax.csv')

    if ("min" in min_max.columns) and ("max" in min_max.columns):
        feature_col = min_max.columns[0]  # usually "Unnamed: 0"
        if feature_col not in ["min", "max"]:
            min_max = min_max.set_index(feature_col)

    # Clean any whitespace issues in feature names
    min_max.index = min_max.index.astype(str).str.strip()

    i = 234
    
    
    pr_normalised = df.loc[i, 'pressure_ratio']
    eta_normalised = df.loc[i, 'efficiency']
    omega_normalised = df.loc[i, 'omega']
    m_dot_normalised = df.loc[i, 'm_dot']

    number_of_trials = 0
    success = False
    while not success and number_of_trials < 10:
        cond = (torch.tensor([omega_normalised, m_dot_normalised, pr_normalised, eta_normalised]).to(device))
        
        rnd = StackedRandomGenerator(device, range(sample_size))
        latents = rnd.randn([sample_size, model.in_dim], device=device)


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
            'R_mean_2': float(denormalised_geometry[6]), 
            'beta_b2': float(denormalised_geometry[7]),
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



        m_dot = m_dot_normalised*(min_max.loc['m_dot', 'max']- min_max.loc['m_dot', 'min']) + min_max.loc['m_dot', 'min']
        omega_original = omega_normalised*(min_max.loc['omega', 'max']- min_max.loc['omega', 'min']) + min_max.loc['omega', 'min']

        PR_original = pr_normalised*(min_max.loc['pressure_ratio', 'max']- min_max.loc['pressure_ratio', 'min']) + min_max.loc['pressure_ratio', 'min']
        eta_original = eta_normalised*(min_max.loc['efficiency', 'max']- min_max.loc['efficiency', 'min']) + min_max.loc['efficiency', 'min']

        omega = (omega_original*60)/(2*np.pi)

        meanline, success = run_meanline(geometry, m_dot, omega)

        number_of_trials += 1

    if success:
        stage_pressure_ratio = meanline.pressure_ratio
        stage_efficiency = meanline.stage_eff
        work_coefficient = meanline.psi
        print(f'mass flow rate: {m_dot} kg/s, RPM: {omega}')
        print(f'Generated design pressure ratio: {stage_pressure_ratio}, efficiency {stage_efficiency}.' )
        print(f'Original design pressure ratio: {PR_original}, efficiency {eta_original}.')
    else:
        print(f'Cannot generate feasible geometry after {number_of_trials}.')






def model_deployment(mode, model_config_path, sample_number=1, AOA=None, Ma=None, Re=None, CL=None, CD=None, 
                     num_steps = None, fig_size = None, manual_seed = None, lim = 0.5, multiple_design = 1, 
                     x_foil_timeout=20, CL_tolerence = 0.01, CD_tolerence = 0.05, max_iteration = 100,
                     distribution_plot_switch = False, off_design_plot_switch = False):
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
        if mode == 'denoise_process_plot':
            denoise_process_plot(model, data_structure, device, sample_number, num_steps, pca, fig_size, manual_seed, num_components, num_components)

        else: 
            validation_plot(model, sample_number, data_structure, device, mode, pca, num_steps, manual_seed, 
                        lim, multiple_design, num_components, num_components, x_foil_timeout, CL_tolerence, 
                        CD_tolerence, max_iteration, distribution_plot_switch, off_design_plot_switch, Re, Ma, AOA, CL, CD)
    
    elif data_structure == '1D_params':
        validation_1D(device, sample_number, model, num_steps)




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
    sdf_results = pd.read_csv('mdl_validation/sdf_pareto_extrapolation.csv')
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


    for i in range(20):
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