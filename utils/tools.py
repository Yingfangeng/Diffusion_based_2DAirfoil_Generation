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
import ot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvista as pv


# Timer to control the time to 
class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()




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
            fig.savefig(file_name_now, facecolor='white', bbox_inches="tight", dpi=dpi, transparent=True)
        else:
            fig.savefig(file_name_now, facecolor='white', bbox_inches="tight", transparent=True)

        
        print(file_name_now, 'is saved.')
    
    return


def string_coords_to_list(x_string, y_string):

    x_string = re.sub(r'[\[\]]', '', x_string)   # remove brackets
    x_list = [float(v) for v in x_string.split()]
    y_string = re.sub(r'[\[\]]', '', y_string)
    y_list = np.array([float(v) for v in y_string.split()])

    return x_list, y_list



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
