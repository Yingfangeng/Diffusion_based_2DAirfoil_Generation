import io
from PIL import Image
import os
import pickle
from tqdm import tqdm
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor


from models.mlp import CFGResNet
from einops import repeat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
 
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    # Simple tensor test
    x = torch.rand(3, 3).to("cuda")
    y = torch.rand(3, 3).to("cuda")
    z = x @ y
    print("Matrix multiplication successful on GPU!")
    print(z)
else:
    print("⚠️ CUDA not available — check your PyTorch or driver setup.")