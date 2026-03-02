import ot
import numpy as np
from tqdm import tqdm
import os
import random


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



def compute_chamfer_distance(A, B):
    final_distance = 0
    for a in tqdm(A):
        distance_list = []
        for b in B:
            distance = np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
            distance_list.append(distance)
        final_distance += min(distance_list)
    return final_distance / len(A)


def distribution_compare():


    directory = "dataset/physical_distribution_reversed"

    all_files = sorted(os.listdir(directory))  # ensure deterministic order

    selected_files = all_files[:100]
    selected_files_2 = all_files[250:350]


    model_generated_distribution = []
    physical_distribution = []
    physical_distribution_2 = []
    
    for file in selected_files:
        try: 
            profiles, _ = load_blade_curve(f'dataset/physical_distribution_reversed/{file}')
            physical_distribution.append(np.concatenate(profiles, axis=0).astype(np.float64))        
        except FileNotFoundError:
            print('NOT FOUND')
    

    for file in selected_files_2:
        try: 
            profiles, _ = load_blade_curve(f'dataset/physical_distribution_reversed/{file}')
            physical_distribution_2.append(np.concatenate(profiles, axis=0).astype(np.float64))        
        except FileNotFoundError:
            print('NOT FOUND')

    for compressor_index in range(100):
        profiles, _ = load_blade_curve(f'generated_compressor_3D_geometry/model_generated_distribution/0.08_85000_3.59_0.80_design_{compressor_index+1}/Main_blade.curve')
        model_generated_distribution.append(np.concatenate(profiles, axis=0).astype(np.float64))

    assert len(physical_distribution) == len(physical_distribution_2)


    final_distance_1 = compute_chamfer_distance(model_generated_distribution, physical_distribution)
    final_distance_2 = compute_chamfer_distance(physical_distribution, model_generated_distribution)
    print("The chamfer distances between the model generated blade distribution and the physical distribution", np.mean([final_distance_1, final_distance_2]))


    final_distance_1 = compute_chamfer_distance(physical_distribution, physical_distribution_2)
    final_distance_2 = compute_chamfer_distance(physical_distribution_2, physical_distribution)

    print("The chamfer distances between two physical distributions", np.mean([final_distance_1, final_distance_2]))


if __name__ == '__main__':
    distribution_compare()