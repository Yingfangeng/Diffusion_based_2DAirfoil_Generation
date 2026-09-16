import ot
import numpy as np
from tqdm import tqdm


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



def distribution_compare():
    model_generated_distribution = []
    physical_distribution = []
    
    for compressor_index in range(5): #107
        try: 
            profiles, _ = load_blade_curve(f'dataset/physical_distribution_reversed/compressor_{compressor_index+1}.curve')
            physical_distribution.append(np.concatenate(profiles, axis=0).astype(np.float64))
        except FileNotFoundError:
            print('NOT FOUND')
    
    for compressor_index in range(5):
        profiles, _ = load_blade_curve(f'generated_compressor_3D_geometry/model_generated_distribution/0.08_85000_3.59_0.80_design_{compressor_index+1}/Main_blade.curve')
        model_generated_distribution.append(np.concatenate(profiles, axis=0).astype(np.float64))

    physical_distribution = np.concatenate(physical_distribution, axis=0).astype(np.float64)
    model_generated_distribution = np.concatenate(model_generated_distribution, axis=0).astype(np.float64)


    def chamfer_distance(A, B):
        
        overall_distance = 0
        for a in tqdm(A):
            distance_list = []
            for b in B:
                distance = ((a[0]-b[0])**2 + (a[1]-b[1])**2 +(a[2]-b[2])**2)**0.5
                distance_list.append(distance)

            shortest_distance = min(distance_list)
            overall_distance += shortest_distance

            
        return overall_distance


    N = len(model_generated_distribution)
    M = len(physical_distribution)
    print(N, M, 'number of points')

    CD_1 = chamfer_distance(model_generated_distribution, physical_distribution)
    CD_2 = chamfer_distance(physical_distribution, model_generated_distribution)
    print("The chamfer distances are", CD_1, CD_2)


if __name__ == '__main__':
    distribution_compare()