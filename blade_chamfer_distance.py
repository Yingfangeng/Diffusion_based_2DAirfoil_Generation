import numpy as np
from tqdm import tqdm
import os
import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from sklearn.neighbors import KernelDensity


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





def compute_chamfer_distance_and_ssim(A, B):
    final_distance = 0.0
    for a in A:
        distance_list = []
        for b in B:
            distance = np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
            distance_list.append(distance)

        b = B[int(np.argmin(distance_list))]

        a_t = torch.from_numpy(a.reshape(16, 512, 3)).permute(2, 0, 1).unsqueeze(0).float()
        b_t = torch.from_numpy(b.reshape(16, 512, 3)).permute(2, 0, 1).unsqueeze(0).float()

        # mn = torch.min(torch.min(a_t), torch.min(b_t))
        # mx = torch.max(torch.max(a_t), torch.max(b_t))
        # a_t = (a_t - mn) / (mx - mn)
        # b_t = (b_t - mn) / (mx - mn)

        final_distance_now = float(structural_similarity_index_measure(a_t, b_t))
        # print(final_distance_now)
        # if final_distance_now < 0.9:
        #     print('Extreme case not covered')

        final_distance += final_distance_now
    return final_distance / len(A)



def kernel_density_estimate_3D(X):

    X = np.concatenate(X, axis=0)

    bandwidth = 1.0
    grid_res = 30

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)

    
    x_min, y_min, z_min = 0, -50, -50
    x_max, y_max, z_max = 35, 35, 40

    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    zs = np.linspace(z_min, z_max, grid_res)

    xx, yy, zz = np.meshgrid(xs, ys, zs)
    grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    log_density = kde.score_samples(grid)
    density = np.exp(log_density)

    return density



def kde_distribution_difference(P, Q):
    P = P / np.sum(P)
    Q = Q / np.sum(Q)


    M = 0.5 * (P + Q)

    KL_PM = np.sum(P * np.log((P + 1e-12) / (M + 1e-12)))
    KL_QM = np.sum(Q * np.log((Q + 1e-12) / (M + 1e-12)))

    JS = 0.5 * (KL_PM + KL_QM)

    return JS




def distribution_compare():


    physical_distribution_path = "dataset/physical_distribution"
    physical_distribution_files = sorted(os.listdir(physical_distribution_path))  # ensure deterministic order

    physical_distribution_file_selected_1 = physical_distribution_files[150:250]
    physical_distribution_file_selected_2 = physical_distribution_files[250:350]
    # print(f'There are {len(physical_distribution_file_selected_1)} physical generated profiles.')

    model_generated_distribution_3D_path = "generated_compressor_3D_geometry/model_generated_distribution_3D"
    model_generated_distribution_3D_files = sorted(os.listdir(model_generated_distribution_3D_path))
    # print(f'There are {len(model_generated_distribution_3D_files)} 3D model generated profiles.')

    model_generated_distribution_1D_path = "dataset/model_generated_distribution_1D"
    model_generated_distribution_1D_files = sorted(os.listdir(model_generated_distribution_1D_path))
    # print(f'There are {len(model_generated_distribution_1D_files)} 1D model generated profiles.')

    model_generated_distribution_3D = []
    model_generated_distribution_1D = []
  
    physical_distribution_1 = []
    physical_distribution_2 = []
    
    for file in physical_distribution_file_selected_1:
        profiles, _ = load_blade_curve(f'dataset/physical_distribution/{file}')
        physical_distribution_1.append(np.concatenate(profiles, axis=0).astype(np.float64))        



    for file in physical_distribution_file_selected_2:
        profiles, _ = load_blade_curve(f'dataset/physical_distribution/{file}')
        physical_distribution_2.append(np.concatenate(profiles, axis=0).astype(np.float64))


    for file in model_generated_distribution_3D_files:
        profiles, _ = load_blade_curve(f'generated_compressor_3D_geometry/model_generated_distribution_3D/{file}/Main_blade.curve')
        model_generated_distribution_3D.append(np.concatenate(profiles, axis=0).astype(np.float64))

    for file in model_generated_distribution_1D_files:
        profiles, _ = load_blade_curve(f'dataset/model_generated_distribution_1D/{file}')
        model_generated_distribution_1D.append(np.concatenate(profiles, axis=0).astype(np.float64))


    assert len(physical_distribution_1) == len(physical_distribution_2)

    
    physical_distribution_density = kernel_density_estimate_3D(physical_distribution_1)
    physical_distribution_density_2 = kernel_density_estimate_3D(physical_distribution_2)
    model_generated_distribution_density_3D = kernel_density_estimate_3D(model_generated_distribution_3D)
    model_generated_distribution_density_1D = kernel_density_estimate_3D(model_generated_distribution_1D)
    difference = kde_distribution_difference(physical_distribution_density, model_generated_distribution_density_3D)
    print('3D and physical', difference)
    difference = kde_distribution_difference(physical_distribution_density, model_generated_distribution_density_1D)
    print('1D and physical', difference)
    difference = kde_distribution_difference(physical_distribution_density, physical_distribution_density_2)
    print('two physical', difference)

    # final_distance_1 = compute_chamfer_distance_and_ssim(model_generated_distribution_3D, physical_distribution_1)
    # final_distance_2 = compute_chamfer_distance_and_ssim(physical_distribution_1, model_generated_distribution_3D)
    # print("The SSIM between the model generated blade distribution and the physical distribution") 
    # print(final_distance_1, final_distance_2, np.mean([final_distance_1, final_distance_2]))

    # final_distance_1 = compute_chamfer_distance_and_ssim(model_generated_distribution_1D, physical_distribution_1)
    # final_distance_2 = compute_chamfer_distance_and_ssim(physical_distribution_1, model_generated_distribution_1D)
    # print("The SSIM between two physical distributions")
    # print(final_distance_1, final_distance_2, np.mean([final_distance_1, final_distance_2]))


    # final_distance_1 = compute_chamfer_distance_and_ssim(physical_distribution_1, physical_distribution_2)
    # final_distance_2 = compute_chamfer_distance_and_ssim(physical_distribution_2, physical_distribution_1)
    # print("The SSIM between two physical distributions")
    # print(final_distance_1, final_distance_2, np.mean([final_distance_1, final_distance_2]))



    # final_distance_1 = compute_chamfer_distance(model_generated_distribution_3D, physical_distribution_1)
    # final_distance_2 = compute_chamfer_distance(physical_distribution_1, model_generated_distribution_3D)
    # print("The CD between the 3D model generated blade distribution and the physical distribution") 
    # print(final_distance_1, final_distance_2, np.mean([final_distance_1, final_distance_2]))


    # final_distance_1 = compute_chamfer_distance(model_generated_distribution_1D, physical_distribution_1)
    # final_distance_2 = compute_chamfer_distance(physical_distribution_1, model_generated_distribution_1D)
    # print("The CD between the 1D model generated blade distribution and the physical distribution") 
    # print(final_distance_1, final_distance_2, np.mean([final_distance_1, final_distance_2]))

    # final_distance_1 = compute_chamfer_distance(physical_distribution_1, physical_distribution_2)
    # final_distance_2 = compute_chamfer_distance(physical_distribution_2, physical_distribution_1)
    # print("The CD between two physical distributions")
    # print(final_distance_1, final_distance_2, np.mean([final_distance_1, final_distance_2]))



if __name__ == '__main__':
    distribution_compare()