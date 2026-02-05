import os
import numpy as np
import pandas as pd
import math
import shutil
import subprocess
from scipy import stats

from .generate_blade_dat_files import CentrifugalCompressor
from .generate_blade_dat_files import create_directories, create_main_blade_files, create_splitter_blade_files,  adjust_splitter_blades_position, main_and_splitter_optimization, compute_rake_angles, find_blade_te_points, validate_te_points
from .convert_csv_to_curve_files import Convert_surfaces_csv_to_curve_files



def read_curve_file(file_path, expected_cols=3):
    """
    Reads the .curve file into a NumPy array (N,3)
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != expected_cols:
            raise ValueError(f"Line '{line}' in {file_path} does not have {expected_cols} columns.")
        data.append(list(map(float, parts)))
    return np.array(data)






def clean_blade_surface_files(ps_file, ss_file, le_file=None, method="dbscan"):
    """
    Automatically cleans blade surface CSV files by detecting and removing outlier points.
    Only processes the pressure side (PS) and suction side (SS) files, leaving the leading edge (LE) file untouched.
    The original files will be overwritten with the cleaned versions.
    Metadata is accurately updated to reflect the cleaned data.
    """

    # Install necessary packages if not already installed
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("Installing scikit-learn for DBSCAN clustering...")
        import pip
        pip.main(['install', 'scikit-learn'])
        from sklearn.cluster import DBSCAN
    
    cleaned_files = []
    
    # Process only PS and SS files
    for file_path, surface_name in [(ps_file, "Pressure Side"), 
                                   (ss_file, "Suction Side")]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        # Create temporary path for the cleaned file
        temp_path = f"{file_path}.temp"
        
        # Read the file data
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # First line is metadata
        metadata_line = lines[0].strip()
        
        # Parse metadata
        try:
            metadata_parts = metadata_line.split(',')
            if len(metadata_parts) == 2:
                points_per_profile = int(metadata_parts[0])
                num_profiles = int(metadata_parts[1])
                
                print(f"Original metadata: {points_per_profile} points per profile, {num_profiles} profiles")
                has_valid_metadata = True
            else:
                print(f"Unexpected metadata format: {metadata_line}")
                has_valid_metadata = False
        except:
            print(f"Could not parse metadata: {metadata_line}")
            has_valid_metadata = False
            points_per_profile = 0
            num_profiles = 0
        
        # Parse points, keeping track of which profile each point belongs to
        profiles = []
        current_profile = []
        point_count = 0
        
        for line in lines[1:]:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            try:
                values = [float(val) for val in line_stripped.split() if val.strip()]
                if len(values) >= 3:
                    point = values[:3]  # Take first 3 values (x,y,z)
                    
                    # Track which profile this point belongs to
                    if has_valid_metadata:
                        if point_count % points_per_profile == 0 and point_count > 0:
                            profiles.append(np.array(current_profile))
                            current_profile = []
                        current_profile.append(point)
                        point_count += 1
                    else:
                        # If we don't have valid metadata, just collect all points
                        current_profile.append(point)
            except ValueError:
                # Skip lines that can't be parsed as numbers
                continue
        
        # Add the last profile if it's not empty
        if current_profile:
            profiles.append(np.array(current_profile))
        
        if not profiles:
            print(f"No valid points found in {file_path}")
            continue
        
        # Apply outlier detection to each profile separately
        cleaned_profiles = []
        total_original_points = 0
        total_kept_points = 0
        
        for profile_idx, profile in enumerate(profiles):
            original_count = len(profile)
            total_original_points += original_count
            
            # Apply outlier detection to this profile
            if len(profile) > 10:  # Only apply to profiles with enough points
                mask = detect_outliers(profile, method)
                cleaned_profile = profile[mask]
            else:
                # For small profiles, keep all points
                cleaned_profile = profile
                mask = np.ones(len(profile), dtype=bool)
            
            kept_count = len(cleaned_profile)
            total_kept_points += kept_count
            
            # Only add non-empty profiles
            if kept_count > 0:
                cleaned_profiles.append(cleaned_profile)
            
            print(f"Profile {profile_idx+1}: Kept {kept_count}/{original_count} points")
        
        # Only proceed with overwriting if outliers were found
        if total_kept_points < total_original_points:
            # Prepare new metadata
            new_num_profiles = len(cleaned_profiles)
            
            if has_valid_metadata and cleaned_profiles:
                # Try to preserve points_per_profile if possible
                new_points_per_profile = max(len(profile) for profile in cleaned_profiles)
                
                # Update points in each profile to have the same length
                for i in range(len(cleaned_profiles)):
                    if len(cleaned_profiles[i]) < new_points_per_profile:
                        # Pad with duplicate points to match new_points_per_profile
                        last_point = cleaned_profiles[i][-1]
                        padding = np.tile(last_point, (new_points_per_profile - len(cleaned_profiles[i]), 1))
                        cleaned_profiles[i] = np.vstack([cleaned_profiles[i], padding])
            else:
                # If we don't have valid original metadata, make a best guess
                new_points_per_profile = max(len(profile) for profile in cleaned_profiles)
            
            # Write filtered data to temporary file
            with open(temp_path, 'w') as f:
                # Write updated metadata
                new_metadata = f"{new_points_per_profile},{new_num_profiles}\n"
                f.write(new_metadata)
                
                # Write points from cleaned profiles
                for profile in cleaned_profiles:
                    for point in profile:
                        # Format with consistent spacing
                        formatted_line = f"  {point[0]:15.12f}   {point[1]:15.12f}    {point[2]:15.12f}\n"
                        f.write(formatted_line)
            
            # Replace original file with temporary file
            shutil.move(temp_path, file_path)
            print(f"Cleaned {surface_name}: Removed {total_original_points - total_kept_points} out of {total_original_points} points")
            print(f"Updated metadata: {new_points_per_profile} points per profile, {new_num_profiles} profiles")
        else:
            print(f"No outliers detected in {surface_name}")
        
        cleaned_files.append(file_path)
    
    # Add the leading edge file to the list of returned files, even though we didn't process it
    if le_file and os.path.exists(le_file):
        cleaned_files.append(le_file)
        print(f"Leading Edge file left unchanged: {le_file}")
    
    return cleaned_files




def detect_outliers(points, method="dbscan"):
    """
    Detect points that are part of the main blade structure (not outliers).
    """

    # Handle empty or single-point arrays
    if len(points) <= 1:
        return np.ones(len(points), dtype=bool)
    
    if method == "iqr":
        # IQR (Interquartile Range) method
        # Calculate IQR for each dimension
        q1 = np.percentile(points, 25, axis=0)
        q3 = np.percentile(points, 75, axis=0)
        iqr = q3 - q1
        
        # Define bounds with typical factor of 1.5
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Create a mask for points within bounds in all dimensions
        mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
        return mask
        
    elif method == "zscore":
        # Z-score method (identifies points that are within 3 standard deviations)
        z_scores = np.abs(stats.zscore(points, axis=0))
        mask = np.all(z_scores < 3, axis=1)
        return mask
        
    elif method == "dbscan":
        # DBSCAN clustering (identifies dense clusters as non-outliers)
        from sklearn.cluster import DBSCAN
        
        # Normalize data for better clustering
        points_normalized = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
        
        # Run DBSCAN with appropriate epsilon and min_samples
        # For smaller point sets, reduce min_samples
        min_samples = min(10, max(3, len(points) // 10))
        db = DBSCAN(eps=0.5, min_samples=min_samples).fit(points_normalized)
        
        # Find the largest cluster
        labels = db.labels_
        if len(set(labels)) > 1:  # If there's more than just outliers
            # Count points in each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            # Exclude noise points (label -1)
            if -1 in unique_labels:
                noise_idx = np.where(unique_labels == -1)[0][0]
                unique_labels = np.delete(unique_labels, noise_idx)
                counts = np.delete(counts, noise_idx)
            
            if len(counts) > 0:
                # Get the label of the largest cluster
                largest_cluster = unique_labels[np.argmax(counts)]
                # Keep points in the largest cluster
                return labels == largest_cluster
        
        # Default: just exclude DBSCAN's identified noise points
        return labels != -1
    
    else:
        # Default: keep all points
        return np.ones(len(points), dtype=bool)












class Blade_Forming_3D:


    def __init__(self, splitter_existence, target_rake_angle, compressor_path, geometry, vaneless_diff_existence, pinching_ratio, hub_percentage_splitter = 0.2, extreme_value_for_imp_inlet = -40):
        
        self.splitter_existence = splitter_existence
        self.vaneless_diff_existence = vaneless_diff_existence
        self.target_rake_angle = target_rake_angle
        self.compressor_path = compressor_path
        self.geometry = geometry

        self.pinching_ratio = pinching_ratio

        self.curve_directory = f"{self.compressor_path}/3D_blades_{self.pinching_ratio:.2f}"

        self.method_meridional = 'opt'
        self.method_angles = 'bezier'
        self.method_thickness = 'custom'
        
        self.hub_percentage_splitter = hub_percentage_splitter
        self.extreme_value_for_imp_inlet = extreme_value_for_imp_inlet
        self.vaneless_diff_existence = vaneless_diff_existence
        
        if pinching_ratio>0:
            self.pinching = True
        else:
            self.pinching = False 

        



    def execute_allblades(self, folder_paths, exe_name):
        """
        Function to execute the blade forming software
        """

        all_succeeded = True
        for folder in folder_paths:
            exe_path = os.path.join(folder, exe_name)
            exe_path = os.path.join('/home/yg1922/Desktop/Yingfan_FYP_Code/Diffusion_based_2DAirfoil_Generation', exe_path)
            
            try:
                subprocess.run(["wine", exe_path], cwd=folder, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Executed {exe_name} successfully in {folder}.")
            except subprocess.CalledProcessError as e:
                print(f"Execution failed in {folder}: {e}")
                all_succeeded = False
            except FileNotFoundError:
                print(f"{exe_name} not found in the folder {folder}.")
                all_succeeded = False

        return all_succeeded



    def execute(self, convert_to_3D):
        
        # initial lean angle
        lean_theta_value = 0

        # inputs from meanline - convert the lengths to mm
        R_hub_1 = self.geometry['R_hub_1']*1000  # impeller inlet hub radius
        R_tip_1 = self.geometry['R_tip_1']*1000  # impeller inlet tip radius
        L_z = self.geometry['L_z']*1000   # axial length
        R_2 = self.geometry['R_mean_2']*1000  # impeller outlet radius
        b_2 = self.geometry['b_2']*1000   # blade width in impeller outlet
        t = self.geometry['t']*1000   # blade thickness

        # get the number of full and splitter blades
        n_blades = self.geometry['nblades']
        n_splitter_blades = self.geometry['n_splitter_blades']
        n_full_blades = n_blades - n_splitter_blades

        # Example calculations for beta_b1_hub, beta_b1_tip, beta_b2
        beta_b1_hub = abs(self.geometry['beta_b1_hub']*180/math.pi)
        beta_b1_tip = abs(self.geometry['beta_b1_tip']*180/math.pi)
        beta_b2 = abs(self.geometry['beta_b2']*180/math.pi)

        
        if self.vaneless_diff_existence == True:
            print(self.geometry)
            b_3 = self.geometry['b3']*1000
            r_3 = self.geometry['r3']*1000
        else:
            b_3 = 0
            r_3 =0

        compressor = CentrifugalCompressor(R_hub_1, R_tip_1, R_2, L_z, b_2, beta_b1_hub, beta_b1_tip, beta_b2, t, self.method_meridional, self.method_angles, self.method_thickness, self.hub_percentage_splitter, self.splitter_existence, n_blades)
        compressor.execute()

        ##### for main blade #####

        # for meridional - main
        x_hub_meridional_main = compressor.return_geometry_lists()['hub_x_meridional_main']
        y_hub_meridional_main = compressor.return_geometry_lists()['hub_y_meridional_main']
        x_shroud_meridional_main = compressor.return_geometry_lists()['shroud_x_meridional_main']
        y_shroud_meridional_main = compressor.return_geometry_lists()['shroud_y_meridional_main']
        x_leading_meridional_main = compressor.return_geometry_lists()['leading_x_meridional_main']
        y_leading_meridional_main = compressor.return_geometry_lists()['leading_y_meridional_main']
        x_trailing_meridional_main = compressor.return_geometry_lists()['trailing_x_meridional_main']
        y_trailing_meridional_main = compressor.return_geometry_lists()['trailing_y_meridional_main']

        # for blade angle - main
        x_hub_blade_main_angle = compressor.return_geometry_lists()['blade_x_hub_main']
        theta_hub_blade_main = compressor.return_geometry_lists()['blade_theta_hub_main']
        x_shroud_blade_main_angle = compressor.return_geometry_lists()['blade_x_shroud_main']
        theta_shroud_blade_main = compressor.return_geometry_lists()['blade_theta_shroud_main']

        # for blade thickness - main
        x_hub_thickness_main = compressor.return_geometry_lists()['thickness_x_hub_main']
        thickness_hub_main = compressor.return_geometry_lists()['thickness_hub_main']
        x_shroud_thickness_main = compressor.return_geometry_lists()['thickness_x_shroud_main']
        thickness_shroud_main = compressor.return_geometry_lists()['thickness_shroud_main']


        if self.splitter_existence == True:

            ##### for splitter blade #####

            # for meridional - splitter
            x_hub_meridional_splitter = compressor.return_geometry_lists()['hub_x_meridional_splitter']
            y_hub_meridional_splitter = compressor.return_geometry_lists()['hub_y_meridional_splitter']
            x_shroud_meridional_splitter = compressor.return_geometry_lists()['shroud_x_meridional_splitter']
            y_shroud_meridional_splitter = compressor.return_geometry_lists()['shroud_y_meridional_splitter']
            x_leading_meridional_splitter = compressor.return_geometry_lists()['leading_x_meridional_splitter']
            y_leading_meridional_splitter = compressor.return_geometry_lists()['leading_y_meridional_splitter']
            x_trailing_meridional_splitter = compressor.return_geometry_lists()['trailing_x_meridional_splitter']
            y_trailing_meridional_splitter = compressor.return_geometry_lists()['trailing_y_meridional_splitter']

            # for blade angle - splitter
            x_hub_blade_splitter_angle = compressor.return_geometry_lists()['blade_x_hub_splitter']
            theta_hub_blade_splitter = compressor.return_geometry_lists()['blade_theta_hub_splitter']
            x_shroud_blade_splitter_angle = compressor.return_geometry_lists()['blade_x_shroud_splitter']
            theta_shroud_blade_splitter = compressor.return_geometry_lists()['blade_theta_shroud_splitter']

            # for blade thickness - splitter
            x_hub_thickness_splitter = compressor.return_geometry_lists()['thickness_x_hub_splitter']
            thickness_hub_splitter = compressor.return_geometry_lists()['thickness_hub_splitter']
            x_shroud_thickness_splitter = compressor.return_geometry_lists()['thickness_x_shroud_splitter']
            thickness_shroud_splitter = compressor.return_geometry_lists()['thickness_shroud_splitter']

        # geometry_path = os.path.join(self.compressor_path,'3D_geometry')
        geometry_path = self.compressor_path



        # create the directories
        main_blades_folder, splitter_blades_folder = create_directories(geometry_path)

        # Generate the dat files for the main and splitter blades in the respective directories
        create_main_blade_files(main_blades_folder,x_hub_meridional_main, y_hub_meridional_main, x_shroud_meridional_main, y_shroud_meridional_main, x_leading_meridional_main, y_leading_meridional_main, x_trailing_meridional_main, y_trailing_meridional_main, x_hub_blade_main_angle, theta_hub_blade_main, x_shroud_blade_main_angle, theta_shroud_blade_main, x_hub_thickness_main, thickness_hub_main, x_shroud_thickness_main, thickness_shroud_main, n_full_blades,lean_theta_value)

        if self.splitter_existence == True:
            create_splitter_blade_files(splitter_blades_folder,x_hub_meridional_splitter, y_hub_meridional_splitter, x_shroud_meridional_splitter, y_shroud_meridional_splitter, x_leading_meridional_splitter, y_leading_meridional_splitter, x_trailing_meridional_splitter, y_trailing_meridional_splitter, x_hub_blade_splitter_angle, theta_hub_blade_splitter, x_shroud_blade_splitter_angle, theta_shroud_blade_splitter, x_hub_thickness_splitter, thickness_hub_splitter, x_shroud_thickness_splitter, thickness_shroud_splitter, n_splitter_blades,lean_theta_value)
        
            # Copy the allblades executable
        if self.splitter_existence:
            shutil.copy('/home/yg1922/Desktop/Yingfan_FYP_Code/Diffusion_based_2DAirfoil_Generation/meanline/allblades_20240530.exe', os.path.join(geometry_path, 'Main_Blades'))
            shutil.copy('/home/yg1922/Desktop/Yingfan_FYP_Code/Diffusion_based_2DAirfoil_Generation/meanline/allblades_20240530.exe', os.path.join(geometry_path, 'Splitter_Blades'))
        else:
            shutil.copy('/home/yg1922/Desktop/Yingfan_FYP_Code/Diffusion_based_2DAirfoil_Generation/meanline/allblades_20240530.exe', os.path.join(geometry_path, 'Main_Blades'))

        exe_name = 'allblades_20240530.exe'
        folder_paths = [os.path.join(geometry_path, 'Main_Blades')]
        if self.splitter_existence:
            folder_paths.append(os.path.join(geometry_path, 'Splitter_Blades'))

        # Adaptive thickness control loop
        success_main = False
        success_splitter = False
        attempt_main = 0
        attempt_splitter = 0
        max_attempts = 50
        if convert_to_3D:
            while (not success_main or not success_splitter) and (attempt_main + attempt_splitter) < max_attempts:
                # Get current geometry
                geometry = compressor.return_geometry_lists()
                
                # Generate dat files for main blades
                create_main_blade_files(main_blades_folder,
                    geometry['hub_x_meridional_main'], geometry['hub_y_meridional_main'],
                    geometry['shroud_x_meridional_main'], geometry['shroud_y_meridional_main'],
                    geometry['leading_x_meridional_main'], geometry['leading_y_meridional_main'],
                    geometry['trailing_x_meridional_main'], geometry['trailing_y_meridional_main'],
                    geometry['blade_x_hub_main'], geometry['blade_theta_hub_main'],
                    geometry['blade_x_shroud_main'], geometry['blade_theta_shroud_main'],
                    geometry['thickness_x_hub_main'], geometry['thickness_hub_main'],
                    geometry['thickness_x_shroud_main'], geometry['thickness_shroud_main'],
                    n_full_blades, lean_theta_value)

                # Generate dat files for splitter blades if they exist
                if self.splitter_existence:
                    create_splitter_blade_files(splitter_blades_folder,
                        geometry['hub_x_meridional_splitter'], geometry['hub_y_meridional_splitter'],
                        geometry['shroud_x_meridional_splitter'], geometry['shroud_y_meridional_splitter'],
                        geometry['leading_x_meridional_splitter'], geometry['leading_y_meridional_splitter'],
                        geometry['trailing_x_meridional_splitter'], geometry['trailing_y_meridional_splitter'],
                        geometry['blade_x_hub_splitter'], geometry['blade_theta_hub_splitter'],
                        geometry['blade_x_shroud_splitter'], geometry['blade_theta_shroud_splitter'],
                        geometry['thickness_x_hub_splitter'], geometry['thickness_hub_splitter'],
                        geometry['thickness_x_shroud_splitter'], geometry['thickness_shroud_splitter'],
                        n_splitter_blades, lean_theta_value)

                # Execute allblades for main blades
                

                if not success_main:
                    success_main = self.execute_allblades([os.path.join(geometry_path, 'Main_Blades')], exe_name)
                    if not success_main:
                        print(f"Main blade attempt {attempt_main + 1} failed. Adjusting thickness factors...")
                        compressor.smart_adjust_thickness_factors(attempt_main, 'main')
                        attempt_main += 1
                
                # Execute allblades for splitter blades
                if self.splitter_existence and not success_splitter:
                    success_splitter = self.execute_allblades([os.path.join(geometry_path, 'Splitter_Blades')], exe_name)
                    if not success_splitter:
                        print(f"Splitter blade attempt {attempt_splitter + 1} failed. Adjusting thickness factors...")
                        compressor.smart_adjust_thickness_factors(attempt_splitter, 'splitter')
                        attempt_splitter += 1

                if success_main and (success_splitter or not self.splitter_existence):
                    print("Allblades executed successfully for all blades.")

                    if self.splitter_existence:
                        print('###############################################')
                        print('INITIAL SPLITTER POSITION ADJUSTMENT')
                        print('###############################################')
                        splitter_angle, adjustment, new_adjustment = adjust_splitter_blades_position(main_blades_folder, splitter_blades_folder, n_full_blades, n_splitter_blades)
                        success_splitter = self.execute_allblades([os.path.join(geometry_path, 'Splitter_Blades')], exe_name)
                        if not success_splitter:
                            print("Failed to execute allblades after initial splitter position adjustment")
                            attempt_splitter += 1
                            continue

                    print('###############################################')
                    print('OPTIMIZING RAKE ANGLES')
                    print('###############################################')
                    
                    convert_object = Convert_surfaces_csv_to_curve_files(geometry_path, self.splitter_existence, self.extreme_value_for_imp_inlet, self.vaneless_diff_existence, b_2, b_3, r_3, self.pinching, self.pinching_ratio, self.curve_directory)
                    success = main_and_splitter_optimization(geometry_path, exe_name, R_hub_1, R_2, b_2, self.splitter_existence, self.hub_percentage_splitter, self.target_rake_angle, self.curve_directory, convert_object)
                    
                    # Initialize rake angles
                    main_rake = None
                    splitter_rake = None

                    if success:
                        if self.splitter_existence:
                            print('###############################################')
                            print('FINAL SPLITTER POSITION VERIFICATION')
                            print('###############################################')
                            splitter_angle, adjustment, new_adjustment = adjust_splitter_blades_position(main_blades_folder, splitter_blades_folder, n_full_blades, n_splitter_blades)
                            success_splitter = self.execute_allblades([os.path.join(geometry_path, 'Splitter_Blades')], exe_name)
                            if not success_splitter:
                                print("Failed to execute allblades after final splitter position adjustment")
                                attempt_splitter += 1
                                continue

                            # Calculate final rake angles
                            convert_object = Convert_surfaces_csv_to_curve_files(geometry_path, self.splitter_existence, self.extreme_value_for_imp_inlet, self.vaneless_diff_existence, b_2, b_3, r_3, self.pinching, self.pinching_ratio, self.curve_directory)
                            convert_object.create_curve_files()

                            hub_file = os.path.join(self.curve_directory,'HubOriginal.curve')
                            shroud_file = os.path.join(self.curve_directory,'ShroudOriginal.curve')
                            splitter_blade_file = os.path.join(self.curve_directory,'BladeSplitter.curve')

                            hub_data = read_curve_file(hub_file, expected_cols=3)
                            shroud_data = read_curve_file(shroud_file, expected_cols=3)
                            splitter_blade_data = read_curve_file(splitter_blade_file, expected_cols=3)
                            split_hub_te, split_shroud_te = find_blade_te_points(hub_data, shroud_data, splitter_blade_data, radius_tol=0.05)
                            
                            is_valid, invalid_value = validate_te_points(split_hub_te, split_shroud_te)
        
                            if not is_valid:
                                splitter_rake = invalid_value  # -1000 for invalid configuration
                            else:
                                splitter_rake = compute_rake_angles(split_hub_te, split_shroud_te)
                            

                            print(f'Final splitter rake angle: {splitter_rake:.2f}°')
                        
                        convert_object = Convert_surfaces_csv_to_curve_files(geometry_path, self.splitter_existence, self.extreme_value_for_imp_inlet, self.vaneless_diff_existence, b_2, b_3, r_3, self.pinching, self.pinching_ratio, self.curve_directory)
                        convert_object.create_curve_files()

                        hub_file = os.path.join(self.curve_directory,'HubOriginal.curve')
                        shroud_file = os.path.join(self.curve_directory,'ShroudOriginal.curve')
                        main_blade_file = os.path.join(self.curve_directory,'BladeMain.curve')

                        hub_data = read_curve_file(hub_file, expected_cols=3)
                        shroud_data = read_curve_file(shroud_file, expected_cols=3)
                        main_blade_data = read_curve_file(main_blade_file, expected_cols=3)

                        main_hub_te, main_shroud_te = find_blade_te_points(hub_data, shroud_data, main_blade_data, radius_tol=0.05)
                        
                        is_valid, invalid_value = validate_te_points(main_hub_te, main_shroud_te)
        
                        if not is_valid:
                            main_rake = invalid_value  # -1000 for invalid configuration
                        else:
                            main_rake = compute_rake_angles(main_hub_te, main_shroud_te)


                        print(f'Final main blade rake angle: {main_rake:.2f}°')

                        break
                    else:
                        print("Failed during rake angle optimization")
                        attempt_main += 1
                        if self.splitter_existence:
                            attempt_splitter += 1


            if success_main and (success_splitter or not self.splitter_existence):
                print(f"Successfully executed with splitter position and lean angle adjustments")
                
                # Convert CSV files to curve files
                convert_object = Convert_surfaces_csv_to_curve_files(geometry_path, self.splitter_existence, self.extreme_value_for_imp_inlet, self.vaneless_diff_existence, b_2, b_3, r_3, self.pinching, self.pinching_ratio, self.curve_directory)
                convert_object.create_curve_files()
                
                print("Geometry generation, splitter position adjustment, lean angle adjustment, and conversion completed successfully.")

                if self.splitter_existence:
                    print('splitter rake:',splitter_rake)
                print('main rake:',main_rake)


            else:
                print(f"Failed to execute allblades successfully after {attempt_main + attempt_splitter} total attempts.")
                print("Main blade success:", success_main)
                if self.splitter_existence:
                    print("Splitter blade success:", success_splitter)
                print("Final thickness factors:", compressor.return_thickness_factors())



        exe_name = 'allblades_20240530.exe'
        cleanup_folders = [os.path.join(geometry_path, 'Main_Blades')]
        if self.splitter_existence:
            cleanup_folders.append(os.path.join(geometry_path, 'Splitter_Blades'))

        for folder in cleanup_folders:
            exe_file = os.path.join(folder, exe_name)
            try:
                if os.path.isfile(exe_file):
                    os.remove(exe_file)
                    print(f"Deleted executable: {exe_file}")
            except Exception as e:
                print(f"WARNING: could not delete {exe_file}: {e}")