import os
import cma
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import minimize
import itertools
import shutil
import subprocess
from scipy import stats
import platform
import signal
from sklearn.cluster import DBSCAN
from matplotlib import rcParams

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_directories(geometry_path):
    """
    Create the directories for the blade folders
    """

    # Create the main geometry folder if it doesn't exist
    if not os.path.exists(geometry_path):
        os.makedirs(geometry_path)

    # Define the subfolders
    main_blades_path = os.path.join(geometry_path, 'Main_Blades')
    splitter_blades_path = os.path.join(geometry_path, 'Splitter_Blades')

    # Create the subfolders if they don't exist
    if not os.path.exists(main_blades_path):
        os.makedirs(main_blades_path)
    
    if not os.path.exists(splitter_blades_path):
        os.makedirs(splitter_blades_path)

    # Return the paths to the subfolders
    return main_blades_path, splitter_blades_path






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

    cleaned_files = []
    
    # Process only PS and SS files
    for file_path, surface_name in [(ps_file, "Pressure Side"), (ss_file, "Suction Side")]:
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




def create_beta_dat(folder, x_hub_angle, theta_hub_angle, x_shroud_angle, theta_shroud_angle):
    """
    Create the file including the blade angle distribution
    """
    
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, 'beta.dat')

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the header
        f.write('meridional, hub angle, casing angle, (add more)\n')

        # Write the specific line with 0 0 1
        f.write('0 0 1\n')
        
        # Write the data rows
        for x_hub, theta_hub, theta_shroud in zip(x_hub_angle, theta_hub_angle, theta_shroud_angle):
            f.write(f"{x_hub} {theta_hub} {theta_shroud}\n")



def create_casing_dat(folder, x_casing, y_casing):
    """
    Create the file including the casing coordinates
    """

    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, 'casing.dat')

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the data rows
        for x, y in zip(x_casing, y_casing):
            f.write(f"{x} {y}\n")




def create_general_dat(folder, num_blades, main_or_splitter):
    """
    Creates a 'general.dat' file in the specified folder with general blade information.
    """
    
    # Define the file path
    file_path = os.path.join(folder, 'general.dat')
    
    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the header
        f.write('general information: axial/radial(1,2),nblade, cut trailing edge or not (1 for yes), eccentricity for leading edge, eccentricity for trailing edge, if using linear theta, if spliter blade\n')
        
        # Define the second line values
        axial_radial = 2
        cut_trailing_edge = 1
        ecc_leading_edge = 1.000
        ecc_trailing_edge = 1.000
        
        # Determine the last three values based on main_or_splitter
        if main_or_splitter == 0:
            splitter_flag = '2 1 0'
        else:
            splitter_flag = '1 0 -2'
        
        # Write the general data
        f.write(f'{axial_radial} {num_blades} {cut_trailing_edge} {ecc_leading_edge:.3f} {ecc_trailing_edge:.3f} {splitter_flag}\n')




def create_hub_dat(folder, x_hub, y_hub):
    """
    Create the file with the hub coordinates
    """
    
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, 'hub.dat')

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the data rows
        for x, y in zip(x_hub, y_hub):
            f.write(f"{x} {y}\n")




def create_le_lean_dat(folder):
    """
    Creates a 'le_lean.dat' file in the specified folder with predefined leading edge lean data
    """
    
    # Define the file path
    file_path = os.path.join(folder, 'le_lean.dat')
    
    # Define the content to write
    le_lean_content = [
        "0 6.375",
        "-0.2 12.10057123",
        "-0.4 15.91761872",
        "-0.666715774 19.1"
    ]
    
    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write each line of content
        for line in le_lean_content:
            f.write(line + '\n')





def create_le_dat(folder, x_leading, y_leading):
    """
    Create the file with the leadng edge coordinates
    """
    
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, 'le.dat')

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the data rows
        for x, y in zip(x_leading, y_leading):
            f.write(f"{x} {y}\n")





def create_lean_theta_dat(folder,lean_theta_value):
    """
    Create the file with the lean angle
    """

    # Define the file path
    file_path = os.path.join(folder, 'lean_theta.dat')

    # Open the file for writing
    with open(file_path, 'w') as f:
        
        f.write(f"{lean_theta_value}\n")
        



def create_te_dat(folder, x_trailing, y_trailing):
    """
    Create the file with the coordinates of the trailing edge   
    """

    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, 'te.dat')

    # Reverse the lists
    x_trailing = x_trailing[::-1]
    y_trailing = y_trailing[::-1]

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the data rows
        for x, y in zip(x_trailing, y_trailing):
            f.write(f"{x} {y}\n")




def create_thickness_dat(folder, x_thickness_hub, thickness_hub, x_thickness_shroud, thickness_shroud):
    
    # Define the file path
    file_path = os.path.join(folder, 'thickness.dat')

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Write the header
        f.write('meridional, hub angle, casing angle, (add more)\n')

        # Write the specific line with 0 0 1
        f.write('0 0 1\n')
        
        # Write the data rows
        for x_hub, theta_hub, theta_shroud in zip(x_thickness_hub, thickness_hub, thickness_shroud):
            f.write(f"{x_hub} {theta_hub} {theta_shroud}\n")




def create_main_blade_files(main_blades_folder,x_hub_meridional_main, y_hub_meridional_main, x_shroud_meridional_main, y_shroud_meridional_main, x_leading_meridional_main, y_leading_meridional_main, x_trailing_meridional_main, y_trailing_meridional_main, x_hub_blade_main_angle, theta_hub_blade_main, x_shroud_blade_main_angle, theta_shroud_blade_main, x_hub_thickness_main, thickness_hub_main, x_shroud_thickness_main, thickness_shroud_main,n_full_blades,lean_theta_value):
    """
    Create all the .dat files in the main blade directrory
    """

    create_beta_dat(main_blades_folder, x_hub_blade_main_angle, theta_hub_blade_main, x_shroud_blade_main_angle, theta_shroud_blade_main)
    create_casing_dat(main_blades_folder, x_shroud_meridional_main, y_shroud_meridional_main)
    create_general_dat(main_blades_folder, n_full_blades,0)
    create_hub_dat(main_blades_folder, x_hub_meridional_main, y_hub_meridional_main)
    create_le_lean_dat(main_blades_folder)
    create_le_dat(main_blades_folder, x_leading_meridional_main, y_leading_meridional_main)
    create_lean_theta_dat(main_blades_folder,lean_theta_value)
    create_te_dat(main_blades_folder,x_trailing_meridional_main, y_trailing_meridional_main)
    create_thickness_dat(main_blades_folder, x_hub_thickness_main, thickness_hub_main, x_shroud_thickness_main, thickness_shroud_main)
   





def create_splitter_blade_files(splitter_blades_folder,x_hub_meridional_splitter, y_hub_meridional_splitter, x_shroud_meridional_splitter, y_shroud_meridional_splitter, x_leading_meridional_splitter, y_leading_meridional_splitter, x_trailing_meridional_splitter, y_trailing_meridional_splitter, x_hub_blade_splitter_angle, theta_hub_blade_splitter, x_shroud_blade_splitter_angle, theta_shroud_blade_splitter, x_hub_thickness_splitter, thickness_hub_splitter, x_shroud_thickness_splitter, thickness_shroud_splitter, n_splitter_blades,lean_theta_value):
    
    meridional_splitter_angle = np.linspace(0, 1, len(theta_hub_blade_splitter), endpoint=True).tolist()
    
    create_beta_dat(splitter_blades_folder, meridional_splitter_angle, theta_hub_blade_splitter, x_shroud_blade_splitter_angle, theta_shroud_blade_splitter)
    create_casing_dat(splitter_blades_folder, x_shroud_meridional_splitter, y_shroud_meridional_splitter)
    create_general_dat(splitter_blades_folder, n_splitter_blades,1)
    create_hub_dat(splitter_blades_folder, x_hub_meridional_splitter, y_hub_meridional_splitter)
    create_le_lean_dat(splitter_blades_folder)
    create_le_dat(splitter_blades_folder, x_leading_meridional_splitter, y_leading_meridional_splitter)
    create_lean_theta_dat(splitter_blades_folder,lean_theta_value)
    create_te_dat(splitter_blades_folder,x_trailing_meridional_splitter, y_trailing_meridional_splitter)

    meridional_splitter_thickness = np.linspace(0, 1, len(thickness_hub_splitter), endpoint=True).tolist()
    create_thickness_dat(splitter_blades_folder, meridional_splitter_thickness, thickness_hub_splitter, x_shroud_thickness_splitter, thickness_shroud_splitter)









def adjust_splitter_blades_position(main_blades_folder, splitter_blades_folder, n_full_blades, n_splitter_blades):
    
    def get_circumferential_angle(y, z):
        """
        Calculate blade angle in a consistent way
        """
        
        angle = np.degrees(np.arctan2(z, y))
        angle = angle % 360
        return angle
    

    def calculate_min_angular_difference(angle1, angle2):
        """
        Calculate minimum angle between two angles
        """

        diff = (angle1 - angle2) % 360
        if diff > 180:
            diff = diff - 360
        return diff


    def read_ps_surface_file(filename):
        print(f"\nReading file: {filename}")
        with open(filename, 'r') as f:
            points_per_profile, num_profiles = map(int, f.readline().strip().split(','))
            print(f"File header - Points per profile: {points_per_profile}, Number of profiles: {num_profiles}")
            points = [list(map(float, line.strip().split())) for line in f]
        
        profiles = np.array(points).reshape(num_profiles, points_per_profile, 3)
        hub_profile = profiles[0]
        print(f"Hub profile shape: {hub_profile.shape}")
        
        # Find trailing edge point
        te_point = max(hub_profile, key=lambda p: p[0])
        print(f"Trailing edge point coordinates (x,y,z): {te_point}")
        return te_point

    print("\n=== Starting Splitter Position Adjustment ===")
    print(f"Number of full blades: {n_full_blades}")
    print(f"Number of splitter blades: {n_splitter_blades}")

    # Read main blade ps_surface.csv
    main_ps_file = os.path.join(main_blades_folder, 'ps_surface.csv')
    print("\nProcessing main blade")
    main_x, main_y, main_z = read_ps_surface_file(main_ps_file)
    main_angle = get_circumferential_angle(main_y, main_z)
    
    print(f"Main blade angle: {main_angle}°")

    # Read splitter blade ps_surface.csv
    splitter_ps_file = os.path.join(splitter_blades_folder, 'ps_surface.csv')
    print("\nProcessing splitter blade")
    splitter_x, splitter_y, splitter_z = read_ps_surface_file(splitter_ps_file)
    splitter_angle = get_circumferential_angle(splitter_y, splitter_z)
    print(f"Splitter blade angle: {splitter_angle}°")

    # Calculate desired splitter angle
    # Calculate desired position
    angular_spacing = 360 / n_full_blades
    desired_angle = (main_angle + angular_spacing/2) % 360
    adjustment = calculate_min_angular_difference(desired_angle, splitter_angle)
    
    print("\n=== Position Calculations ===")
    print(f"Angular spacing between main blades: {angular_spacing}°")
    print(f"Desired splitter angle: {desired_angle}°")
    print(f"Normalized adjustment: {adjustment}°")

    # Update the general.dat file
    general_file = os.path.join(splitter_blades_folder, 'general.dat')
    print(f"\nUpdating general.dat file: {general_file}")
    
    with open(general_file, 'r') as f:
        lines = f.readlines()
        print("Current general.dat content:")
        for i, line in enumerate(lines):
            print(f"Line {i}: {line.strip()}")
    
    parts = lines[1].split()
    current_adjustment = float(parts[-1])
    print(f"\nCurrent adjustment in general.dat: {current_adjustment}")
    
    new_adjustment = current_adjustment + adjustment
    parts[-1] = f"{new_adjustment:.6f}"
    lines[1] = " ".join(parts) + "\n"
    
    print(f"New adjustment to be written: {new_adjustment:.6f}")
    
    with open(general_file, 'w') as f:
        f.writelines(lines)
        print("\ngeneral.dat file updated successfully")

    print("\n=== Final Summary ===")
    print(f"Main blade angle: {main_angle:.2f}°")
    print(f"Original splitter angle: {splitter_angle:.2f}°")
    print(f"Desired splitter angle: {desired_angle:.2f}°")
    print(f"Applied adjustment: {adjustment:.2f}°")
    print(f"Final adjustment in general.dat: {new_adjustment:.6f}")

    return splitter_angle, adjustment, new_adjustment





def run_external_tool(folder, exe_name, timeout=30):
    """
    Runs an external executable tool in a given folder using a new process group so that if
    the execution exceeds the timeout all spawned processes are terminated.
    """

    exe_path = os.path.join(folder, exe_name)
    try:
        # Start the process in a new process group
        if platform.system() == "Windows":
            proc = subprocess.Popen(exe_path, cwd=folder, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(["wine", exe_path], cwd=folder, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for the process to complete within the timeout period
        proc.wait(timeout=timeout)
        print(f"Executed {exe_name} successfully in {folder}.")
        return True
    
    except subprocess.TimeoutExpired as e:
        print(f"Execution timed out in {folder} after {timeout} seconds: {e}")
        
        if platform.system() == "Windows":
            # Sends CTRL_BREAK_EVENT to the new process group
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.kill()
        return False
    except Exception as e:
        print(f"Execution failed in {folder}: {e}")
        return False



def read_ps_surface(file_path):
    with open(file_path, 'r') as f:
        points_per_profile = int(f.readline().strip().split(',')[0])
        lines = f.readlines()
    
    profiles = []
    for i in range(0, len(lines), points_per_profile):
        profile = np.array([list(map(float, line.split())) for line in lines[i:i+points_per_profile]])
        profiles.append(profile)
    
    return profiles





def find_blade_te_points(hub_data, shroud_data, blade_data, radius_tol=0.05):
    """
    Attempt to find trailing edge points on the blade that are near the hub or shroud surfaces.
    
    1. For each x in the blade data, find the corresponding hub radius or shroud radius
       at (approximately) the same x, if it exists.
    2. Compare the blade's yz-radius to the hub or shroud yz-radius at that x.
    3. Keep only blade points where |radius_blade - radius_hub_or_shroud| < radius_tol
    4. Among those points, pick the one with largest x as candidate TE.
    """

    def build_x_radius_map(curve_data):
        x_radius_map = {}
        for pt in curve_data:
            x_val = round(pt[0], 3)
            r_val = np.sqrt(pt[1]**2 + pt[2]**2)
            if x_val not in x_radius_map:
                x_radius_map[x_val] = r_val
            else:
                x_radius_map[x_val] = max(x_radius_map[x_val], r_val)
        return x_radius_map

    hub_map = build_x_radius_map(hub_data)
    shroud_map = build_x_radius_map(shroud_data)

    hub_candidates = []
    shroud_candidates = []

    for pt in blade_data:
        x_val = round(pt[0], 3)
        blade_r = np.sqrt(pt[1]**2 + pt[2]**2)
        
        if x_val in hub_map:
            hub_r = hub_map[x_val]
            if abs(blade_r - hub_r) < radius_tol:
                hub_candidates.append(pt)
        
        if x_val in shroud_map:
            shroud_r = shroud_map[x_val]
            if abs(blade_r - shroud_r) < radius_tol:
                shroud_candidates.append(pt)
    
    hub_te_blade = None
    shroud_te_blade = None

    if hub_candidates:
        hub_candidates = np.array(hub_candidates)
        idx_max_x = np.argmax(hub_candidates[:,0])
        hub_te_blade = hub_candidates[idx_max_x]
    
    if shroud_candidates:
        shroud_candidates = np.array(shroud_candidates)
        idx_max_x = np.argmax(shroud_candidates[:,0])
        shroud_te_blade = shroud_candidates[idx_max_x]

    return hub_te_blade, shroud_te_blade







def compute_rake_angles(hub_te, shroud_te):
    """
    Signed rake angle calculation:
    - Positive rake: When hub TE is more negative in Y than shroud TE (dy > 0)
    - Negative rake: When hub TE is more positive in Y than shroud TE (dy < 0)
    
    Implementation:
     1) angle_3D = angle between the TE vector and +x (0..180)
     2) rake_magnitude = 180 - angle_3D
     3) sign determined by dy (positive if hub TE is more negative in Y)
    """
    
    if hub_te is None or shroud_te is None:
        return None

    v_te = shroud_te - hub_te   # [dx, dy, dz]
    dx, dy, dz = v_te
    v_norm = np.linalg.norm(v_te)
    if v_norm < 1e-9:
        return None  # degenerate

    # Step 1) 3D angle from +x
    angle_3d = np.degrees(np.arccos(dx / v_norm))  # 0..180
    # Step 2) Rake magnitude
    rake_magnitude = 180.0 - angle_3d

    # Step 3) Determine sign based on dy:
    # Positive rake if hub TE is more negative in Y (dy > 0)
    # Negative rake if hub TE is more positive in Y (dy < 0)
    sign = +1 if dy > 0 else -1
    
    return sign * rake_magnitude



def validate_te_points(hub_te, shroud_te, min_separation=0.0001, min_dy=0.0001):
    """
    Validates TE points configuration.
    Returns:
    - (True, None) if valid
    - (False, -1000) if invalid
    
    Validation criteria:
    1. Both points must exist
    2. Points must be sufficiently separated
    3. Must have non-zero dy component
    """
    
    # Check if points exist
    if hub_te is None or shroud_te is None:
        return False, -1000
        
    # Calculate separations
    v_te = shroud_te - hub_te
    dx, dy, dz = v_te
    separation = np.linalg.norm(v_te)
    
    # Check minimum separation
    if separation < min_separation:
        print(f"Warning: TE points too close: {separation:.3f}mm < {min_separation}mm")
        return False, -1000
        
    # Check for zero or very small dy
    if abs(dy) < min_dy:
        print(f"Warning: Near-zero dy component: {abs(dy):.3f}mm < {min_dy}mm")
        return False, -1000
        
    return True, None


def read_initial_lean_angle(blade_folder):
    """
    Read the initial lean angle from the lean_theta.dat file in the given blade folder.
    """

    lean_theta_file = os.path.join(blade_folder, 'lean_theta.dat')
    with open(lean_theta_file, 'r') as f:
        initial_lean_angle = float(f.read().strip())
    return initial_lean_angle




def optimize_lean_angle(blade_folder, main_splitter_mode, target_rake_angle, r_hub_1, r_2, b_2, initial_lean_angle, exe_name, curve_directory, convert_to_csv_object, tolerance=0.1, max_iterations=50):
    """
    Optimizes the lean angle to achieve a target rake angle using an adaptive search
    that can handle steep sign changes and boundary cases.
    """
    
    convert_to_csv_object.create_curve_files()
   

    def objective_function(lean_angle):
        """
        Calculate the difference between the actual rake angle and the target rake angle.
        Returns (error, actual_rake_angle) tuple.
        """
        lean_theta_file = os.path.join(blade_folder, 'lean_theta.dat')
        with open(lean_theta_file, 'w') as f:
            f.write(f"{lean_angle:.6f}")
        
        # Run external tool with timeout (e.g., 10 seconds)
        success = run_external_tool(blade_folder, exe_name, timeout=25)
        if not success:
            return float('inf'), float('inf')
        
        convert_to_csv_object.create_curve_files()
        
        # (Rest of your code that reads the curve files and computes the rake angle)
        if main_splitter_mode == 0:
            hub_file = os.path.join(curve_directory, 'HubOriginal.curve')
            shroud_file = os.path.join(curve_directory, 'ShroudOriginal.curve')
            main_blade_file = os.path.join(curve_directory, 'BladeMain.curve')

            hub_data = read_curve_file(hub_file, expected_cols=3)
            shroud_data = read_curve_file(shroud_file, expected_cols=3)
            main_blade_data = read_curve_file(main_blade_file, expected_cols=3)

            main_hub_te, main_shroud_te = find_blade_te_points(hub_data, shroud_data, main_blade_data, radius_tol=0.05)
            is_valid, invalid_value = validate_te_points(main_hub_te, main_shroud_te)
            if not is_valid:
                actual_rake_angle = invalid_value  # -1000 for invalid configuration
            else:
                actual_rake_angle = compute_rake_angles(main_hub_te, main_shroud_te)

        elif main_splitter_mode == 1:
            hub_file = os.path.join(curve_directory, 'HubOriginal.curve')
            shroud_file = os.path.join(curve_directory, 'ShroudOriginal.curve')
            splitter_blade_file = os.path.join(curve_directory, 'BladeSplitter.curve')

            hub_data = read_curve_file(hub_file, expected_cols=3)
            shroud_data = read_curve_file(shroud_file, expected_cols=3)
            splitter_blade_data = read_curve_file(splitter_blade_file, expected_cols=3)

            split_hub_te, split_shroud_te = find_blade_te_points(hub_data, shroud_data, splitter_blade_data, radius_tol=0.05)
            is_valid, invalid_value = validate_te_points(split_hub_te, split_shroud_te)
            if not is_valid:
                actual_rake_angle = invalid_value  # -1000 for invalid configuration
            else:
                actual_rake_angle = compute_rake_angles(split_hub_te, split_shroud_te)

        return actual_rake_angle - target_rake_angle, actual_rake_angle








    def adaptive_search(low, high):
        """
        Adaptive search that can handle steep sign changes and boundary cases.
        Key features:
        1. Tracks all evaluations to identify trends
        2. Expands bounds when necessary
        3. Uses sampling to handle difficult regions
        4. Can restart search in different regions if needed
        """

        # Store all evaluated points
        all_points = []
        
        # Track best solution found
        best_error = float('inf')
        best_lean = None
        best_rake = None
        
        # Track number of consecutive iterations with minimal improvement
        stalled_iterations = 0
        full_range_searches = 0  # Count how many times we've done a broad search
        
        # Initial bounds
        current_low = low
        current_high = high
        
        # Initial evaluations
        low_error, low_rake = objective_function(low)
        high_error, high_rake = objective_function(high)
        
        all_points.append((low, low_error, low_rake))
        all_points.append((high, high_error, high_rake))
        
        print(f"Initial bounds: LOW={low:.3f} (Rake={low_rake:.2f}, Error={low_error:.3f})")
        print(f"Initial bounds: HIGH={high:.3f} (Rake={high_rake:.2f}, Error={high_error:.3f})")
        
        # Update best results
        if abs(low_error) < abs(best_error):
            best_error = low_error
            best_lean = low
            best_rake = low_rake
            
        if abs(high_error) < abs(best_error):
            best_error = high_error
            best_lean = high
            best_rake = high_rake
        
        # Early return if we already have a good solution
        if abs(best_error) <= tolerance:
            return best_lean, best_rake
        
        # Track brackets we've already explored to avoid getting stuck
        explored_brackets = set()
        
        # Main search loop
        iteration = 0
        while iteration < max_iterations - 2:
            # If bounds are too close, we might be stuck in a region with a steep sign change
            if abs(current_high - current_low) < 1e-6:
                print(f"Search bounds have converged to the same value: {current_low:.6f}")
                print("Restarting search in a different region...")
                
                # Expand search range dramatically if we're stuck
                if full_range_searches < 2:  # Limit full range searches
                    # Try a full range search
                    current_low = low - 30 * (1 + full_range_searches)  # Expand more each time
                    current_high = high + 30 * (1 + full_range_searches)
                    print(f"Performing full range search: [{current_low:.3f}, {current_high:.3f}]")
                    full_range_searches += 1
                else:
                    # Sample several points across the original range to find new promising regions
                    print("Sampling across original range to find new promising regions...")
                    sample_points = np.linspace(low - 20, high + 20, 7)
                    
                    # Evaluate sample points we haven't tried yet
                    sample_results = []
                    for sample in sample_points:
                        # Skip points we've already evaluated
                        if any(abs(sample - x) < 1e-6 for x, _, _ in all_points):
                            continue
                            
                        sample_error, sample_rake = objective_function(sample)
                        all_points.append((sample, sample_error, sample_rake))
                        sample_results.append((sample, sample_error, sample_rake))
                        
                        # Update best if better
                        if abs(sample_error) < abs(best_error):
                            best_error = sample_error
                            best_lean = sample
                            best_rake = sample_rake
                    
                    # Find the best sample point
                    if sample_results:
                        best_sample = min(sample_results, key=lambda x: abs(x[1]))
                        
                        # Set new search bounds around this point
                        current_low = best_sample[0] - 10
                        current_high = best_sample[0] + 10
                        print(f"New search region: [{current_low:.3f}, {current_high:.3f}]")
                    else:
                        # If all sample points were already evaluated, we've covered a lot of ground
                        # Try with the best point we've found so far
                        current_low = best_lean - 5
                        current_high = best_lean + 5
                        print(f"Focusing search around best point found: {best_lean:.3f}")
                
                # Reset stalled counter
                stalled_iterations = 0
                continue
            
            # Choose next point to evaluate
            mid = (current_low + current_high) / 2
            
            # If we've already evaluated this point (or very close to it), perturb slightly
            while any(abs(mid - x) < 1e-6 for x, _, _ in all_points):
                # Try a slight perturbation
                delta = (current_high - current_low) * 0.1
                mid += delta
                # If still too close, try another approach
                if mid >= current_high:
                    mid = current_low + (current_high - current_low) / 3
            
            error, actual_rake = objective_function(mid)
            all_points.append((mid, error, actual_rake))
            
            print(f"Iteration {iteration}: Lean={mid:.3f}, Rake={actual_rake:.2f}, Error={error:.3f}")
            print(f"Current bounds: LOW={current_low:.3f}, HIGH={current_high:.3f}")
            iteration += 1
            
            # Check if this is a better solution
            if abs(error) < abs(best_error):
                # Track improvement
                improvement = abs(best_error) - abs(error)
                print(f"Improved solution: Error reduced by {improvement:.3f}")
                
                # Update best solution
                best_error = error
                best_lean = mid
                best_rake = actual_rake
                
                # Reset stalled counter if significant improvement
                if improvement > tolerance / 10:
                    stalled_iterations = 0
                else:
                    stalled_iterations += 1
            else:
                stalled_iterations += 1
                
            # Check if we found a solution within tolerance
            if abs(error) <= tolerance:
                print(f"Found solution within tolerance: Lean={mid:.3f}, Error={error:.3f}")
                return mid, actual_rake
            
            # Get current bound errors
            current_low_error = next(e for x, e, _ in all_points if abs(x - current_low) < 1e-6)
            current_high_error = next(e for x, e, _ in all_points if abs(x - current_high) < 1e-6)
            
            # Check if we have a sign change (solution bracketing)
            bracket_key = None
            
            if error * current_low_error <= 0:
                # Solution is between current_low and mid
                bracket_key = (round(current_low, 3), round(mid, 3))
                
                # Check if we've already thoroughly explored this bracket
                if bracket_key not in explored_brackets:
                    print(f"Sign change detected: solution between {current_low:.3f} and {mid:.3f}")
                    current_high = mid
                    stalled_iterations = 0  # Reset stalled counter when we find a bracket
                    
                    # Mark this bracket as explored if it's small enough
                    if abs(current_high - current_low) < 1.0:
                        explored_brackets.add(bracket_key)
                    
                    continue
                
            if error * current_high_error <= 0:
                # Solution is between mid and current_high
                bracket_key = (round(mid, 3), round(current_high, 3))
                
                # Check if we've already thoroughly explored this bracket
                if bracket_key not in explored_brackets:
                    print(f"Sign change detected: solution between {mid:.3f} and {current_high:.3f}")
                    current_low = mid
                    stalled_iterations = 0  # Reset stalled counter when we find a bracket
                    
                    # Mark this bracket as explored if it's small enough
                    if abs(current_high - current_low) < 1.0:
                        explored_brackets.add(bracket_key)
                    
                    continue
            
            # If we found a bracket but it's already been explored, treat it as if no bracket was found
            
            # Check if we're stalled (not making significant progress)
            if stalled_iterations >= 3:
                print(f"Search stalled for {stalled_iterations} iterations - expanding search bounds")
                
                # Analyze all points to determine trend direction
                sorted_points = sorted(all_points, key=lambda x: x[0])
                
                # Calculate if errors are generally decreasing toward low or high end
                trend_to_low = False
                trend_to_high = False
                
                if len(sorted_points) >= 6:
                    # Check the trend in errors
                    low_errors = [abs(e) for _, e, _ in sorted_points[:3]]
                    high_errors = [abs(e) for _, e, _ in sorted_points[-3:]]
                    
                    trend_to_low = low_errors[0] < np.mean(low_errors[1:])
                    trend_to_high = high_errors[-1] < np.mean(high_errors[:-1])
                
                # Check magnitude of current errors and which is smaller
                at_low_end = abs(current_low_error) < abs(current_high_error)
                
                # Decide which bound to expand
                if trend_to_low or (not trend_to_high and at_low_end):
                    # Expand lower bound
                    bound_range = current_high - current_low
                    # Ensure a minimum expansion of 5 units
                    new_low = current_low - max(5, bound_range)
                    print(f"Expanding lower bound from {current_low:.3f} to {new_low:.3f}")
                    
                    # Skip if we've already evaluated this point
                    if any(abs(new_low - x) < 1e-6 for x, _, _ in all_points):
                        new_low = current_low - max(7, bound_range * 1.5)
                        print(f"Adjusting to avoid previously evaluated point: new_low = {new_low:.3f}")
                    
                    # Evaluate new point
                    new_low_error, new_low_rake = objective_function(new_low)
                    all_points.append((new_low, new_low_error, new_low_rake))
                    iteration += 1
                    
                    # Update best if better
                    if abs(new_low_error) < abs(best_error):
                        best_error = new_low_error
                        best_lean = new_low
                        best_rake = new_low_rake
                    
                    # Check for sign change with current bounds
                    if new_low_error * error <= 0:
                        # Found bracket between new_low and mid
                        current_low = new_low
                        current_high = mid
                    elif new_low_error * current_low_error <= 0:
                        # Found bracket between new_low and current_low
                        current_high = current_low
                        current_low = new_low
                    else:
                        # No bracket, but update bound
                        current_low = new_low
                else:
                    # Expand upper bound
                    bound_range = current_high - current_low
                    # Ensure a minimum expansion of 5 units
                    new_high = current_high + max(5, bound_range)
                    print(f"Expanding upper bound from {current_high:.3f} to {new_high:.3f}")
                    
                    # Skip if we've already evaluated this point
                    if any(abs(new_high - x) < 1e-6 for x, _, _ in all_points):
                        new_high = current_high + max(7, bound_range * 1.5)
                        print(f"Adjusting to avoid previously evaluated point: new_high = {new_high:.3f}")
                    
                    # Evaluate new point
                    new_high_error, new_high_rake = objective_function(new_high)
                    all_points.append((new_high, new_high_error, new_high_rake))
                    iteration += 1
                    
                    # Update best if better
                    if abs(new_high_error) < abs(best_error):
                        best_error = new_high_error
                        best_lean = new_high
                        best_rake = new_high_rake
                    
                    # Check for sign change with current bounds
                    if new_high_error * error <= 0:
                        # Found bracket between mid and new_high
                        current_low = mid
                        current_high = new_high
                    elif new_high_error * current_high_error <= 0:
                        # Found bracket between current_high and new_high
                        current_low = current_high
                        current_high = new_high
                    else:
                        # No bracket, but update bound
                        current_high = new_high
                
                # Reset stalled counter after bound expansion
                stalled_iterations = 0
                continue
            
            # If not stalled, continue with normal binary search logic
            
            # Analyze local trend around midpoint
            sorted_points = sorted(all_points, key=lambda x: x[0])
            mid_indices = [i for i, (x, _, _) in enumerate(sorted_points) if abs(x - mid) < 1e-6]
            if mid_indices:
                mid_idx = mid_indices[0]
                
                # Check if we have neighboring points to determine trend
                if mid_idx > 0 and mid_idx < len(sorted_points) - 1:
                    left_lean, left_error, _ = sorted_points[mid_idx - 1]
                    right_lean, right_error, _ = sorted_points[mid_idx + 1]
                    
                    # See if error improves in either direction
                    left_better = abs(left_error) < abs(error)
                    right_better = abs(right_error) < abs(error)
                    
                    if left_better and not right_better:
                        print(f"Error improves toward LEFT (from {abs(error):.3f} to {abs(left_error):.3f})")
                        current_high = mid
                        continue
                        
                    if right_better and not left_better:
                        print(f"Error improves toward RIGHT (from {abs(error):.3f} to {abs(right_error):.3f})")
                        current_low = mid
                        continue
            
            # If we get here, we couldn't determine from local trend
            # Use error magnitude to guide search    
            # Find the best point in each half of the current range
            left_points = [(x, e) for x, e, _ in all_points if current_low <= x < mid]
            right_points = [(x, e) for x, e, _ in all_points if mid < x <= current_high]
            
            best_left = min(left_points, key=lambda x: abs(x[1])) if left_points else (current_low, current_low_error)
            best_right = min(right_points, key=lambda x: abs(x[1])) if right_points else (current_high, current_high_error)
            
            # Compare best errors
            if abs(best_left[1]) < abs(best_right[1]):
                print(f"Best results toward LEFT side (error={abs(best_left[1]):.3f})")
                current_high = mid
            else:
                print(f"Best results toward RIGHT side (error={abs(best_right[1]):.3f})")
                current_low = mid
        
        # Return best result if we reach max iterations
        print(f"\nReached maximum iterations. Best solution found:")
        print(f"Lean angle: {best_lean:.3f}, Rake angle: {best_rake:.3f}, Error: {best_error:.3f}")
        return best_lean, best_rake

    # Start the search with the initial range
    optimal_lean_angle, final_rake_angle = adaptive_search(0, 10)
    return optimal_lean_angle, final_rake_angle



def main_and_splitter_optimization(geometry_path, exe_name, R_hub_1, R_2, b_2, splitter_existence, hub_percentage_splitter, target_rake_angle, curve_directory ,convert_to_csv_object):
   
   main_blade_folder = os.path.join(geometry_path, 'Main_Blades')
   splitter_blade_folder = os.path.join(geometry_path, 'Splitter_Blades')
   
   print("\nOptimizing main blade rake angle...")
   final_lean_angle_main, final_rake_angle_main = optimize_lean_angle(main_blade_folder, 0, target_rake_angle, R_hub_1, R_2, b_2, 0, exe_name, curve_directory ,convert_to_csv_object ,tolerance=90)
   print(f"Main blade: Lean={final_lean_angle_main:.2f}°, Rake={final_rake_angle_main:.2f}°")
    
   if splitter_existence:
       print("\nOptimizing splitter blade rake angle...")
       # First pass with main blade rake as target
       final_lean_angle_splitter, final_rake_angle_splitter = optimize_lean_angle(splitter_blade_folder, 1, final_rake_angle_main, R_hub_1, R_2, b_2, 0, exe_name, curve_directory ,convert_to_csv_object , tolerance=90)
       
       # Check rake angle difference
       rake_diff = abs(final_rake_angle_main - final_rake_angle_splitter)
       if rake_diff > 0.2:
           print(f"\nRefinining splitter rake angle (current diff: {rake_diff:.2f}°)...")
           final_lean_angle_splitter, final_rake_angle_splitter = optimize_lean_angle(splitter_blade_folder, 1, final_rake_angle_main, R_hub_1, R_2, b_2, final_lean_angle_splitter, exe_name, curve_directory ,convert_to_csv_object , tolerance=0.05)
       
       print(f"Splitter blade: Lean={final_lean_angle_splitter:.2f}°, Rake={final_rake_angle_splitter:.2f}°")
       print(f"Final rake angle difference: {abs(final_rake_angle_main - final_rake_angle_splitter):.2f}°")
       
   return abs(final_rake_angle_main - final_rake_angle_splitter) <= 0.2 if splitter_existence else True



def append_lean_angle(blade_folder,lean_angle):
    """
    Append a specified lean angle to a blade
    """

    lean_theta_file = os.path.join(blade_folder, 'lean_theta.dat')

    # Update the lean_theta.dat file
    with open(lean_theta_file, 'w') as f:
        f.write(f"{lean_angle:.6f}")




class CentrifugalCompressor:
    def __init__(self, R_1_hub, R_1_tip, R_2, L_z, b_2, beta_b1_hub, beta_b1_tip, beta_b2, t,
                 method_meridional, method_angles, method_thickness, hub_percentage_splitter,
                 splitter_existence, num_blades):
        """
        Initialize the CentrifugalCompressor class with geometry and method parameters.
        """
        self.R_1_hub = R_1_hub
        self.R_1_tip = R_1_tip
        self.R_2 = R_2
        self.L_z = L_z
        self.b_2 = b_2
        self.beta_b1_hub = beta_b1_hub
        self.beta_b1_tip = beta_b1_tip
        self.beta_b2 = beta_b2
        self.t = t
        self.method_meridional = method_meridional
        self.method_angles = method_angles
        self.method_thickness = method_thickness
        self.hub_percentage_splitter = hub_percentage_splitter
        self.splitter_existence = splitter_existence
        self.num_blades = num_blades
        
        # Separate thickness factors for main and splitter blades

        # main blades
        self.main_t_tip_reduction_factor = 2.4
        self.main_t_hub_increase_factor = 1.6
        self.main_t_edge_decrease_factor = 3

        # splitter blades
        self.splitter_t_tip_reduction_factor = 2.4
        self.splitter_t_hub_increase_factor = 1.6
        self.splitter_t_edge_decrease_factor = 3
        self.splitter_t_reduction_factor = 1


        # Set initial thickness values
        self.update_thickness_values()

        self.iteration = 0
        self.n_control_points = 4

        

    def execute(self):

        self.optimal_params = self.optimize()
        self.generate_meridional_contour_optimisation(self.optimal_params)        
        self.generate_blade_angle_distribution_bezier()
        self.generate_main_blade_thickness()

        if self.splitter_existence:
            self.identify_splitter_start_percentage()
            self.generate_splitter_meridional_view()
            self.generate_splitter_angles()
            self.generate_splitter_blade_thickness()


    def execute_only_meridional(self):
        
        self.optimal_params = self.optimize()
        self.generate_meridional_contour_optimisation(self.optimal_params)

        if self.splitter_existence:
            self.identify_splitter_start_percentage()
            self.generate_splitter_meridional_view()


    def execute_only_angles(self):
        
        self.generate_blade_angle_distribution_bezier()

        if self.splitter_existence:
            self.identify_splitter_start_percentage()
            self.generate_splitter_angles()


    def execute_only_thickness(self):
    
        self.generate_main_blade_thickness()

        if self.splitter_existence:
            self.identify_splitter_start_percentage()
            self.generate_splitter_blade_thickness()



    def update_thickness_values(self):
        """
        Update thickness ratio values
        """
        
        # Main blade thickness values
        self.t_tip = self.t / self.main_t_tip_reduction_factor
        self.t_hub = self.t * self.main_t_hub_increase_factor
        self.t_edge = self.t / self.main_t_edge_decrease_factor

        # Splitter blade thickness values
        self.t_splitter = self.t / self.splitter_t_reduction_factor
        self.t_edge_splitter = self.t_splitter / self.splitter_t_edge_decrease_factor
        self.t_tip_splitter = self.t_splitter / self.splitter_t_tip_reduction_factor
        self.t_hub_splitter = self.t_splitter * self.splitter_t_hub_increase_factor

        print(f"Updated thickness values: t_tip={self.t_tip:.4f}, t_hub={self.t_hub:.4f}, "
              f"t_edge={self.t_edge:.4f}, t_splitter={self.t_splitter:.4f}")




    def smart_adjust_thickness_factors(self, attempt, blade_type):
        """
        Adjust the thickness factors until a valid blade geometry is obtained
        """

        base_step = 0.1
        max_step = 0.5
        
        if blade_type == 'main':
            factors = ['main_t_tip_reduction_factor', 'main_t_hub_increase_factor']
        else:  # splitter
            factors = ['splitter_t_tip_reduction_factor', 'splitter_t_hub_increase_factor']
        
        steps = [i * base_step for i in range(-int(max_step/base_step), int(max_step/base_step) + 1)]
        combinations = list(itertools.product(steps, repeat=len(factors)))
        combinations.sort(key=lambda x: sum(abs(i) for i in x))
        
        adjustment = combinations[attempt % len(combinations)]
        
        for factor, step in zip(factors, adjustment):
            current_value = getattr(self, factor)
            new_value = max(0.1, current_value + step)
            setattr(self, factor, new_value)
        
        self.update_thickness_values()

        if blade_type == 'main':
            self.generate_main_blade_thickness()
        else:
            self.generate_splitter_blade_thickness()

        return getattr(self, factors[0]), getattr(self, factors[1])




    def reset_thickness_factors(self):
        """
        Reset thickness factors to their default values
        """

        self.t_tip_reduction_factor = 2
        self.t_hub_increase_factor = 1.2
        self.t_edge_decrease_factor = 3
        self.t_splitter_reduction_factor = 1.4
        self.update_thickness_values()



    def return_thickness_factors(self):
        """
        Return current thickness factors as a dictionary
        """

        return {'t_tip_reduction_factor': self.main_t_tip_reduction_factor, 't_hub_increase_factor': self.main_t_hub_increase_factor, 't_edge_decrease_factor': self.main_t_edge_decrease_factor, 't_splitter_reduction_factor': self.splitter_t_reduction_factor}





    def bezier_curve(self, points, num=5000):
        """
        Generate a Bezier curve based on some control points - Best practice in Turbomachinery applications
        """

        n = len(points) - 1
        t = np.linspace(0, 1, num)
        curve = np.zeros((num, 2))
        for i, point in enumerate(points):
            curve += np.outer(comb(n, i) * (1-t)**(n-i) * t**i, point)
        return curve[:, 0], curve[:, 1]



    def generate_meridional_contours_bezier(self,num_points = 1000):
        """
        Generate the meridional curves of the main blades
        """
        
        # Hub contour control points
        self.hub_points = np.array([
            [0, self.R_1_hub],
            [0.1 * self.L_z, self.R_1_hub * 1.02],  # Slight initial curve
            [0.85 * self.L_z, self.R_1_hub + 0.3 * (self.R_2  - self.R_1_hub)],  # Start of main curvature
            [0.98 * self.L_z, self.R_2  - 0.1 * (self.R_2 - self.R_1_hub)],  # Approaching outlet
            [self.L_z, self.R_2]  # Outlet
        ])

        self.x_hub_curve_main, self.y_hub_curve_main = self.bezier_curve(self.hub_points)
        self.hub_curve_main = np.array([self.x_hub_curve_main, self.y_hub_curve_main])

        # Shroud contour control points
        self.shroud_points = np.array([
            [0, self.R_1_tip],
            [0.15 * (self.L_z-self.b_2), self.R_1_tip + 0.02 * (self.R_2 - self.R_1_tip)],  # Slight initial expansion
            [0.5 * (self.L_z-self.b_2), self.R_1_tip + 0.3 * (self.R_2 - self.R_1_tip)],  # Main expansion
            [0.9 * (self.L_z-self.b_2), self.R_1_tip + 0.75 * (self.R_2 - self.R_1_tip)],  # Approaching outlet
            [self.L_z-self.b_2, self.R_2]  # Outlet
        ])

        self.x_casing_curve_main, self.y_casing_curve_main = self.bezier_curve(self.shroud_points)
        self.casing_curve_main = np.array([self.x_casing_curve_main, self.y_casing_curve_main])

        # Generate leading edge coordinates
        self.x_leading_main = np.linspace(0, 0, num_points)
        self.y_leading_main = np.linspace(self.R_1_hub, self.R_1_tip, num_points)

        # Generate trailing edge coordinates
        self.x_trailing_main = np.linspace(self.L_z - self.b_2, self.L_z, num_points)
        self.y_trailing_main = np.linspace(self.R_2, self.R_2, num_points)




    def generate_meridional_contour_optimisation(self, params, num_points=10000):
        """
        Function to optimise the control points of a bezier curve for the meridional view of the impeller
        """

        # Infer the number of control points from the length of params
        use_four_points = len(params) == 8  # 4 for hub + 4 for shroud

        if use_four_points:
            hub_params = params[:4]
            shroud_params = params[4:]
            hub_control_points = np.array([
                [0, self.R_1_hub],
                [self.L_z * 0.15, self.R_1_hub + hub_params[0]],
                [self.L_z * 0.7, self.R_1_hub + hub_params[1]],
                [self.L_z * 0.8, self.R_1_hub + hub_params[2]],
                [self.L_z * 0.9, self.R_1_hub + hub_params[3]],
                [self.L_z, self.R_2]
            ])


            
            
            shroud_control_points = np.array([
                [0, self.R_1_tip],
                [(self.L_z-self.b_2)* 0.2, self.R_1_tip + shroud_params[0]],
                [(self.L_z-self.b_2) * 0.4, self.R_1_tip + shroud_params[1]],
                [(self.L_z-self.b_2) * 0.5, self.R_1_tip + shroud_params[2]],
                [(self.L_z-self.b_2) * 0.9, self.R_1_tip + shroud_params[3]],
                [self.L_z-self.b_2, self.R_2]
            ])

        else:

            hub_params = params[:3]
            shroud_params = params[3:]
            hub_control_points = np.array([
                [0, self.R_1_hub],
                [self.L_z * 0.2, self.R_1_hub + hub_params[0]],
                [self.L_z * 0.6, self.R_1_hub + hub_params[1]],
                [self.L_z * 0.98, self.R_1_hub + hub_params[2]],
                [self.L_z, self.R_2]
            ])


            

            shroud_control_points = np.array([
                [0, self.R_1_tip],
                [(self.L_z-self.b_2)* 0.2, self.R_1_tip + shroud_params[0]],
                [(self.L_z-self.b_2) * 0.5, self.R_1_tip + shroud_params[1]],
                [(self.L_z-self.b_2) * 0.9, self.R_1_tip + shroud_params[2]],
                [self.L_z-self.b_2, self.R_2]
            ])
        
        self.hub_points = hub_control_points
        self.shroud_points = shroud_control_points

        self.hub_curve_main = np.array(self.bezier_curve(hub_control_points, num_points))
        self.casing_curve_main = np.array(self.bezier_curve(shroud_control_points, num_points))
        
        self.x_hub_curve_main, self.y_hub_curve_main = self.hub_curve_main
        self.x_casing_curve_main, self.y_casing_curve_main = self.casing_curve_main

        # Generate leading edge coordinates
        self.x_leading_main = np.linspace(0, 0, num_points)
        self.y_leading_main = np.linspace(self.R_1_hub, self.R_1_tip, num_points)

        # Generate trailing edge coordinates
        self.x_trailing_main = np.linspace(self.L_z - self.b_2, self.L_z, num_points)
        self.y_trailing_main = np.linspace(self.R_2, self.R_2, num_points)


    


    def objective_function(self, params):
        """
        Objective function for bezier curve control points optimisation
        """

        self.generate_meridional_contour_optimisation(params)
        
        # Area ratio
        area_ratio = (self.casing_curve_main[1, -1]**2 - self.hub_curve_main[1, -1]**2) / \
                    (self.casing_curve_main[1, 0]**2 - self.hub_curve_main[1, 0]**2)
        target_area_ratio = 1.6
        area_ratio_penalty = ((area_ratio - target_area_ratio) / target_area_ratio)**2

        # Diffusion factor
        diffusion_factor = 1 - (self.casing_curve_main[1, -1] - self.hub_curve_main[1, -1]) / \
                        (self.casing_curve_main[1, 0] - self.hub_curve_main[1, 0])
        
        max_diffusion_factor = 0.6
        diffusion_penalty = max(0, (diffusion_factor - max_diffusion_factor) / max_diffusion_factor)**2

        # Curvature
        hub_curvature = np.gradient(np.gradient(self.hub_curve_main[1], self.hub_curve_main[0]), self.hub_curve_main[0])
        shroud_curvature = np.gradient(np.gradient(self.casing_curve_main[1], self.casing_curve_main[0]), self.casing_curve_main[0])
        max_curvature = max(np.max(np.abs(hub_curvature)), np.max(np.abs(shroud_curvature)))
        curvature_penalty = (max_curvature / (self.R_2 - self.R_1_hub))**2

        # Monotonicity constraint
        hub_monotonicity = np.sum(np.maximum(0, -np.diff(self.hub_curve_main[1])))
        shroud_monotonicity = np.sum(np.maximum(0, -np.diff(self.casing_curve_main[1])))
        monotonicity_penalty = (hub_monotonicity + shroud_monotonicity) / self.L_z

        # Passage area variation
        passage_areas = np.pi * (self.casing_curve_main[1]**2 - self.hub_curve_main[1]**2)
        area_variation = np.std(np.diff(passage_areas)) / np.mean(passage_areas)
        area_variation_penalty = area_variation**2

        # Inlet horizontality penalty - shroud
        hub_inlet_slope = (self.hub_curve_main[1][100] - self.hub_curve_main[1][0]) / (self.hub_curve_main[0][10] - self.hub_curve_main[0][0])
        shroud_inlet_slope = (self.casing_curve_main[1][10] - self.casing_curve_main[1][0]) / (self.casing_curve_main[0][10] - self.casing_curve_main[0][0])
        inlet_horizontality_penalty_shroud = (shroud_inlet_slope**2) / 2
        
        # Outlet verticality penalty
        hub_outlet_slope = (self.hub_curve_main[1][-1] - self.hub_curve_main[1][-11]) / (self.hub_curve_main[0][-1] - self.hub_curve_main[0][-11])
        shroud_outlet_slope = (self.casing_curve_main[1][-1] - self.casing_curve_main[1][-11]) / (self.casing_curve_main[0][-1] - self.casing_curve_main[0][-11])
        outlet_verticality_penalty = ((1/hub_outlet_slope)**2 + (1/shroud_outlet_slope)**2) / 2

        # Smoothness penalty
        hub_smoothness = np.sum(np.diff(np.gradient(self.hub_curve_main[1], self.hub_curve_main[0]))**2)
        shroud_smoothness = np.sum(np.diff(np.gradient(self.casing_curve_main[1], self.casing_curve_main[0]))**2)
        smoothness_penalty = (hub_smoothness + shroud_smoothness) / self.L_z

        # Curvature continuity penalty
        hub_curvature_continuity = np.sum(np.diff(hub_curvature)**2)
        shroud_curvature_continuity = np.sum(np.diff(shroud_curvature)**2)
        curvature_continuity_penalty = (hub_curvature_continuity + shroud_curvature_continuity) / self.L_z

        # Hub inlet gradient penalty (allow small positive gradient)
        desired_hub_inlet_slope = 1  # Adjust this value as needed
        hub_inlet_gradient_penalty = (hub_inlet_slope - desired_hub_inlet_slope)**2

        # Composite objective function with adjusted weights
        objective = (
            area_ratio_penalty +
            1 * diffusion_penalty +  # Slightly reduced
            3 * curvature_penalty +  # Slightly reduced
            5 * monotonicity_penalty +  # Slightly reduced
            area_variation_penalty +
            10 * inlet_horizontality_penalty_shroud +  # Slightly reduced
            10 * smoothness_penalty +  # Slightly reduced
            10 * curvature_continuity_penalty +  # Slightly reduced
            5 * hub_inlet_gradient_penalty +  # Slightly reduced
            30 * outlet_verticality_penalty  # Significantly increased
        )

        return objective



    def optimize(self, use_four_points=False):
        """
        Optimisation algorithm for bezier curve control points
        """

        self.use_four_points = use_four_points
        
        if use_four_points:
            
            
            bounds = [
                (0, 0.2 * (self.R_2 - self.R_1_hub)),
                (0, 0.3 * (self.R_2 - self.R_1_hub)),
                (0, 0.3 * (self.R_2 - self.R_1_hub)),
                (0, 0.2 * (self.R_2 - self.R_1_hub)),
                (0, 0.2 * (self.R_2 - self.R_1_tip)),
                (0, 0.3 * (self.R_2 - self.R_1_tip)),
                (0, 0.3 * self.b_2),
                (0, 0.2 * self.b_2)
            ]
        
        else:
           
           
            bounds = [
                (0, 0.2 * (self.R_2 - self.R_1_hub)),
                (0, 0.3 * (self.R_2 - self.R_1_hub)),
                (0, 0.2 * (self.R_2 - self.R_1_hub)),
                (0, 0.2 * (self.R_2 - self.R_1_tip)),
                (0, 0.3 * (self.R_2 - self.R_1_tip)),
                (0, 0.2 * self.b_2)
            ]

        initial_guess = np.array([b[1]/2 for b in bounds])

        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        cma_bounds = [lower_bounds, upper_bounds]

        # Try different optimization algorithms
        algorithms = [
            ("SLSQP", lambda: minimize(self.objective_function, initial_guess, method='SLSQP', bounds=bounds, options={'ftol': 1e-50, 'disp': True, 'maxiter': 3000})),
            ("L-BFGS-B", lambda: minimize(self.objective_function, initial_guess, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-10, 'disp': True, 'maxiter': 300})),
            # ("Differential Evolution", lambda: differential_evolution(self.objective_function, bounds, disp=True, popsize=100, mutation=(0.5, 1), recombination=0.7, maxiter=1000)),
            # ("Basin-Hopping", lambda: basinhopping(self.objective_function, initial_guess, minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds}, niter=10, T=1.0, stepsize=0.1, disp=True)),
            ("CMA-ES", lambda: cma.fmin(self.objective_function, initial_guess, 0.5, options={'bounds': cma_bounds, 'tolfun': 1e-10, 'maxiter': 500}))  # Corrected CMA-ES
        ]

        best_result = None
        best_value = float('inf')

        for name, algorithm in algorithms:
            print(f"\nRunning optimization with {name}")
            result = algorithm()
            if hasattr(result, 'fun'):
                objective_value = result.fun
                params = result.x
            else:  # If the optimizer doesn't return a result object with a `fun` attribute, e.g., CMA-ES
                objective_value = result[1]
                params = result[0]
            if objective_value < best_value:
                best_value = objective_value
                best_result = params
                print(f"New best result found with {name}: {best_value}")

        print(f"\nBest optimization result: {best_value}")
        return best_result







    def generate_blade_angle_distribution_bezier(self):
        """
        Generate the blade angle distribution for the main blade based on Bezier curves
        """
        
        # Hub blade angle distribution
        # self.hub_angle_points = np.array([
        #     [0, self.beta_b1_hub],
        #     [0.2, self.beta_b1_hub * 0.45],  # Slight decrease at inducer
        #     [0.5, self.beta_b1_hub * 0.45],   # Start of main angle change
        #     # [0.8, 0.75*(0.15*self.beta_b1_hub + 0.85*self.beta_b2) ],  # Middle of impeller
        #     [0.8, self.beta_b2 * 0.85],      # Approaching exit angle
        #     [1, self.beta_b2]                # Exit angle
        # ])


        ############ NEW ##############################
        self.hub_angle_points = np.array([
            [0, self.beta_b1_hub],
            [0.25, self.beta_b1_hub * 0.5],                        # Less sharp decrease at inducer
            [0.5, self.beta_b1_hub * 0.45],                         # Less extreme minimum point
            [0.75, self.beta_b2 * 0.75],                             # Smoother rise toward exit
            [1, self.beta_b2]                                       # Exit angle
        ])


        ##### MODIFIED FOR LOWER HUB ANGES #####
        # self.hub_angle_points = np.array([
        #     [0, self.beta_b1_hub],
        #     [0.25, self.beta_b1_hub * 0.50],                        # Less sharp decrease at inducer
        #     [0.5, self.beta_b1_hub * 0.45],                         # Less extreme minimum point
        #     [0.75, self.beta_b2 * 0.80],                             # Smoother rise toward exit
        #     [1, self.beta_b2]                                       # Exit angle
        # ])




        ########## FOR NASA CC3 GENERATED ONE ########
        # self.hub_angle_points = np.array([
        #     [0, self.beta_b1_hub],
        #     [0.15, self.beta_b1_hub * 0.22],  # Steeper initial drop
        #     [0.4, self.beta_b1_hub * 0.25],  # Lower minimum, shifted right
        #     [0.75, self.beta_b2 * 0.36],      # More gradual rise initially
        #     [1, self.beta_b2]                # Exit angle
        # ])

        self.x_hub_curve_main_angle, self.theta_hub_curve_main = self.bezier_curve(self.hub_angle_points)

        # Shroud blade angle distribution
        # self.shroud_angle_points = np.array([
        #     [0, self.beta_b1_tip],
        #     # [0.1, self.beta_b1_tip * 0.99],  # Very slight decrease at inducer
        #     [0.3, self.beta_b1_tip * 0.95],  # Start of main angle change
        #     [0.6, 0.15*self.beta_b1_tip + 0.85*self.beta_b2],  # Middle of impeller
        #     [0.9, self.beta_b2 * 0.95],      # Approaching exit angle
        #     [1, self.beta_b2]                # Exit angle
        # ])


        ###################################################


        ######### NEW ####### CASEY #########
        self.shroud_angle_points = np.array([
            [0, self.beta_b1_tip],
            [0.25, self.beta_b1_tip - 0.13 * max(0.001, self.beta_b1_tip - self.beta_b2)],  # Slight decrease at inducer
            [0.5, self.beta_b1_tip - 1.0 * max(0.001, self.beta_b1_tip - self.beta_b2)],   # Middle point
            [0.75, self.beta_b2 - 0.4 * max(0.001, self.beta_b1_tip - self.beta_b2)],      # Approaching exit
            [1, self.beta_b2]                                                              # Exit angle
        ])


        ######### FOR LOWER INLET TIP BLADE ANGLES #########
        # self.shroud_angle_points = np.array([
        #     [0, self.beta_b1_tip],
        #     [0.25, self.beta_b1_tip - 0.2 * max(0.001, self.beta_b1_tip - self.beta_b2)],  # Slight decrease at inducer
        #     [0.5, self.beta_b1_tip - 1.8 * max(0.001, self.beta_b1_tip - self.beta_b2)],   # Middle point
        #     [0.75, self.beta_b2 - 1.35 * max(0.001, self.beta_b1_tip - self.beta_b2)],      # Approaching exit
        #     [1, self.beta_b2]                                                              # Exit angle
        # ])


        #######################################



        # ########## FOR NASA CC3 GENERATED ONE ########
        # self.shroud_angle_points = np.array([
        #     [0, self.beta_b1_tip],
        #     [0.2, self.beta_b1_tip * 0.58],   # Steeper initial drop
        #     [0.35, self.beta_b1_tip * 0.55],  # Lower minimum
        #     [0.6, self.beta_b1_tip * 0.4 + 0.25*self.beta_b2],  # Lower middle section
        #     [0.85, self.beta_b2 * 0.65],      # More gradual approach to exit
        #     [1, self.beta_b2]                 # Exit angle
        # ])

        self.x_casing_curve_main_angle, self.theta_casing_curve_main = self.bezier_curve(self.shroud_angle_points)
        
    


        
    def identify_splitter_start_percentage(self, tol=5e-2):
        """
        Find where the splitter leading edge begins, expressed as

            • stream-wise fraction on the hub curve   (≈ user-given 0.40)
            • *and* the corresponding fraction on the shroud curve
            that sits at the same x-coordinate.

        ‘Stream-wise fraction’ is now measured with TRUE arc-length, not just Δx.
        """

        # ---------- 1. cumulative arc-length fractions on hub & shroud ----------
        def frac_along_curve(x, y):
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dx**2 + dy**2)          # true element length
            s  = np.concatenate(([0.0], np.cumsum(ds)))
            return s / s[-1]                     # 0 … 1

        cum_hub = frac_along_curve(*self.hub_curve_main)
        cum_sh  = frac_along_curve(*self.casing_curve_main)

        # store for later diagnostics / plots
        self.cum_hub  = cum_hub
        self.cum_cas  = cum_sh

        # ---------- 2. hub index whose fraction best matches user’s request -----
        hub_idx = np.argmin(np.abs(cum_hub - self.hub_percentage_splitter))
        self.hub_x_splitter = self.hub_curve_main[0][hub_idx]
        self.hub_y_splitter = self.hub_curve_main[1][hub_idx]
        self.splitter_start_hub_frac = cum_hub[hub_idx]          # ~ 0.400

        # ---------- 3. find same x on the shroud & grab its stream-wise frac ----
        sh_idx = np.argmin(np.abs(self.casing_curve_main[0] - self.hub_x_splitter))
        self.casing_x_splitter  = self.casing_curve_main[0][sh_idx]
        self.casing_y_splitter  = self.casing_curve_main[1][sh_idx]
        self.splitter_start_shroud_frac = cum_sh[sh_idx]         # ~ 0.548

        # ---------- 4. slice downstream portions for the splitter curves -------
        self.hub_x_splitter_curve    = self.hub_curve_main[0][hub_idx:]
        self.hub_y_splitter_curve    = self.hub_curve_main[1][hub_idx:]
        self.casing_x_splitter_curve = self.casing_curve_main[0][sh_idx:]
        self.casing_y_splitter_curve = self.casing_curve_main[1][sh_idx:]

        # ---------- 5. report ---------------------------------------------------
        print(f"Meridional splitter start → "
            f"hub frac={self.splitter_start_hub_frac:.3f} @ x={self.hub_x_splitter:.3f}, "
            f"shroud frac={self.splitter_start_shroud_frac:.3f} @ x={self.casing_x_splitter:.3f}")
        
        




        
    def generate_splitter_meridional_view(self, num_points=1000):
        """
        Build the straight‐line leading edge and full meridional contours
        for the splitter, starting at the points set in identify_splitter_start_percentage.
        """
        # 1) straight‐line leading edge between the two start points
        self.leading_edge_x_splitter = np.linspace(self.hub_x_splitter,
                                                self.casing_x_splitter,
                                                num_points)
        self.leading_edge_y_splitter = np.linspace(self.hub_y_splitter,
                                                self.casing_y_splitter,
                                                num_points)

        # 2) down-stream hub & shroud from those same split points
        #    (we already sliced in identify_splitter_start_percentage)
        #    just prepend the exact start point to ensure continuity
        self.splitter_hub_curve = np.vstack([
            [self.hub_x_splitter,   self.hub_y_splitter],
            np.column_stack((self.hub_x_splitter_curve,    self.hub_y_splitter_curve))
        ]).T

        self.splitter_casing_curve = np.vstack([
            [self.casing_x_splitter, self.casing_y_splitter],
            np.column_stack((self.casing_x_splitter_curve, self.casing_y_splitter_curve))
        ]).T

        # 3) copy trailing‐edge same as main (you already had these)
        self.x_trailing_splitter = self.x_trailing_main
        self.y_trailing_splitter = self.y_trailing_main

        print("Generated splitter meridional view with "
            f"{self.splitter_hub_curve.shape[1]} hub pts & "
            f"{self.splitter_casing_curve.shape[1]} shroud pts.")



    
    
    def generate_splitter_angles(self):
        """
        Create hub- and shroud-side angle distributions for the splitter
        blade, starting at their respective main-blade streamwise fractions
        (hub ≈ splitter_start_hub_frac, shroud ≈ splitter_start_shroud_frac)
        and resampled so both sides contain the same number of points.
        """

        # ------------------------------------------------
        # 1.  cumulative (true) fractions on the main blade
        # ------------------------------------------------
        dx_hub   = np.diff(self.x_hub_curve_main_angle)
        frac_hub = np.concatenate(([0.0], np.cumsum(dx_hub)))
        frac_hub /= frac_hub[-1]                       # 0 … 1

        dx_sh    = np.diff(self.x_casing_curve_main_angle)
        frac_sh  = np.concatenate(([0.0], np.cumsum(dx_sh)))
        frac_sh  /= frac_sh[-1]

        # ------------------------------------------------
        # 2.  locate splitter-start indices (fractions are
        #     stored earlier in identify_splitter_start_percentage)
        # ------------------------------------------------
        hub_start_idx = np.argmin(
            np.abs(frac_hub - self.splitter_start_hub_frac))

        sh_start_idx  = np.argmin(
            np.abs(frac_sh  - self.splitter_start_shroud_frac))

        # ------------------------------------------------
        # 3.  slices of x- and θ-arrays from those indices
        # ------------------------------------------------
        hub_x_slice  = self.x_hub_curve_main_angle[hub_start_idx:]
        sh_x_slice   = self.x_casing_curve_main_angle[sh_start_idx:]

        th_hub_slice = self.theta_hub_curve_main[hub_start_idx:]
        th_sh_slice  = self.theta_casing_curve_main[sh_start_idx:]

        frac_hub_sl  = frac_hub[hub_start_idx:]      # 0.400 … 1
        frac_sh_sl   = frac_sh[sh_start_idx:]        # 0.548 … 1

        # ------------------------------------------------
        # 4.  resample both sides on a common parameter grid
        # ------------------------------------------------
        n_common  = max(len(th_hub_slice), len(th_sh_slice))
        s_common  = np.linspace(0.0, 1.0, n_common)   # parameter 0–1

        def resample(y_old):
            s_old = np.linspace(0.0, 1.0, len(y_old))
            return np.interp(s_common, s_old, y_old)

        # angles
        self.theta_splitter_hub    = resample(th_hub_slice)
        self.theta_splitter_casing = resample(th_sh_slice)

        # x-coordinates (needed for thickness, beta.dat, etc.)
        self.x_splitter_hub_angle    = resample(hub_x_slice)
        self.x_splitter_casing_angle = resample(sh_x_slice)

        # fractions for **plotting on the main-blade axis**
        self.split_hub_frac_main_axis = resample(frac_hub_sl)
        self.split_cas_frac_main_axis = resample(frac_sh_sl)

        # fractions for a local 0–1 axis (legacy / other uses)
        self.split_hub_frac_slice = s_common
        self.split_cas_frac_slice = s_common

        # ------------------------------------------------
        # 5.  diagnostics
        # ------------------------------------------------
        print(f"Splitter angles generated:"
            f" hub pts = {len(self.theta_splitter_hub)},"
            f" shroud pts = {len(self.theta_splitter_casing)}")
        print(f"   hub starts at main-axis fraction "
            f"{self.split_hub_frac_main_axis[0]:.3f}")
        print(f"   shroud starts at main-axis fraction "
            f"{self.split_cas_frac_main_axis[0]:.3f}")



    def create_thickness_distribution(self, x, t_edge, t_max, leading_edge_ratio, constant_ratio, trailing_edge_ratio):
        """
        Create blade thickness distribution
        """
        
        thickness = np.zeros_like(x)
        
        # Leading edge (elliptical distribution)  # CHANGE TO PARABOLIC FOR BOTH
        leading_edge_mask = x < leading_edge_ratio
        x_normalized = x[leading_edge_mask] / leading_edge_ratio
        thickness[leading_edge_mask] = t_edge + (t_max - t_edge) * np.sqrt(1 - (1 - x_normalized)**2)
        
        # Constant thickness section
        constant_mask = (x >= leading_edge_ratio) & (x < 1 - trailing_edge_ratio)
        thickness[constant_mask] = t_max
        
        # Trailing edge (parabolic distribution)
        trailing_edge_mask = x >= 1 - trailing_edge_ratio
        x_normalized = (x[trailing_edge_mask] - (1 - trailing_edge_ratio)) / trailing_edge_ratio
        thickness[trailing_edge_mask] = t_max - (t_max - t_edge) * x_normalized**2
        
        return thickness



    def generate_main_blade_thickness(self, num_points=5000):
        """
        Create blade thickness distribution for main blade
        """

        x = np.linspace(0, 1, num_points)
        
        # Tip thickness
        thickness_tip = self.create_thickness_distribution(x, self.t_edge, self.t_tip, 0.2, 0.7, 0.1)
        
        # Hub thickness
        thickness_hub = self.create_thickness_distribution(x, self.t_edge, self.t_hub, 0.2, 0.5, 0.3)
        
        self.x_shroud_thickness_main = x
        self.thickness_shroud_main = thickness_tip
        self.x_hub_thickness_main = x
        self.thickness_hub_main = thickness_hub





    def generate_splitter_blade_thickness(self, num_points=5000):
        """
        Create splitter blade thickness distribution
        """

        # Use the correct meridional positions for the splitter blade
        x_hub = self.x_splitter_hub_angle
        x_shroud = self.x_splitter_casing_angle

        # Normalize x values for thickness distribution calculation
        x_hub_norm = (x_hub - x_hub.min()) / (x_hub.max() - x_hub.min())
        x_shroud_norm = (x_shroud - x_shroud.min()) / (x_shroud.max() - x_shroud.min())
        
        # Generate thickness distributions
        thickness_hub = self.create_thickness_distribution(x_hub_norm, self.t_edge_splitter, self.t_hub_splitter, 0.2, 0.5, 0.3)
        thickness_shroud = self.create_thickness_distribution(x_shroud_norm, self.t_edge_splitter, self.t_tip_splitter, 0.2, 0.7, 0.1)
        
        # Store the results
        self.x_splitter_thickness_hub = x_hub
        self.thickness_splitter_hub = thickness_hub
        self.x_splitter_thickness_shroud = x_shroud
        self.thickness_splitter_shroud = thickness_shroud

        # Print some debug information
        print("Splitter Hub: x range =", x_hub.min(), "to", x_hub.max(), "thickness range =", thickness_hub.min(), "to", thickness_hub.max())
        print("Splitter Shroud: x range =", x_shroud.min(), "to", x_shroud.max(), "thickness range =", thickness_shroud.min(), "to", thickness_shroud.max())






    def return_geometry_lists(self):
        
        if self.splitter_existence== True:

            geometry = {
                'hub_x_meridional_main': self.x_hub_curve_main,
                'hub_y_meridional_main': self.y_hub_curve_main,
                'shroud_x_meridional_main': self.x_casing_curve_main,
                'shroud_y_meridional_main': self.y_casing_curve_main,
                'leading_x_meridional_main': self.x_leading_main,
                'leading_y_meridional_main': self.y_leading_main,
                'trailing_x_meridional_main': self.x_trailing_main,
                'trailing_y_meridional_main': self.y_trailing_main,
                'blade_x_hub_main': self.x_hub_curve_main_angle,
                'blade_theta_hub_main': self.theta_hub_curve_main,
                'blade_x_shroud_main': self.x_casing_curve_main_angle,
                'blade_theta_shroud_main': self.theta_casing_curve_main,
                'thickness_x_hub_main': self.x_hub_thickness_main,
                'thickness_hub_main': self.thickness_hub_main,
                'thickness_x_shroud_main': self.x_shroud_thickness_main,
                'thickness_shroud_main': self.thickness_shroud_main,
                'hub_x_meridional_splitter': self.hub_x_splitter_curve,
                'hub_y_meridional_splitter': self.hub_y_splitter_curve,
                'shroud_x_meridional_splitter': self.casing_x_splitter_curve,
                'shroud_y_meridional_splitter': self.casing_y_splitter_curve,
                'leading_x_meridional_splitter': self.leading_edge_x_splitter,
                'leading_y_meridional_splitter': self.leading_edge_y_splitter,
                'trailing_x_meridional_splitter': self.x_trailing_splitter,
                'trailing_y_meridional_splitter': self.y_trailing_splitter,
                'blade_x_hub_splitter': self.x_splitter_hub_angle,
                'blade_theta_hub_splitter': self.theta_splitter_hub,
                'blade_x_shroud_splitter': self.x_splitter_casing_angle,
                'blade_theta_shroud_splitter': self.theta_splitter_casing,
                'thickness_x_hub_splitter': self.x_splitter_thickness_hub,
                'thickness_hub_splitter': self.thickness_splitter_hub,
                'thickness_x_shroud_splitter': self.x_splitter_thickness_shroud,
                'thickness_shroud_splitter': self.thickness_splitter_shroud
            }

        else:

            geometry = {
                'hub_x_meridional_main': self.x_hub_curve_main,
                'hub_y_meridional_main': self.y_hub_curve_main,
                'shroud_x_meridional_main': self.x_casing_curve_main,
                'shroud_y_meridional_main': self.y_casing_curve_main,
                'leading_x_meridional_main': self.x_leading_main,
                'leading_y_meridional_main': self.y_leading_main,
                'trailing_x_meridional_main': self.x_trailing_main,
                'trailing_y_meridional_main': self.y_trailing_main,
                'blade_x_hub_main': self.x_hub_curve_main_angle,
                'blade_theta_hub_main': self.theta_hub_curve_main,
                'blade_x_shroud_main': self.x_casing_curve_main_angle,
                'blade_theta_shroud_main': self.theta_casing_curve_main,
                'thickness_x_hub_main': self.x_hub_thickness_main,
                'thickness_hub_main': self.thickness_hub_main,
                'thickness_x_shroud_main': self.x_shroud_thickness_main,
                'thickness_shroud_main': self.thickness_shroud_main
            }

        return geometry





    def plot_meridional(self, save_path):
        """
        Plot the meridional view of main and splitter,
        with explicit markers at splitter start.
        """
        # Optional: customize fonts & line widths
        rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'lines.linewidth': 1.5,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
        })

        plt.figure(figsize=(8, 6))
        # main contours
        plt.plot(self.hub_curve_main[0],    self.hub_curve_main[1],    '-', label='Hub Main')
        plt.plot(self.casing_curve_main[0], self.casing_curve_main[1], '-', label='Shroud Main')

        if self.splitter_existence:
            # splitter start markers
            plt.plot(self.hub_x_splitter,    self.hub_y_splitter,    'X', ms=8, label='Hub Split Start')
            plt.plot(self.casing_x_splitter, self.casing_y_splitter, 'D', ms=6, label='Shroud Split Start')
            # downstream splitter contours
            plt.plot(self.hub_x_splitter_curve,    self.hub_y_splitter_curve,    '--', label='Hub Splitter')
            plt.plot(self.casing_x_splitter_curve, self.casing_y_splitter_curve, ':',  label='Shroud Splitter')

        plt.xlabel('Axial Length (mm)')
        plt.ylabel('Radius (mm)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='best', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'meridional.png'), dpi=300)
        plt.show()


    def plot_blade_angle_distribution(self, save_path):
        """
        Plot blade‐angle vs true streamwise fraction for both main and splitter,
        using each curve’s own start fraction and marking the start points explicitly.
        """
        # set fonts, sizes, line widths, etc.
        rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'lines.linewidth': 1.5,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
        })

        fig, ax = plt.subplots(figsize=(10, 8))

        # --- main blade curves ---
        ax.plot(self.x_hub_curve_main_angle,   self.theta_hub_curve_main,   '-', label='Hub Main')
        ax.plot(self.x_casing_curve_main_angle,self.theta_casing_curve_main,'-', label='Shroud Main')

        
        # --- splitter curves ----------------------------------------------
        if self.splitter_existence:
            ax.plot(self.split_hub_frac_main_axis,  self.theta_splitter_hub,
                    '--', label='Hub Splitter')
            ax.plot(self.split_cas_frac_main_axis,  self.theta_splitter_casing,
                    ':',  label='Shroud Splitter')

            # mark real start points
            ax.scatter(self.split_hub_frac_main_axis[0], self.theta_splitter_hub[0],
                    marker='X', s=100, color='purple', label='Hub Split Start')
            ax.scatter(self.split_cas_frac_main_axis[0], self.theta_splitter_casing[0],
                    marker='D', s=100, color='brown',  label='Shroud Split Start')



        # control points for the main blade
        ax.scatter(self.hub_angle_points[:,0],   self.hub_angle_points[:,1],
                marker='o', edgecolor='black', label='Hub Ctrl Pt')
        ax.scatter(self.shroud_angle_points[:,0],self.shroud_angle_points[:,1],
                marker='s', edgecolor='black', label='Shroud Ctrl Pt')

        ax.set_xlabel('Streamwise Fraction')
        ax.set_ylabel('Blade Angle (°)')
        ax.set_ylim(0, 60)
        ax.grid(True)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        # save & show
        plt.savefig(os.path.join(save_path, 'blade_angle.png'),
                    dpi=1000, bbox_inches='tight')
        plt.show()

        # print out the actual start fractions so you can cross-check by eye
        # These two print lines below do not run
        # print(f"Hub splitter really starts at fraction {self.split_hub_frac_slice[0]:.3f}")
        # print(f"Shroud splitter really starts at fraction {self.split_cas_frac_slice[0]:.3f}")









    
    def plot_blade_thickness_distribution(self, save_path):
        """
        Plot the blade thickness distribution
        """
        # Explicit style settings (no ellipsis)
        rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'lines.linewidth': 1.5,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
        })

        plt.figure(figsize=(8, 6))
        # main blade
        plt.plot(self.x_hub_thickness_main,    self.thickness_hub_main,    '-', label='Hub Main')
        plt.plot(self.x_shroud_thickness_main, self.thickness_shroud_main, '-', label='Shroud Main')

        # splitter, if present
        if self.splitter_existence:
            plt.plot(self.x_splitter_thickness_hub,    self.thickness_splitter_hub,    '--', label='Hub Splitter')
            plt.plot(self.x_splitter_thickness_shroud, self.thickness_splitter_shroud, '--', label='Shroud Splitter')

        plt.xlabel('Normalized Meridional Distance')
        plt.ylabel('Blade Thickness (mm)')
        plt.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # save & show
        plt.savefig(os.path.join(save_path, 'blade_thickness.png'),
                    dpi=1000, bbox_inches='tight')
        plt.show()
