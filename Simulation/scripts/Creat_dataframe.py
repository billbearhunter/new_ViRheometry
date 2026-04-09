import numpy as np
import pandas as pd
import os

# Define the base directory containing all data set folders
base_dir = 'data'

# Define a function to extract parameters from folder name
def extract_parameters(folder_name):
    parts = folder_name.split('_')
    n = float(parts[1])
    eta = float(parts[3])
    sigma_y = float(parts[6])
    return n, eta, sigma_y

# Define a function to extract particle data from a .dat file
def extract_particle_data_x(dat_file_path):
    with open(dat_file_path, 'rb') as file:
        num_points = int.from_bytes(file.read(4), 'little')
        positions = np.frombuffer(file.read(3 * num_points * 4), dtype=np.float32).reshape((num_points, 3))
    return positions[:, 0].max()

# Define a function to extract setup from folder name
def extract_particle_setup(dat_file_path):
    parts = dat_file_path.split('_')
    height = float(parts[1])
    width = float(parts[2])
    return height, width
    
records = []
columns = ['n', 'eta', 'sigma_y']

# Iterate over each dataset folder
for parameter_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, parameter_folder)
    data = []
    print(folder_path)
    if os.path.isdir(folder_path):
        #extract parameter from folder name
        n, eta, sigma_y = extract_parameters(parameter_folder)
        print(f"Processing dataset: n={n}, eta={eta}, sigma_y={sigma_y}")   
        data = [n, eta, sigma_y]
    for setup in os.listdir(folder_path):
        setup_path = os.path.join(folder_path, setup)
        print(setup_path)
        if os.path.isdir(setup_path):
            height, width = extract_particle_setup(setup)
            print(f"Processing dataset: height={height}, width={width}")

            data = data + [height, width]
        
        # Initialize a list to store x_diff values for this dataset
        x_diffs = []

        # # Process the config files and extract their maximum x values
        for config_num in range(9):
            dat_file_name = f"config_0{config_num}.dat"
            dat_file_path = os.path.join(setup_path, dat_file_name)
            print(dat_file_path)
            if os.path.exists(dat_file_path):
                if config_num > 0:  # Skip config00 as it is the reference
                    
                    x_diff =np.float64(extract_particle_data_x(dat_file_path) - extract_particle_data_x(os.path.join(setup_path, "config_00.dat")))
                    x_diffs.append(x_diff)

        # Add the record for this dataset to the list (ensure the order corresponds to config numbering)
        data = data + x_diffs

    records.append(data)

# print(records)
# Define column names
columns = ['n', 'eta', 'sigma_y'] + [f'x_{i:02d}' for i in range(1, 21)]
# print(columns)

# Create the DataFrame from the records
df = pd.DataFrame(records, columns=columns)
# df = df.sort_values(by=['n', 'eta', 'sigma_y'])

# df['n'] = df['n'].apply(lambda x: f'{x:.8f}')
# df['eta'] = df['eta'].apply(lambda x: f'{x:.8f}')
# df['sigma_y'] = df['sigma_y'].apply(lambda x: f'{x:.8f}')

df.info()



csv_data_dir = 'x_diff_data.csv'
df.to_csv(csv_data_dir, index=False)

dat_dir = 'x_diff_data.dat'

#try to save df as biary file using pickle not csv
df.to_pickle(dat_dir)


# print(df)
