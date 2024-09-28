
# import os

# # Define the directory path
# results_dir = 'C:/automatic_vehicular_control/results/highway_ramp/plots'

# def check_file_access(directory):
#     try:
#         subdirectories = os.listdir(directory)
#         for subdirectory in subdirectories:
#             subdir_path = os.path.join(directory, subdirectory)
#             if os.path.isdir(subdir_path):
#                 print(f"Checking directory: {subdir_path}")
#                 files = os.listdir(subdir_path)
#                 for filename in files:
#                     file_path = os.path.join(subdir_path, filename)
#                     try:
#                         print(f"Checking file: {file_path}")
#                         with open(file_path, 'rb') as file:
#                             print(f"Successfully accessed file: {file_path}")
#                     except PermissionError as e:
#                         print(f"Permission error while accessing file: {file_path}")
#                         print(e)
#                     except FileNotFoundError as e:
#                         print(f"File not found error while accessing file: {file_path}")
#                         print(e)
#                     except IsADirectoryError as e:
#                         print(f"Is a directory error while accessing file: {file_path}")
#                         print(e)
#                     except IOError as e:
#                         print(f"IO error while accessing file: {file_path}")
#                         print(e)
#             else:
#                 print(f"{subdir_path} is not a directory, skipping.")
#     except Exception as e:
#         print(f"An error occurred while listing files in directory: {directory}")
#         print(e)

# # Check file access
# check_file_access(results_dir)


import os
import zipfile

results_dir = 'C:/automatic_vehicular_control/results/highway_ramp/plots'

def inspect_zip_files(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            npz_path = os.path.join(subdir_path, 'trajectories.npz')
            if os.path.exists(npz_path):
                try:
                    with zipfile.ZipFile(npz_path, 'r') as zip_ref:
                        print(f"Contents of {npz_path}:")
                        zip_ref.printdir()
                        for file_name in zip_ref.namelist():
                            with zip_ref.open(file_name) as file:
                                print(f"{file_name}: {file.read().decode('utf-8', errors='ignore')[:100]}...")  # Print the first 100 characters
                except Exception as e:
                    print(f"Failed to read {npz_path} as a zip file: {e}")

inspect_zip_files(results_dir)

