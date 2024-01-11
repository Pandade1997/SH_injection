import os
import scipy.io as sio
import numpy as np

# Adjust the printing options to display the full content including decimal points
np.set_printoptions(suppress=False, threshold=np.inf, precision=None, linewidth=200)

# Replace 'your_test_name' with the actual test name
test_name = 'mic_16'
# Replace 'path_to_your_directory' with the actual path to your directory
res1path = '/home/imu_panjiahui/SH_generalization_num/shc_assisted_igcrn_baseline_pro2/model_miso_new_data/result_model_best/'

# Construct the full file path
file_path = os.path.join(res1path, test_name + '_metrics.mat')

# Read and print the contents of the .mat file
if os.path.isfile(file_path):
    data = sio.loadmat(file_path)
    for key, value in data.items():
        print(f"{key}: {value}")
else:
    print(f"The file {file_path} does not exist.")
