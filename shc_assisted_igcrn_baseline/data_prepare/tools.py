import os
import shutil
import random
import scipy.io
import numpy as np


def move(dir_A, dir_B, percent):
    # 如果目录 B 不存在，则创建目录 B
    if not os.path.exists(dir_B):
        os.makedirs(dir_B)

    # 获取目录 A 中所有 .wav 文件的路径
    wav_files = [f for f in os.listdir(dir_A) if f.endswith('.WAV')]

    # 计算要提取的文件数
    num_files_to_extract = int(len(wav_files) * percent)

    # 随机选择要提取的文件
    files_to_extract = random.sample(wav_files, num_files_to_extract)

    # 将选定的文件移动到目录 B
    for file_name in files_to_extract:
        file_path = os.path.join(dir_A, file_name)
        dest_path = os.path.join(dir_B, file_name)
        shutil.move(file_path, dest_path)
    return True


def split_txt(input_path, part):
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
        num_lines = len(lines)
        chunk_size = num_lines // part
        rir_list = ['0.2', '0.4', '0.6']
        snr_list = ['-5', '-2', '0', '2', '5']
        i = 0
        for snr in snr_list:
            for rir in rir_list:
                start_index = i * chunk_size
                end_index = (i + 1) * chunk_size
                if i == part - 1:
                    end_index = num_lines
                chunk = lines[start_index:end_index]
                i += 1
                with open(
                        f'/ddnstor/imu_wutc/dataset/pjh/SHdomainlarge/loader_txt/test_clean_t60_{rir}_snr{snr}_list.txt',
                        'w') as output_file:
                    output_file.writelines(chunk)


def gen_wav_scp(file_dir, txt_path):
    file_paths = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    with open(txt_path, "w") as f:
        for file_path in file_paths:
            f.write(file_path + "\n")


def compare_folders(folder1, folder2, folder3):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    files3 = set(os.listdir(folder3))

    # 找到不一致的文件
    diff_files = (files1 ^ files2) | (files1 ^ files3) | (files2 ^ files3)

    for file in diff_files:
        file_path1 = os.path.join(folder1, file)
        file_path2 = os.path.join(folder2, file)
        file_path3 = os.path.join(folder3, file)

        # 删除不一致的文件
        if os.path.exists(file_path1):
            print(file_path1)
            # os.remove(file_path1)
        if os.path.exists(file_path2):
            print(file_path2)
            # os.remove(file_path2)
        if os.path.exists(file_path3):
            print(file_path3)
            # os.remove(file_path3)
    return True


def merge_files(folder_path, output_file):
    rir_list = ['1.0', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    snr_list = ['-10', '-5', '0', '5', '10']

    merged_content = ""
    for snr in snr_list:
        for rir in rir_list:
            txt_path = folder_path + 'wav_scp_test_' + rir + '_snr' + snr + '.txt'
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as file:
                    file_content = file.read()
                    merged_content += file_content
    with open(output_file, "w") as merged_file:
        merged_file.write(merged_content)
    return True


def extract_lines(source_file, target_file, num_lines):
    with open(source_file, 'r') as source:
        with open(target_file, 'w') as target:
            for i, line in enumerate(source):
                if i < num_lines:
                    target.write(line)
                else:
                    break
    return True


def extract_lines_random(source_file, target_file, num_lines):
    import random
    # 打开A.txt文件进行读取
    with open(source_file, 'r') as file:
        lines = file.readlines()
    # 随机选择300,000行内容
    selected_lines = random.sample(lines, num_lines)
    # 打开b.txt文件进行写入
    with open(target_file, 'w') as file:
        file.writelines(selected_lines)
    print('内容写入完成！')
    return True


def copy_fileA_part_to_fileB(source_folder, target_folder, num):
    import os
    import random
    import shutil

    # 源文件夹和目标文件夹路径
    source_folder = source_folder
    target_folder = target_folder
    num = num

    # 获取源文件夹中所有.wav文件的路径列表
    wav_files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]

    # 随机选择3万个.wav文件
    selected_files = random.sample(wav_files, num)

    # 复制选中的文件到目标文件夹
    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.copyfile(source_path, target_path)

    print('文件复制完成！')


# 求圆形阵列的半径
import itertools
import math


def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def calculate_average_distance(mic_coords, mic1, mic2):
    total_distance = 0
    for coord in mic_coords:
        total_distance += euclidean_distance(coord, mic1) + euclidean_distance(coord, mic2)
    return total_distance / (2 * len(mic_coords))


def find_array_radius(mic_coords):
    min_average_distance = float('inf')
    min_mic_pair = None

    for mic1, mic2 in itertools.combinations(mic_coords, 2):
        avg_distance = calculate_average_distance(mic_coords, mic1, mic2)
        if avg_distance < min_average_distance:
            min_average_distance = avg_distance
            min_mic_pair = (mic1, mic2)

    diameter = euclidean_distance(min_mic_pair[0], min_mic_pair[1])
    radius = diameter / 2
    return radius


if __name__ == '__main__':
    flag = 7
    if flag == 1:
        # 定义目录 A 和目录 B 的路径
        dir_A = '/ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/Data/train_val_clean/'
        dir_B = '/ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/Data/val_clean/'

        # 1、按照比例移动A的内容到B中
        # 定义要提取的文件比例（这里是 10%）
        percent = 0.1
        move(dir_A, dir_B, percent)

    elif flag == 2:
        # 2、移动目录A 到目录B中
        # 使用 shutil.move() 函数将目录 A 移动到目录 B 中
        dir_A = '/home/imu_panjiahui/SHdomain/SHC_baseline/data_prepare/MIC_data/'
        dir_B = '/ddnstor/imu_panjiahui/dataset/SHdomain/'
        shutil.move(dir_A, dir_B)

    elif flag == 3:
        # 3、A.txt里的内容分为5部分 分别写入五个.txt文件中
        input_path = '/ddnstor/imu_wutc/dataset/pjh/SHdomainlarge/loader_txt/test_clean_list.txt'
        part = 15
        split_txt(input_path, part)

    elif flag == 4:
        # 4、把生成的数据路径写入wav_scp
        file_dir = '/data01/data_pjh/SHdomainNew/generated_data_cir_uniform_16/train/mix'
        txt_path = '/data01/data_pjh/SHdomainNew/loader_txt/wav_scp_train.txt'
        gen_wav_scp(file_dir, txt_path)

    elif flag == 5:
        # 5、比较并删除不一致的文件
        folderA = '/data01/data_pjh/SHdomainlarge/generated_data_cir_uniform_9/train/mix'
        folderB = '/data01/data_pjh/SHdomainlarge/generated_data_cir_uniform_9/train/noreverb_ref'
        folderC = '/data01/data_pjh/SHdomainlarge/generated_data_cir_uniform_9/train/reverb_ref'
        compare_folders(folderA, folderB, folderC)



    elif flag == 6:
        # 6、将文件夹内的 几个txt 文件的内容合并到一个新的 merged.txt 文件中：
        folder_path = '/data01/data_pjh/SHdomainNew/loader_txt/'
        output_file = '/data01/data_pjh/SHdomainNew/loader_txt/wav_scp_test.txt'
        merge_files(folder_path, output_file)

    elif flag == 7:
        # 7、加载.mat文件并输出
        # test_list = [
        #     'test_t60_0.2_snr-5', 'test_t60_0.2_snr0', 'test_t60_0.2_snr5',
        #     'test_t60_0.3_snr-5', 'test_t60_0.3_snr0', 'test_t60_0.3_snr5',
        #     'test_t60_0.4_snr-5', 'test_t60_0.4_snr0', 'test_t60_0.4_snr5',
        #     'test_t60_0.5_snr-5', 'test_t60_0.5_snr0', 'test_t60_0.5_snr5',
        #     'test_t60_0.6_snr-5', 'test_t60_0.6_snr0', 'test_t60_0.6_snr5',
        #     'test_t60_0.7_snr-5', 'test_t60_0.7_snr0', 'test_t60_0.7_snr5',
        #     'test_t60_0.8_snr-5', 'test_t60_0.8_snr0', 'test_t60_0.8_snr5',
        #     'test_t60_0.9_snr-5', 'test_t60_0.9_snr0', 'test_t60_0.9_snr5',
        #     # 'test_t60_1.0_snr-5', 'test_t60_1.0_snr0', 'test_t60_1.0_snr5',
        # ]
        test_list = ['test_0.2_snr-5', 'test_0.2_snr0', 'test_0.2_snr5',
                     'test_0.3_snr-5', 'test_0.3_snr0', 'test_0.3_snr5',
                     'test_0.4_snr-5', 'test_0.4_snr0', 'test_0.4_snr5',
                     'test_0.5_snr-5', 'test_0.5_snr0', 'test_0.5_snr5',
                     'test_0.6_snr-5', 'test_0.6_snr0', 'test_0.6_snr5',
                     ]
        snr_list = [-5, 0, 5]
        t60_list = [0.2, 0.3, 0.4, 0.5, 0.6]

        for snr in snr_list:
            mix_avg = 0
            gcrn_avg = 0
            igcrn_avg = 0
            time_avg = 0
            assiated_avg = 0
            for t60 in t60_list:
                test_name = 'test_' + str(t60) + '_snr' + str(snr)
                gcrn_path = '/home/imu_panjiahui/SHdomain_3_Introducing_SHC/shc_assisted_gcrn_baseline/GCRN_baseline_mini/result_model_best/' + test_name  # final
                igcrn_path = '/home/imu_panjiahui/SHdomain_3_Introducing_SHC/shc_assisted_igcrn_baseline/Lock_baseline_mini/result_model_best/' + test_name  # final
                pro_time_path = '/home/imu_panjiahui/SHdomain_3_Introducing_SHC/shc_assisted_igcrn_propose_time_srcTosh_center/lock_models_order_4_position_mini/result_model_best/' + test_name  # final
                pro_assisted_path = '/home/imu_panjiahui/SHdomain_3_Introducing_SHC/shc_assisted_igcrn_propose_independentEncoder_srcTosh_center/lock_models_order_4_mini/result_model_best/' + test_name  # final

                gcrn = scipy.io.loadmat(gcrn_path + '_metrics.mat')
                igcrn = scipy.io.loadmat(igcrn_path + '_metrics.mat')
                pro_time = scipy.io.loadmat(pro_time_path + '_metrics.mat')
                pro_assisted = scipy.io.loadmat(pro_assisted_path + '_metrics.mat')

                # PESQ
                # mix_avg += float("{:.2f}".format(float(gcrn['pesq_mix'])))
                # gcrn_avg += float("{:.2f}".format(float(gcrn['pesq_est'])))
                # igcrn_avg += float("{:.2f}".format(float(igcrn['pesq_est'])))
                # time_avg += float("{:.2f}".format(float(pro_time['pesq_est'])))
                # assiated_avg += float("{:.2f}".format(float(pro_assisted['pesq_est'])))

            #     print(test_name + " PESQ:")
            #     print(
            #         'mix:' + "{:.2f}".format(float(gcrn['pesq_mix'])) + ' '
            #         + 'gcrn:' + "{:.2f}".format(float(gcrn['pesq_est'])) + ' '
            #         + 'igcrn:' + "{:.2f}".format(float(igcrn['pesq_est'])) + ' '
            #         + 'pro_time:' + "{:.2f}".format(float(pro_time['pesq_est'])) + ' '
            #         + 'pro_assisted:' + "{:.2f}".format(float(pro_assisted['pesq_est'])) + ' '
            #
            #     )
            # print('SNR = ' + str(snr))
            # print('mix_avg : ' + str("{:.2f}".format(mix_avg / 5)))
            # print('gcrn_avg : ' + str("{:.2f}".format(gcrn_avg / 5)))
            # print('igcrn_avg : ' + str("{:.2f}".format(igcrn_avg / 5)))
            # print('time_avg : ' + str("{:.2f}".format(time_avg / 5)))
            # print('assiated_avg : ' + str("{:.2f}".format(assiated_avg / 5)))

                # stoi
                mix_avg += float("{:.2f}".format(float(gcrn['stoi_mix'])* 100))
                gcrn_avg += float("{:.2f}".format(float(gcrn['stoi_est'])* 100))
                igcrn_avg += float("{:.2f}".format(float(igcrn['stoi_est'])* 100))
                time_avg += float("{:.2f}".format(float(pro_time['stoi_est'])* 100))
                assiated_avg += float("{:.2f}".format(float(pro_assisted['stoi_est'])* 100))

                print(test_name + " STOI:")
                print(
                    'mix:' + "{:.2f}".format(float(gcrn['stoi_mix']) * 100) + ' '
                    + 'gcrn:' + "{:.2f}".format(float(gcrn['stoi_est']) * 100) + ' '
                    + 'igcrn:' + "{:.2f}".format(float(igcrn['stoi_est']) * 100) + ' '
                    + 'pro_time:' + "{:.2f}".format(float(pro_time['stoi_est']) * 100) + ' '
                    + 'pro_assisted:' + "{:.2f}".format(float(pro_assisted['stoi_est']) * 100) + ' '

                )
            print('SNR = ' + str(snr))
            print('mix_avg : ' + str("{:.2f}".format(mix_avg / 5)))
            print('gcrn_avg : ' + str("{:.2f}".format(gcrn_avg / 5)))
            print('igcrn_avg : ' + str("{:.2f}".format(igcrn_avg / 5)))
            print('time_avg : ' + str("{:.2f}".format(time_avg / 5)))
            print('assiated_avg : ' + str("{:.2f}".format(assiated_avg / 5)))



    elif flag == 8:
        # 8、提取A.txt前80000行的内容，写入b.txt
        source_file = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt_cir_uniform_9/wav_scp_train.txt'
        target_file = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt_cir_uniform_9/wav_scp_train_23670.txt'
        num_lines = 23670
        extract_lines(source_file, target_file, num_lines)

        # 随机提取A.txt中的300000行的内容，写入b.txt
        source_file = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt_cir_uniform_9/wav_scp_val.txt'
        target_file = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt_cir_uniform_9/wav_scp_val_2580.txt'
        num_lines = 2580
        extract_lines_random(source_file, target_file, num_lines)

    elif flag == 9:
        mic_path = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/RIR/cir_uniform_9/train_val_rir/MIC/mic_array_pos_9_0.4.npy'
        mic_data = np.load(mic_path)  # <class 'tuple'>: (C, 3)
        # Example microphone coordinates: [(x1, y1, z1), (x2, y2, z2), ...]
        mic_coordinates = mic_data
        array_radius = find_array_radius(mic_coordinates)
        print("Array Radius:", array_radius)
