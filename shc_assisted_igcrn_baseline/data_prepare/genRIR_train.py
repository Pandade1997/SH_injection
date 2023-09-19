import random

import numpy as np
import scipy.io.wavfile as wavfile
import os
from scipy import signal
import gpuRIR
import soundfile as sf


def genRIR(RIR_data_path, MIC_data_path, num_rirs):
    fs = 16000
    # 设置房间大小和RT60范围
    room_size = [6, 5, 4]
    t60 = np.random.uniform(0.2, 1, size=(num_rirs, 1))  # 训练集的混响：0.2-1

    # 生成麦克风位置
    mic_array_radius = 0.035  # 10cm
    num_mics = 9
    mic_theta = np.arange(0, 2 * np.pi, 2 * np.pi / num_mics)
    mic_x = mic_array_radius * np.cos(mic_theta)
    mic_y = mic_array_radius * np.sin(mic_theta)
    mic_z = np.zeros(num_mics) + 1  # 麦克风高度为1米

    # 随机生成麦克风中心位置
    mic_center = np.random.rand(3) * (np.array(room_size) - 2 * mic_array_radius) + mic_array_radius

    # 将麦克风位置从极坐标转换为直角坐标
    mic_pos = np.stack((mic_x, mic_y, mic_z), axis=1) + mic_center

    while np.any(mic_pos > room_size) or np.any(mic_pos < 0):
        # 随机生成麦克风中心位置
        mic_center = np.random.rand(3) * (np.array(room_size) - 2 * mic_array_radius) + mic_array_radius

        # 将麦克风位置从极坐标转换为直角坐标
        mic_pos = np.stack((mic_x, mic_y, mic_z), axis=1) + mic_center

    # 生成5000个RIR
    for i in range(num_rirs):
        print(i)
        # 设置随机声源位置
        src_pos_1 = np.random.rand(3) * (np.array(room_size) - 2) + 1  # 随机位置，距离麦克风阵列中心1米
        while np.any(src_pos_1 + 1 > room_size) or np.any(src_pos_1 - 1 < 0):  # 避免声源位置超出房间范围
            src_pos_1 = np.random.rand(3) * (np.array(room_size) - 2) + 1

        # 设置随机噪声位置
        src_pos_2 = np.random.rand(3) * (np.array(room_size) - 2) + 1  # 随机位置，距离麦克风阵列中心1米
        while np.any(src_pos_2 + 1 > room_size) or np.any(src_pos_2 - 1 < 0):  # 避免声源位置超出房间范围
            src_pos_2 = np.random.rand(3) * (np.array(room_size) - 2) + 1

        # 生成RIR
        nb_img = gpuRIR.t2n(0.375, room_size)
        beta = gpuRIR.beta_SabineEstimation(room_size, t60[i])
        # print(beta)

        rir_1 = gpuRIR.simulateRIR(room_sz=room_size, pos_src=np.array([list(src_pos_1)]), pos_rcv=mic_pos, beta=beta,
                                   nb_img=nb_img,
                                   fs=fs, Tmax=0.375, mic_pattern='omni')
        rir_2 = gpuRIR.simulateRIR(room_sz=room_size, pos_src=np.array([list(src_pos_2)]), pos_rcv=mic_pos, beta=beta,
                                   nb_img=nb_img,
                                   fs=fs, Tmax=0.375, mic_pattern='omni')

        # rir = gpuRIR.simulateRIR(room_sz=room_size, pos_src=src_pos, pos_rcv=mic_pos, beta=beta,
        #                          nb_img=nb_img,
        #                          fs=fs, Tmax=0.375, mic_pattern='omni')

        rir = np.zeros([num_mics * 2, rir_1.shape[-1]])
        rir[0:num_mics, :] = rir_1.squeeze()
        rir[num_mics:num_mics * 2, :] = rir_2.squeeze()
        # print(rir.shape)

        # 保存RIR和麦克风位置信息
        np.save(MIC_data_path + '/mic_array_pos_' + str(i) + '_' + str(t60[i]) + '.npy', mic_pos)
        np.save(RIR_data_path + '/rir_' + str(i) + '_' + str(t60[i]) + '.npy', rir)

        # 将RIR写为音频文件
        # rir = rir / np.max(np.abs(rir))  # 归一化到 [-1, 1]
        # rir = np.int16(rir * 32767)  # 转换为16位整数
        # wavfile.write(RIR_data_path + '/rir_%d.wav' % i, 16000, rir.T)
        # print(222222)

        # 弧形
        # L = 10000
        # audio = np.zeros((1, L))
        # audio[:, 0] = 1  # 第一帧为1
        # print(rir.shape)
        # output_audio = signal.fftconvolve(rir[0, :, :], audio)[:, :L]
        # output_audio = output_audio / np.max(np.abs(output_audio))
        # sf.write(RIR_data_path + '/out_%d.wav' % i, output_audio.T, 16000)
    return True


def loadRIR(load_path):
    # 读取RIR npy文件
    data = np.load(load_path)
    # print(1111111111)

    return True


if __name__ == '__main__':
    # 定义目录名
    out_path = '/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/RIR/cir_uniform_9/train_rir/'
    txt_path = '/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/loader_txt_cir_uniform_9/'
    RIR_data_path = os.path.join(out_path, "RIR")
    MIC_data_path = os.path.join(out_path, "MIC")
    num_rirs = 80000

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(RIR_data_path):
        os.makedirs(RIR_data_path)
    if not os.path.exists(MIC_data_path):
        os.makedirs(MIC_data_path)
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    # 1、生成房间冲激响应，在V100上运行
    genRIR(RIR_data_path, MIC_data_path, num_rirs)

    # 2、把所有rir和mic写入txt中
    # 获取文件夹内所有文件路径
    RIR_file_paths = []
    for root, dirs, files in os.walk(RIR_data_path):
        for file in files:
            file_path = os.path.join(root, file)
            RIR_file_paths.append(file_path)

    MIC_file_paths = []
    for root, dirs, files in os.walk(MIC_data_path):
        for file in files:
            file_path = os.path.join(root, file)
            MIC_file_paths.append(file_path)

    # 将文件路径写入txt文件
    RIR_txt_path = os.path.join(txt_path, "train_rir.txt")
    with open(RIR_txt_path, "w") as f:
        for file_path in RIR_file_paths:
            f.write(file_path + "\n")

    MIC_txt_path = os.path.join(txt_path, "train_mic.txt")
    with open(MIC_txt_path, "w") as f:
        for file_path in MIC_file_paths:
            f.write(file_path + "\n")

