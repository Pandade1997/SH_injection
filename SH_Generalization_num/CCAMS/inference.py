# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:18 2020

@author: admin
"""
import shutil

import argparse
import librosa
import torch
import os, fnmatch
import numpy as np
import soundfile as sf
from scipy import signal, io
from tqdm import tqdm
from evaluation_fixed import NB_PESQ, STOI
from networks.IGCRN import IGCRN
import warnings
from multiprocessing import Pool
from config.train_config import *
from networks.enhancer import PonderEnhancer

warnings.filterwarnings("ignore")

device = 'cpu'


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def calculate_metrics(ref, mix, est):
    return NB_PESQ(ref, mix), NB_PESQ(ref, est), STOI(ref, mix), STOI(ref, est)


def wav_generator(mix_path, ref_path):
    mix = audioread(mix_path)
    ref = audioread(ref_path)

    input = torch.tensor(np.float32(mix.T)).unsqueeze(0).to(device)
    load_network.to(device)  # Make sure your network is on GPU
    load_network.eval()
    with torch.no_grad():
        outputs, ponder = load_network(input)
    est = outputs.squeeze().T.cpu().detach().numpy()

    # sf.write(modelpath + 'clean.wav', ref, 16000)
    # sf.write(modelpath + 'mix.wav', mix, 16000)
    # sf.write(modelpath + 'est.wav', est, 16000)

    pesq_mix, pesq_est, stoi_mix, stoi_est = calculate_metrics(ref[:, 0], mix[:, 0], est)
    pesq_mix = np.mean(np.array(pesq_mix))
    pesq_est = np.mean(np.array(pesq_est))
    stoi_mix = np.mean(np.array(stoi_mix))
    stoi_est = np.mean(np.array(stoi_est))
    return pesq_mix, pesq_est, stoi_mix, stoi_est


if __name__ == "__main__":
    modelpath = 'model_miso_new_data/'

    # test_list = ['test_mic_8', 'test_mic_4', 'test_mic_12', 'test_mic_16']
    # test_list = ['test_mic_4']
    # test_list = ['test_mic_8']
    # test_list = ['test_mic_12']
    test_list = ['test_mic_16']
    file_path = '/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir'

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="complex_ponder_med_k_9_mostdb_5_warmup")
    params_function = parser.parse_args().cfg
    params = globals()[params_function]()

    for test_name in test_list:
        test_wav_scp = os.path.join(file_path, 'loader_txt', 'wav_scp', 'wav_scp_' + test_name + '.txt')
        wav_path = os.path.join(file_path, 'generated_data', test_name, 'mix')
        ref_dir = os.path.join(file_path, 'generated_data', test_name, 'noreverb_ref')

        modelname = modelpath + 'model_best.pth'

        print(str(modelname))
        print("Processing test ..." + str(test_name))

        load_network = PonderEnhancer(params['model'])

        # new_state_dict = {}
        # for k, v in torch.load(str(modelname), map_location=device).items():
        #     new_state_dict[k[7:]] = v  # 键值包含‘module.’ 则删除
        # load_network.load_state_dict(new_state_dict, strict=False)
        # load_network = load_network.to(device)

        # load_network = torch.nn.DataParallel(load_network)
        load_network.load_state_dict(torch.load(modelname))
        load_network = load_network.cuda()

        pesq_mix_list = []
        pesq_est_list = []
        stoi_mix_list = []
        stoi_est_list = []
        with open(test_wav_scp, 'r', encoding='utf-8') as infile:
            data = infile.readlines()
            for i in tqdm(range(len(data))):
                utt_id = data[i].strip("\n").split('/')[-1]
                mix_path = os.path.join(wav_path, utt_id)
                ref_path = os.path.join(ref_dir, utt_id)
                try:
                    pesq_mix, pesq_est, stoi_mix, stoi_est = wav_generator(mix_path, ref_path)
                    pesq_mix_list.append(pesq_mix)
                    pesq_est_list.append(pesq_est)
                    stoi_mix_list.append(stoi_mix)
                    stoi_est_list.append(stoi_est)
                except:
                    print(utt_id)
                    pesq_mix_list.append(pesq_mix_list[0])
                    pesq_est_list.append(pesq_est_list[0])
                    stoi_mix_list.append(stoi_mix_list[0])
                    stoi_est_list.append(stoi_est_list[0])

        pesq_mix = np.mean(np.array(pesq_mix_list))
        pesq_est = np.mean(np.array(pesq_est_list))
        stoi_mix = np.mean(np.array(stoi_mix_list))
        stoi_est = np.mean(np.array(stoi_est_list))

        # print("model_best_pesq_result:")
        print(test_name + "_result:")
        print('pesq_mix:' + str(pesq_mix) + '   ' + 'pesq_est:' + str(pesq_est) + '   ' + 'stoi_mix:' + str(
            stoi_mix) + '   ' + 'stoi_est:' + str(stoi_est))

        res1path = modelpath + '/result_model_best/'
        if not os.path.isdir(res1path):
            os.makedirs(res1path)
        io.savemat(res1path + test_name + '_metrics.mat',
                   {'pesq_mix': pesq_mix, 'pesq_est': pesq_est, 'stoi_mix': stoi_mix, 'stoi_est': stoi_est})
