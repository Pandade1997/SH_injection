import numpy as np
import numpy
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import random
import os
import scipy.special as sp
import spaudiopy
import torch
import torch as th

import torch.utils.data as tud
from scipy import signal
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

# from loader.SHT import compute_steering_vector, mic_positions_spherical, microphone_positions_spherical

eps = np.finfo(np.float32).eps
channel = 16
L = 4


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip()
            path_list.append(tmp)


def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


def microphone_positions_spherical(cartesian_positions):
    positions_spherical = np.zeros(cartesian_positions.shape)
    for i, (x, y, z) in enumerate(cartesian_positions):
        positions_spherical[i] = cart2sph(x, y, z)
    return positions_spherical


def compute_steering_vector(positions_spherical, L, freq, c=343):
    N_mics = positions_spherical.shape[0]
    k = 2 * np.pi * freq / c
    steering_vector = np.zeros((N_mics, (L + 1) ** 2), dtype=np.complex64)
    for i, (r, theta, phi) in enumerate(positions_spherical):
        for l in range(L + 1):
            for m in range(-l, l + 1):
                Y = sp.sph_harm(m, l, theta, phi)
                steering_vector[i, l ** 2 + l + m] = Y * np.exp(-1j * k * r)
                # steering_vector[i, l ** 2 + l + m] = Y
    return steering_vector


class FixDataset(Dataset):

    def __init__(self,
                 wav_scp,
                 mix_dir,
                 ref_dir,
                 mic_dir,
                 repeat=1,
                 chunk=4,
                 sample_rate=16000,
                 sph_order=1,
                 ):
        super(FixDataset, self).__init__()

        self.wav_list = list()
        parse_scp(wav_scp, self.wav_list)
        self.mix_dir = mix_dir
        self.ref_dir = ref_dir
        self.segment_length = chunk * sample_rate
        self.wav_list *= repeat
        self.mic_dir = mic_dir
        self.n_fft = 512
        self.hop_length = 256
        self.window = torch.hamming_window(self.n_fft)
        self.sample_rate = 1.0
        self.sph_order = sph_order

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        utt_id = self.wav_list[index]
        mix_path = utt_id
        ref_path = os.path.join(self.ref_dir, utt_id.split('/')[-1])
        # print(utt_id.split('#')[2].split('_'))

        # L x C
        mix = audioread(mix_path)  # <class 'tuple'>: (64000, 16)
        ref = audioread(ref_path)

        # 计算球谐系数：
        # 增加麦克风位置,读取麦克风位置信息
        mic_path = os.path.join(self.mic_dir, 'mic_array_pos_' + utt_id.split('#')[2].split('_')[-2] + '_' +
                                utt_id.split('#')[2].split('_')[-1])
        mic_data = np.load(mic_path)  # <class 'tuple'>: (C, 3)

        # Compute the center of the array
        center = np.mean(mic_data, axis=0)
        # Subtract the center from each coordinate to get the new coordinates
        transformed_coords = mic_data - center

        mic_positions_spherical = self.microphone_positions_spherical(transformed_coords)
        colat = mic_positions_spherical[:, 2]  # 极角
        azi = mic_positions_spherical[:, 1]  # 方位角
        sh_type = 'real'

        coeffs = spaudiopy.sph.src_to_sh(mix.T, azi, colat, self.sph_order, sh_type)
        coeffs_target = spaudiopy.sph.src_to_sh(ref.T, azi, colat, self.sph_order, sh_type)

        egs = {
            "sph_input": np.float32(coeffs),
            "mix_input": np.float32(mix.T),
            "target": np.float32(coeffs_target),  # <class 'tuple'>: (16, 64000)
            # "steering_vectors": steering_vectors,
        }
        return egs

    def cart2sph(self, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return r, theta, phi

    def microphone_positions_spherical(self, cartesian_positions):
        positions_spherical = np.zeros(cartesian_positions.shape)
        for i, (x, y, z) in enumerate(cartesian_positions):
            positions_spherical[i] = self.cart2sph(x, y, z)
        return positions_spherical


def make_fix_loader(wav_scp, mix_dir, ref_dir, mic_dir, batch_size=8, repeat=1, num_workers=16,
                    chunk=4, sample_rate=16000, sph_order=1):
    dataset = FixDataset(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        mic_dir=mic_dir,
        repeat=repeat,
        chunk=chunk,
        sample_rate=sample_rate,
        sph_order=sph_order,
    )

    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )
    return loader
