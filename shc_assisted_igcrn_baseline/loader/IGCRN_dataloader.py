import numpy as np
import numpy
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import random
import os

import torch
import torch as th

import torch.utils.data as tud
from scipy import signal
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

eps = np.finfo(np.float32).eps
ft_len = 512
ft_overlap = 256
channel = 16


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


class FixDataset(Dataset):

    def __init__(self,
                 wav_scp,
                 mix_dir,
                 ref_dir,
                 repeat=1,
                 chunk=4,
                 sample_rate=16000,
                 ):
        super(FixDataset, self).__init__()

        self.wav_list = list()
        parse_scp(wav_scp, self.wav_list)
        self.mix_dir = mix_dir
        self.ref_dir = ref_dir
        self.segment_length = chunk * sample_rate
        self.wav_list *= repeat

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        utt_id = self.wav_list[index]
        mix_path = utt_id
        ref_path = os.path.join(self.ref_dir, utt_id.split('/')[-1])
        # L x C
        mix = audioread(mix_path)
        ref = audioread(ref_path)

        egs = {
            "input": np.float32(mix.T),  # <class 'tuple'>: (16, 64000)
            "target": np.float32(ref.T),  # <class 'tuple'>: (16, 64000)
        }
        return egs


def make_fix_loader(wav_scp, mix_dir, ref_dir, batch_size=8, repeat=1, num_workers=16,
                    chunk=4, sample_rate=16000):
    dataset = FixDataset(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        repeat=repeat,
        chunk=chunk,
        sample_rate=sample_rate,
    )

    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )
    return loader


def test_loader():
    # wav_scp是mix的路径：ls -R /ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/generated_data/train/mix/*.wav > /home/imu_panjiahui/SHdomain/SHC_baseline/loader/wav_scp_train.txt
    wav_scp = '/home/imu_panjiahui/SHdomain/SHC_baseline/loader/wav_scp_train.txt'
    mix_dir = '/ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/generated_data/train/mix'
    ref_dir = '/ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/generated_data/train/noreverb_ref'
    repeat = 1
    num_worker = 16
    chunk = 6
    sample_rate = 16000
    batch_size = 16

    loader = make_fix_loader(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        batch_size=batch_size,
        repeat=repeat,
        num_workers=num_worker,
        chunk=chunk,
        sample_rate=sample_rate,
    )

    cnt = 0
    print('len: ', len(loader))
    for idx, egs in enumerate(loader):
        # B X C X 2 X F X T --> B X F X 2 X C X T
        mix = egs['mix'].transpose(1, 3)
        target = egs['ref'].transpose(1, 3)

        # 将前两个维度合并为一个新的维度
        mix = mix.reshape(batch_size * int(ft_len / 2 + 1), 2, channel, mix.shape[-1])
        target = target.reshape(batch_size * int(ft_len / 2 + 1), 2, channel, target.shape[-1])

        cnt = cnt + 1
        print('cnt: {}'.format(cnt))
        print('egs["mix"].shape: ', egs["mix"].shape)
        print('egs["ref"].shape: ', egs["ref"].shape)
        print()
        # sf.write('./data/input/inputs_' + str(cnt) + '.wav', egs["mix"][0], 16000)
        # sf.write('./data/label/inputs_' + str(cnt) + '.wav', egs["ref"][0], 16000)
        if cnt >= 10:
            break
    print('done!')


if __name__ == "__main__":
    test_loader()
