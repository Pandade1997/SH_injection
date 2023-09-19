# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:26:54 2020

@author: admin
"""
import random
import warnings
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import argparse
from tqdm import tqdm
from loader.dataloader import make_fix_loader
from networks.IGCRN import IGCRN

warnings.filterwarnings("ignore")
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser("PW-NBDF base")

parser.add_argument('--train_wav_scp', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/train/wav_scp_train_8W.txt')
parser.add_argument('--train_mix_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/train/mix')
parser.add_argument('--train_ref_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/train/noreverb_ref')
parser.add_argument('--train_mic_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/RIR/cir_uniform_9/train_rir/MIC')

parser.add_argument('--val_wav_scp', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/val/wav_scp_val_4k.txt')
parser.add_argument('--val_mix_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/val/mix')
parser.add_argument('--val_ref_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/val/noreverb_ref')
parser.add_argument('--val_mic_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/RIR/cir_uniform_9/val_rir/MIC')

parser.add_argument('--gpuid', type=int, default=0, help='Using which gpu')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epoch for training')
parser.add_argument('--num_worker', type=int, default=6, help='Num_workers: if on PC, set 0')
parser.add_argument('--lr', type=float, default=1e-3, help='Fine tuning learning rate')
parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
parser.add_argument('--fft_len', type=int, default=512)
parser.add_argument('--channel', type=int, default=9)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--chunk', type=int, default=4)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--sph_order', type=int, default=4)

args = parser.parse_args()
NowTime = time.localtime()

if __name__ == "__main__":

    iter_count = 0
    batch_size = args.batch_size
    num_worker = args.num_worker
    fft_len = args.fft_len
    channel = args.channel
    repeat = args.repeat
    chunk = args.chunk
    sample_rate = args.sample_rate
    sph_order = args.sph_order
    sph_channel = (sph_order + 1) ** 2

    train_wav_scp = args.train_wav_scp
    train_mix_dir = args.train_mix_dir
    train_ref_dir = args.train_ref_dir
    train_mic_dir = args.train_mic_dir
    val_wav_scp = args.val_wav_scp
    val_mix_dir = args.val_mix_dir
    val_ref_dir = args.val_ref_dir
    val_mic_dir = args.val_mic_dir


    def count_parameters(network):
        return sum(p.numel() for p in network.parameters() if p.requires_grad)


    print("##################### Trainning  model ###########################")

    network = IGCRN(in_ch=sph_channel)
    print(f'The model has {count_parameters(network):,} trainable parameters')

    network = torch.nn.DataParallel(network)
    network = network.cuda()

    # network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    loss_function = nn.MSELoss()

    writer = SummaryWriter('runs/Fine_tuning_{}/'.format(time.strftime("%Y-%m-%d-%H-%M-%S", NowTime)))

    modelpath = 'lock_models_order_4_position_random/'  # 正确的极角和方位角
    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)

    loss_train_epoch = []
    loss_val_epoch = []

    loss_train_sequence = []
    loss_val_sequence = []
    # add:
    min_val_loss = float("inf")
    val_no_impv = 0

    for epoch in range(args.num_epoch):

        train_loader = make_fix_loader(
            wav_scp=train_wav_scp,
            mix_dir=train_mix_dir,
            ref_dir=train_ref_dir,
            mic_dir=train_mic_dir,
            batch_size=batch_size,
            repeat=repeat,
            num_workers=num_worker,
            chunk=chunk,
            sample_rate=sample_rate,
            sph_order=sph_order,
        )

        val_loader = make_fix_loader(
            wav_scp=val_wav_scp,
            mix_dir=val_mix_dir,
            ref_dir=val_ref_dir,
            mic_dir=val_mic_dir,
            batch_size=batch_size,
            repeat=repeat,
            num_workers=num_worker,
            chunk=chunk,
            sample_rate=sample_rate,
            sph_order=sph_order,
        )

        print("############################ Epoch {} ################################".format(epoch + 1))
        ############# Train ############################################################################################################

        network.train()  # set the network in train mode

        for idx, egs in tqdm(enumerate(train_loader), total=len(train_loader)):
            # B X C X T
            sph_input = egs['sph_input'].cuda()
            target = egs['target'].cuda()

            outputs = network(sph_input)

            # compute loss
            loss = loss_function(outputs, target)
            loss_train_sequence.append(loss.detach().cpu().numpy())

            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalars('Loss', {"Train": loss.item()}, iter_count)
            iter_count += 1

        loss_train_epoch.append(np.mean(loss_train_sequence[epoch * len(train_loader):(epoch + 1) * len(train_loader)]))

        ############# Validation ######################################################################################################
        network.eval()

        for idx, egs in tqdm(enumerate(val_loader), total=len(val_loader)):
            sph_input = egs['sph_input'].cuda()
            target = egs['target'].cuda()

            with torch.no_grad():
                outputs = network(sph_input)

            # compute loss
            loss_val = loss_function(outputs, target)

            loss_val_sequence.append(loss_val.detach().cpu().numpy())

        loss_val_epoch.append(np.mean(loss_val_sequence[epoch * len(val_loader):(epoch + 1) * len(val_loader)]))

        ############## Save Model ######################################################################################################
        torch.save(network.state_dict(), modelpath + 'network_epoch{}.pth'.format(epoch + 1))

        ############## lr halve and Early stop ######################################################################################################
        new_loss = loss_val_epoch[epoch]
        if new_loss <= min_val_loss:
            min_val_loss = loss_val_epoch[epoch]
            val_no_impv = 0
            torch.save(network.state_dict(), modelpath + 'model_best.pth')
        else:
            val_no_impv += 1
            optim_state = optimizer.state_dict()
            optim_state["param_groups"][0]["lr"] = optim_state["param_groups"][0]["lr"] / 2.0
            optimizer.load_state_dict(optim_state)
            print("Learning rate is adjusted to %5f" % (optim_state["param_groups"][0]["lr"]))
            if val_no_impv >= 5:
                print("No improvements and apply early-stopping")
                break

        ############## Loss evaluation ######################################################################################################
        np.save(modelpath + 'loss_val_epoch.npy', loss_val_epoch)
        np.save(modelpath + 'loss_train_epoch.npy', loss_train_epoch)

        curves = [loss_train_epoch, loss_val_epoch]
        labels = ['train_loss', 'val_loss']

        f1 = plt.figure(epoch + 1)
        plt.title("MSELoss of general model")
        plt.xlabel('Epoch')
        plt.ylabel('MSEloss')
        for i, curve in enumerate(curves):
            plt.plot(curve, label=labels[i])
        plt.legend()
        f1.savefig(modelpath + 'Network_loss.png')

        writer.add_scalars('Loss', {"Validation": loss_val_epoch[epoch]}, iter_count)
