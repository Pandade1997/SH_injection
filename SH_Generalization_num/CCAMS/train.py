# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:26:54 2020

@author: admin
"""
import random
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import argparse
from tqdm import tqdm
import warnings
from loader.IGCRN_dataloader import make_fix_loader
from networks.IGCRN import IGCRN
import torch, gc
from config.train_config import *

from networks.enhancer import PonderEnhancer

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
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_train.txt')
parser.add_argument('--train_mix_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/generated_data/train/mix')
parser.add_argument('--train_ref_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/generated_data/train/noreverb_ref')
parser.add_argument('--train_mic_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC')

parser.add_argument('--val_wav_scp', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_val.txt')
parser.add_argument('--val_mix_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/generated_data/val/mix')
parser.add_argument('--val_ref_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/generated_data/val/noreverb_ref')
parser.add_argument('--val_mic_dir', type=str,
                    default='/ddnstor/imu_panjiahui/dataset/SHGeneralization/cir_random_num_TVT/multi_channel/Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC')

parser.add_argument('--gpuid', type=int, default=0, help='Using which gpu')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epoch for training')
parser.add_argument('--num_worker', type=int, default=4, help='Num_workers: if on PC, set 0')
parser.add_argument('--lr', type=float, default=1e-3, help='Fine tuning learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--fft_len', type=int, default=512)
parser.add_argument('--channel', type=int, default=8)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--chunk', type=int, default=2)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--resume', default=False)

args = parser.parse_args()

# CUDA for PyTorch
device = 'cpu'

NowTime = time.localtime()

parser.add_argument("--cfg", default="complex_ponder_med_k_9_mostdb_5_warmup")
params_function = parser.parse_args().cfg
params = globals()[params_function]()

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    iter_count = 0
    batch_size = args.batch_size
    num_worker = args.num_worker
    fft_len = args.fft_len
    channel = args.channel
    repeat = args.repeat
    chunk = args.chunk
    sample_rate = args.sample_rate
    train_wav_scp = args.train_wav_scp
    train_mix_dir = args.train_mix_dir
    train_ref_dir = args.train_ref_dir
    val_wav_scp = args.val_wav_scp
    val_mix_dir = args.val_mix_dir
    val_ref_dir = args.val_ref_dir

    resume = args.resume


    def count_parameters(network):
        return sum(p.numel() for p in network.parameters() if p.requires_grad)


    print("##################### Trainning  model ###########################")
    if resume:
        modelpath = 'model_miso_newdata/'
        modelname = modelpath + 'network_epoch17.pth'
        network = PonderEnhancer(params['model'])
        network = torch.nn.DataParallel(network)
        network.load_state_dict(torch.load(modelname))
        network = network.cuda()
    else:
        network = PonderEnhancer(params['model'])
        # network = torch.nn.DataParallel(network)
        print(f'The model has {count_parameters(network):,} trainable parameters')
        network = network.cuda()

        # network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    loss_function = nn.MSELoss()

    writer = SummaryWriter('runs/Fine_tuning_{}/'.format(time.strftime("%Y-%m-%d-%H-%M-%S", NowTime)))

    modelpath = 'model_miso_new_data/'

    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)

    loss_train_epoch = []
    loss_val_epoch = []
    metric_val_epoch = []

    loss_train_sequence = []
    loss_val_sequence = []
    metric_val_sequence = []

    # add:
    min_val_loss = float("inf")
    best_val_metric = -float("inf")

    val_no_impv = 0

    for epoch in range(args.num_epoch):

        train_loader = make_fix_loader(
            wav_scp=train_wav_scp,
            mix_dir=train_mix_dir,
            ref_dir=train_ref_dir,
            batch_size=batch_size,
            repeat=repeat,
            num_workers=num_worker,
            chunk=chunk,
            sample_rate=sample_rate,
        )

        val_loader = make_fix_loader(
            wav_scp=val_wav_scp,
            mix_dir=val_mix_dir,
            ref_dir=val_ref_dir,
            batch_size=batch_size,
            repeat=repeat,
            num_workers=num_worker,
            chunk=chunk,
            sample_rate=sample_rate,
        )

        print("############################ Epoch {} ################################".format(epoch + 1))
        ############# Train ############################################################################################################

        network.train()  # set the network in train mode
        # gc.collect()
        # torch.cuda.empty_cache()

        for idx, egs in tqdm(enumerate(train_loader), total=len(train_loader)):
            # B X C X T
            inputs = egs['input'].cuda()
            target = egs['target'].cuda()

            # inputs = egs['input'].to(device)  # bct
            # target = egs['target'].to(device)  # bct

            outputs, ponder = network(inputs)

            # compute loss
            loss = loss_function(outputs.squeeze(), target[:, 0, :])
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
            # inputs = egs['input'].to(device)
            # target = egs['target'].to(device)
            inputs = egs['input'].cuda()
            target = egs['target'].cuda()
            with torch.no_grad():
                outputs, ponder = network(inputs)

            # compute loss
            loss_val = loss_function(outputs.squeeze(), target[:, 0, :])

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
        # plt.ylim([0.04,0.15])
        for i, curve in enumerate(curves):
            plt.plot(curve, label=labels[i])
        plt.legend()
        f1.savefig(modelpath + 'Network_loss.png')

        writer.add_scalars('Loss', {"Validation": loss_val_epoch[epoch]}, iter_count)
