#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : GCRN_phasenOut_mask_with_mapping_stride1_2_5kernal.py
# Author            : Loiu94 <jetliu1994@foxmail.com>
# Date              : 13.08.2020
# Last Modified Date: 17.02.2021
# Last Modified By  : Loiu94 <jetliu1994@foxmail.com>


import torch.nn as nn
import torch.nn.functional as F
import torch
import sys, os

sys.path.append(os.path.dirname(__file__))


class GLSTM(nn.Module):
    def __init__(self, in_features=None, out_features=None, mid_features=None, hidden_size=1024, groups=2):
        super(GLSTM, self).__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList(
            [nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList(
            [nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups
        self.mid_features = mid_features

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=(0, 0)):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GCRN(nn.Module):
    def __init__(self, in_ch=6):
        super(GCRN, self).__init__()
        self.n_fft = 320
        self.hop_length = 160
        self.window = torch.hamming_window(self.n_fft)

        self.conv1 = GluConv2d(in_channels=in_ch * 2, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.conv2 = GluConv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.conv3 = GluConv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv4 = GluConv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.conv5 = GluConv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))

        self.glstm = GLSTM(groups=2)

        self.conv5_t_1 = GluConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.conv4_t_1 = GluConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv3_t_1 = GluConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.conv2_t_1 = GluConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.conv1_t_1 = GluConvTranspose2d(in_channels=32, out_channels=in_ch, kernel_size=(1, 3), stride=(1, 2))

        self.conv5_t_2 = GluConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.conv4_t_2 = GluConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv3_t_2 = GluConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.conv2_t_2 = GluConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.conv1_t_2 = GluConvTranspose2d(in_channels=32, out_channels=in_ch, kernel_size=(1, 3), stride=(1, 2))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.bn5_t_1 = nn.BatchNorm2d(128)
        self.bn4_t_1 = nn.BatchNorm2d(64)
        self.bn3_t_1 = nn.BatchNorm2d(32)
        self.bn2_t_1 = nn.BatchNorm2d(16)
        self.bn1_t_1 = nn.BatchNorm2d(in_ch)

        self.bn5_t_2 = nn.BatchNorm2d(128)
        self.bn4_t_2 = nn.BatchNorm2d(64)
        self.bn3_t_2 = nn.BatchNorm2d(32)
        self.bn2_t_2 = nn.BatchNorm2d(16)
        self.bn1_t_2 = nn.BatchNorm2d(in_ch)

        self.elu = nn.ELU(inplace=True)

        self.fc1 = nn.Linear(in_features=161, out_features=161)
        self.fc2 = nn.Linear(in_features=161, out_features=161)

    def stft(self, x):
        b, m, t = x.shape[0], x.shape[1], x.shape[2],
        x = x.reshape(-1, t)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device))
        F, T = X.shape[1], X.shape[2]
        X = X.reshape(b, m, F, T, 2)
        X = torch.cat([X[..., 0], X[..., 1]], dim=1)
        return X

    def istft(self, Y, t):
        b, c, F, T = Y.shape
        m_out = int(Y.shape[1] // 2)
        Y_r = Y[:, :m_out]
        Y_i = Y[:, m_out:]
        Y = torch.stack([Y_r, Y_i], dim=-1)
        Y = Y.reshape(-1, F, T, 2)
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, length=t, window=self.window.to(Y.device))
        y = y.reshape(b, m_out, y.shape[-1])
        return y

    def forward(self, x):
        # x:[batch, channel, frequency, time]
        X0 = self.stft(x)

        X0 = X0.permute(0, 1, 3, 2)

        e1 = self.elu(self.bn1(self.conv1(X0)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))
        e5 = self.elu(self.bn5(self.conv5(e4)))

        out = e5

        out = self.glstm(out)

        out = torch.cat((out, e5), dim=1)

        d5_1 = self.elu(torch.cat((self.bn5_t_1(self.conv5_t_1(out)), e4), dim=1))
        d4_1 = self.elu(torch.cat((self.bn4_t_1(self.conv4_t_1(d5_1)), e3), dim=1))
        d3_1 = self.elu(torch.cat((self.bn3_t_1(self.conv3_t_1(d4_1)), e2), dim=1))
        d2_1 = self.elu(torch.cat((self.bn2_t_1(self.conv2_t_1(d3_1)), e1), dim=1))
        d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))

        d5_2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(out)), e4), dim=1))
        d4_2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(d5_2)), e3), dim=1))
        d3_2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(d4_2)), e2), dim=1))
        d2_2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(d3_2)), e1), dim=1))
        d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))

        out1 = self.fc1(d1_1)
        out2 = self.fc2(d1_2)
        Y = torch.cat([out1, out2], dim=1)

        Y = Y.permute(0, 1, 3, 2)

        y = self.istft(Y, t=x.shape[-1])

        return y


def complexity():
    from ptflops import get_model_complexity_info
    model = GCRN()
    mac, param = get_model_complexity_info(model, (2, 16000), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(mac, param)


if __name__ == '__main__':
    complexity()
