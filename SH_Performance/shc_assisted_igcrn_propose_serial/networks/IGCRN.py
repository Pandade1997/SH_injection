import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch
import sys, os

sys.path.append(os.path.dirname(__file__))


class convGLU(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=(5, 1), stride=(1, 1),
                 padding=(2, 0), dilation=1, groups=1):
        super(convGLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, padding_mode='circular', dilation=dilation, groups=groups,
                              bias=False)
        self.convGate = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, padding_mode='circular', dilation=dilation,
                                  groups=groups, bias=True)
        self.gate_act = nn.Sigmoid()
        self.LN = LayerNorm(in_channels, f=257)

    def forward(self, inputs):
        outputs = self.conv(inputs) * self.gate_act(self.convGate(self.LN(inputs)))
        return outputs


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm([c, f])

    def forward(self, x):
        x = self.ln(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class IGCRN(nn.Module):
    def __init__(self, in_ch=9, out_ch=2, channels=64, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)
        self.act = nn.ELU()

        self.e1 = convGLU(in_ch * 2, channels)
        self.e2 = convGLU(channels, channels, dilation=(2, 1), padding=(4, 0))
        self.e3 = convGLU(channels, channels, dilation=(4, 1), padding=(8, 0))
        self.e4 = convGLU(channels, channels, dilation=(8, 1), padding=(16, 0))
        self.e5 = convGLU(channels, channels, dilation=(16, 1), padding=(32, 0))
        self.e6 = convGLU(channels, channels, dilation=(32, 1), padding=(64, 0))

        self.BNe6 = LayerNorm(channels, f=257)
        self.ch_lstm = ch_lstm(in_ch=channels, feat_ch=channels * 2, out_ch=channels)

        self.d16 = convGLU(1 * channels, channels, dilation=(32, 1), padding=(64, 0))
        self.d15 = convGLU(2 * channels, channels, dilation=(16, 1), padding=(32, 0))
        self.d14 = convGLU(2 * channels, channels, dilation=(8, 1), padding=(16, 0))
        self.d13 = convGLU(2 * channels, channels, dilation=(4, 1), padding=(8, 0))
        self.d12 = convGLU(2 * channels, channels, dilation=(2, 1), padding=(4, 0))
        self.d11 = convGLU(2 * channels, channels, dilation=(1, 1), padding=(2, 0))

        self.convOUT = convGLU(channels, in_ch * 2)

    def stft(self, x):
        b, m, t = x.shape[0], x.shape[1], x.shape[2],
        x = x.reshape(-1, t)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device),
                       return_complex=False)
        # print(X.shape)
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
        y = torch.istft(torch.view_as_complex(Y), n_fft=self.n_fft, hop_length=self.hop_length, length=t,
                        window=self.window.to(Y.device))
        y = y.reshape(b, m_out, y.shape[-1])
        return y

    def forward(self, x):
        # x: b m t
        X0 = self.stft(x)
        # X0:[batch, channel, frequency, time]

        # X0 = x

        e1 = self.e1(torch.cat([X0], dim=1))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        lstm_out = self.ch_lstm(self.BNe6(e6))

        d16 = self.d16(torch.cat([e6 * lstm_out], dim=1))
        d15 = self.d15(torch.cat([e5, d16], dim=1))
        d14 = self.d14(torch.cat([e4, d15], dim=1))
        d13 = self.d13(torch.cat([e3, d14], dim=1))
        d12 = self.d12(torch.cat([e2, d13], dim=1))
        d11 = self.d11(torch.cat([e1, d12], dim=1))
        Y = self.convOUT(d11)
        #  return {'wav':y, 'hc_dict': hc_dict}
        y = self.istft(Y, t=x.shape[-1])
        return y


class ch_lstm(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=2, batch_first=True, bidirectional=bi)
        self.bi = 1 if bi == False else 2
        self.linear_lstm_out2 = nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch

    def forward(self, e5):
        shape_in2 = e5.shape
        lstm_in2 = e5.permute(0, 2, 3, 1).reshape(-1, shape_in2[3], shape_in2[1])
        lstm_out2, _ = self.lstm2(lstm_in2.float())
        lstm_out2 = self.linear_lstm_out2(lstm_out2)
        lstm_out2 = lstm_out2.reshape(shape_in2[0], shape_in2[2], shape_in2[3], self.out_ch).permute(0, 3, 1, 2)

        return lstm_out2


if __name__ == '__main__':
    # complexity()
    from thop import profile, clever_format
    from torch.autograd import Variable

    input_1 = Variable(torch.FloatTensor(torch.rand(1, 50, 257, 63)))
    net = IGCRN(in_ch=25)
    macs, params = profile(net, inputs=(input_1,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
