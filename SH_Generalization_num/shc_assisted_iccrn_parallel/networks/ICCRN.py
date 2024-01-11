import torch.nn as nn
import torch.fft
import torch
from einops import rearrange


class CFB(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(CFB, self).__init__()
        self.conv_gate = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1,
                                   padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1,
                                    padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1,
                              padding=(1, 0), dilation=1, groups=1, bias=True)
        self.ceps_unit = CepsUnit(ch=out_channels)
        self.LN0 = LayerNorm(in_channels, f=160)
        self.LN1 = LayerNorm(out_channels, f=160)
        self.LN2 = LayerNorm(out_channels, f=160)

    def forward(self, x):
        g = torch.sigmoid(self.conv_gate(self.LN0(x)))
        x = self.conv_input(x)
        y = self.conv(self.LN1(g * x))
        y = y + self.ceps_unit(self.LN2((1 - g) * x))
        return y


class CepsUnit(nn.Module):
    def __init__(self, ch):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f = CH_LSTM_F(ch * 2, ch, ch * 2)
        self.LN = LayerNorm(ch * 2, f=81)

    def forward(self, x0):
        x0 = torch.fft.rfft(x0, 160, 2)
        x = torch.cat([x0.real, x0.imag], 1)
        x = self.ch_lstm_f(self.LN(x))
        x = x[:, :self.ch] + 1j * x[:, self.ch:]
        x = x * x0
        x = torch.fft.irfft(x, 160, 2)
        return x


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(1, c, f, 1))
        self.b = nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)

    def forward(self, x):
        mean = x.mean([1, 2], keepdim=True)
        std = x.std([1, 2], keepdim=True)
        x = (x - mean) / (std + 1e-8) * self.w + self.b
        return x


class NET(nn.Module):
    def __init__(self, channels=20, in_shc=25 * 2, in_stft=1 * 2, out_ch=1 * 2):
        super().__init__()
        self.act = nn.ELU()
        self.n_fft = 319
        self.hop_length = 160
        self.window = torch.hamming_window(self.n_fft)

        self.in_ch_lstm_shc = CH_LSTM_F(in_shc, channels, channels)
        self.in_conv_shc = nn.Conv2d(in_channels=in_shc + channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1_shc = CFB(channels, channels)
        self.cfb_e2_shc = CFB(channels, channels)
        self.cfb_e3_shc = CFB(channels, channels)
        self.cfb_e4_shc = CFB(channels, channels)
        self.cfb_e5_shc = CFB(channels, channels)

        self.in_ch_lstm_stft = CH_LSTM_F(in_stft, channels, channels)
        self.in_conv_stft = nn.Conv2d(in_channels=in_stft + channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1_stft = CFB(channels, channels)
        self.cfb_e2_stft = CFB(channels, channels)
        self.cfb_e3_stft = CFB(channels, channels)
        self.cfb_e4_stft = CFB(channels, channels)
        self.cfb_e5_stft = CFB(channels, channels)

        self.ln = LayerNorm(channels * 2, 160)
        self.ch_lstm = CH_LSTM_T(in_ch=channels * 2, feat_ch=channels * 2, out_ch=channels * 2, num_layers=2)

        self.cfb_d5 = CFB(1 * channels * 2, channels * 2)
        self.cfb_d4 = CFB(2 * channels * 2, channels * 2)
        self.cfb_d3 = CFB(2 * channels * 2, channels * 2)
        self.cfb_d2 = CFB(2 * channels * 2, channels * 2)
        self.cfb_d1 = CFB(2 * channels * 2, channels * 2)

        self.out_ch_lstm = CH_LSTM_T(2 * channels * 2, channels, channels * 2 * 2)
        self.out_conv = nn.Conv2d(in_channels=channels * 3 * 2, out_channels=out_ch, kernel_size=(1, 1), padding=(0, 0),
                                  bias=True)

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
        m_out = int(c // 2)
        Y_r = Y[:, :m_out]
        Y_i = Y[:, m_out:]
        Y = torch.stack([Y_r, Y_i], dim=-1)
        Y = Y.reshape(-1, F, T, 2)
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, length=t, window=self.window.to(Y.device))
        y = y.reshape(b, m_out, y.shape[-1])
        return y

    def forward(self, x_shc, x_stft):
        # x:[batch, channel, frequency, time]
        X0_shc = self.stft(x_shc)
        e0_shc = self.in_ch_lstm_shc(X0_shc)
        e0_shc = self.in_conv_shc(torch.cat([e0_shc, X0_shc], 1))
        e1_shc = self.cfb_e1_shc(e0_shc)
        e2_shc = self.cfb_e2_shc(e1_shc)
        e3_shc = self.cfb_e3_shc(e2_shc)
        e4_shc = self.cfb_e4_shc(e3_shc)
        e5_shc = self.cfb_e5_shc(e4_shc)

        X0_stft = self.stft(x_stft)
        e0_stft = self.in_ch_lstm_stft(X0_stft)
        e0_stft = self.in_conv_stft(torch.cat([e0_stft, X0_stft], 1))
        e1_stft = self.cfb_e1_stft(e0_stft)
        e2_stft = self.cfb_e2_stft(e1_stft)
        e3_stft = self.cfb_e3_stft(e2_stft)
        e4_stft = self.cfb_e4_stft(e3_stft)
        e5_stft = self.cfb_e5_stft(e4_stft)

        e0_merge = torch.cat((e0_shc, e0_stft), dim=1)
        e1_merge = torch.cat((e1_shc, e1_stft), dim=1)
        e2_merge = torch.cat((e2_shc, e2_stft), dim=1)
        e3_merge = torch.cat((e3_shc, e3_stft), dim=1)
        e4_merge = torch.cat((e4_shc, e4_stft), dim=1)
        e5_merge = torch.cat((e5_shc, e5_stft), dim=1)

        lstm_out = self.ch_lstm(self.ln(e5_merge))

        d5 = self.cfb_d5(torch.cat([e5_merge * lstm_out], dim=1))
        d4 = self.cfb_d4(torch.cat([e4_merge, d5], dim=1))
        d3 = self.cfb_d3(torch.cat([e3_merge, d4], dim=1))
        d2 = self.cfb_d2(torch.cat([e2_merge, d3], dim=1))
        d1 = self.cfb_d1(torch.cat([e1_merge, d2], dim=1))

        d0 = self.out_ch_lstm(torch.cat([e0_merge, d1], dim=1))
        Y = self.out_conv(torch.cat([d0, d1], dim=1))

        y = self.istft(Y, t=x_shc.shape[-1])

        return y


class CH_LSTM_T(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if bi == False else 2
        self.linear = nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b, c, f, t = x.shape
        x = rearrange(x, 'b c f t -> (b f) t c')
        x, _ = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b f) t c -> b c f t', b=b, f=f, t=t)
        return x


class CH_LSTM_F(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.linear = nn.Linear(2 * feat_ch, out_ch)
        self.out_ch = out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b, c, f, t = x.shape
        x = rearrange(x, 'b c f t -> (b t) f c')
        x, _ = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b t) f c -> b c f t', b=b, f=f, t=t)
        return x


def complexity():
    from thop import profile, clever_format
    from torch.autograd import Variable

    input_1 = Variable(torch.FloatTensor(torch.rand(1, 25, 16000)))
    input_2 = Variable(torch.FloatTensor(torch.rand(1, 1, 16000)))
    net = NET()
    macs, params = profile(net, inputs=(input_1, input_2,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)


if __name__ == '__main__':
    complexity()
# 4.919G
# 474.042K
