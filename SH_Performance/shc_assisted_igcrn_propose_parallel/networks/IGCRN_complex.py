import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch
import sys, os

from ptflops import get_model_complexity_info
from thop import profile
from torch.nn.modules.activation import MultiheadAttention

sys.path.append(os.path.dirname(__file__))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, q, k, v):
        q = q.permute(1, 0, 2).contiguous()
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()
        atten_out, att_mask = self.self_attn(query=q, key=k, value=v, attn_mask=None, key_padding_mask=None)
        atten_out = atten_out.permute(1, 0, 2).contiguous()

        return atten_out, att_mask


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
    def __init__(self, in_ch_sph=25, in_ch_stft=9, out_ch=9, channels=32, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)
        self.act = nn.ELU()

        self.e1_sph = convGLU(in_ch_sph * 2, channels)
        self.e2_sph = convGLU(channels, channels, dilation=(2, 1), padding=(4, 0))
        self.e3_sph = convGLU(channels, channels, dilation=(4, 1), padding=(8, 0))
        self.e4_sph = convGLU(channels, channels, dilation=(8, 1), padding=(16, 0))
        self.e5_sph = convGLU(channels, channels, dilation=(16, 1), padding=(32, 0))
        self.e6_sph = convGLU(channels, channels, dilation=(32, 1), padding=(64, 0))

        self.e1_stft = convGLU(in_ch_stft * 2, channels)
        self.e2_stft = convGLU(channels, channels, dilation=(2, 1), padding=(4, 0))
        self.e3_stft = convGLU(channels, channels, dilation=(4, 1), padding=(8, 0))
        self.e4_stft = convGLU(channels, channels, dilation=(8, 1), padding=(16, 0))
        self.e5_stft = convGLU(channels, channels, dilation=(16, 1), padding=(32, 0))
        self.e6_stft = convGLU(channels, channels, dilation=(32, 1), padding=(64, 0))

        self.BNe6 = LayerNorm(channels * 2, f=257)
        self.ch_lstm = ch_lstm(in_ch=channels * 2, feat_ch=channels * 4, out_ch=channels * 2)

        self.d16 = convGLU(1 * channels * 2, channels * 2, dilation=(32, 1), padding=(64, 0))
        self.d15 = convGLU(2 * channels * 2, channels * 2, dilation=(16, 1), padding=(32, 0))
        self.d14 = convGLU(2 * channels * 2, channels * 2, dilation=(8, 1), padding=(16, 0))
        self.d13 = convGLU(2 * channels * 2, channels * 2, dilation=(4, 1), padding=(8, 0))
        self.d12 = convGLU(2 * channels * 2, channels * 2, dilation=(2, 1), padding=(4, 0))
        self.d11 = convGLU(2 * channels * 2, channels * 2, dilation=(1, 1), padding=(2, 0))

        self.convOUT = convGLU(channels * 2, out_ch * 2)
        self.d_model = 257
        self.nhead = 1
        self.dropout = 0.1
        self.atten_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dropout=self.dropout)

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

    def forward(self, sph_input, mix_input):
        # x: b m t
        # def forward(self, sph_input, mix_input):
        # sph_single = self.stft(sph_input)
        # mix_single = self.stft(mix_input)

        sph_single = sph_input
        mix_single = mix_input
        e1_sph = self.e1_sph(torch.cat([sph_single], dim=1))
        e2_sph = self.e2_sph(e1_sph)
        e3_sph = self.e3_sph(e2_sph)
        e4_sph = self.e4_sph(e3_sph)
        e5_sph = self.e5_sph(e4_sph)
        e6_sph = self.e6_sph(e5_sph)

        e1_stft = self.e1_stft(torch.cat([mix_single], dim=1))
        e2_stft = self.e2_stft(e1_stft)
        e3_stft = self.e3_stft(e2_stft)
        e4_stft = self.e4_stft(e3_stft)
        e5_stft = self.e5_stft(e4_stft)
        e6_stft = self.e6_stft(e5_stft)

        e1_merge = torch.cat((e1_sph, e1_stft), dim=1)
        e2_merge = torch.cat((e2_sph, e2_stft), dim=1)
        e3_merge = torch.cat((e3_sph, e3_stft), dim=1)
        e4_merge = torch.cat((e4_sph, e4_stft), dim=1)
        e5_merge = torch.cat((e5_sph, e5_stft), dim=1)
        e6_merge = torch.cat((e6_sph, e6_stft), dim=1)

        lstm_out = self.ch_lstm(self.BNe6(e1_merge))

        d16 = self.d16(torch.cat([e6_merge * lstm_out], dim=1))
        d15 = self.d15(torch.cat([e5_merge, d16], dim=1))
        d14 = self.d14(torch.cat([e4_merge, d15], dim=1))
        d13 = self.d13(torch.cat([e3_merge, d14], dim=1))
        d12 = self.d12(torch.cat([e2_merge, d13], dim=1))
        d11 = self.d11(torch.cat([e1_merge, d12], dim=1))
        Y = self.convOUT(d11)
        #  return {'wav':y, 'hc_dict': hc_dict}
        # y = self.istft(Y, t=wav_len)
        return Y


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
    input_2 = Variable(torch.FloatTensor(torch.rand(1, 18, 257, 63)))
    net = IGCRN(25)
    macs, params = profile(net, inputs=(input_1, input_2,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)

# 19.519G
# 1.824M