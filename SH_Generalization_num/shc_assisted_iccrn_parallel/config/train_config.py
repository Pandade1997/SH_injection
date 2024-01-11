""" 
This is where you specify model params and which dataset to use. You will call these configs from train.py.
"""


def complex_ponder_med_k_9_mostdb_5_warmup():
    hyp = {
        'model': {
            'stft_win_size': 512,
            'stft_hop': 128,
            'hidden_size': 256,
            'n_heads': 4,
            'dropout': 0.25,
            'complex_mask': True,
            'two_d_convs': False,
            'fixed_k': 9,

            'encoder_params': {
                'n_layers': 2,
                'in_channels': [512 // 2 + 1] * 2,
                'out_channels': [512 // 2 + 1] * 2,
                'kernel_size': [5] * 2,
                'stride': [1] * 2,
                'padding': [2] * 2,
                'dilation': [1] * 2,
            },

            'decoder_params': {
                'n_layers': 4,
                'in_channels': [512 // 2 + 1] * 4,
                'out_channels': [512 // 2 + 1] * 4,
                'kernel_size': [5] * 4,
                'stride': [1] * 4,
                'padding': [2] * 4,
                'dilation': [1] * 4,
            }
        }
    }

    return hyp
