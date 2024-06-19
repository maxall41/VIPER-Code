import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv


# class GELU(nn.Module):
#    def forward(self, input):
#        return F.gelu(input)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.gelu = nn.Mish(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)


        #
#        np.save("")
        #        csvwriter.writerows(temp.tolist())
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch
from functools import partial

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1=False, strides=1, dropout=0.3):
        super().__init__()

        self.process = nn.Sequential (
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        if use_conv1:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv1 = None

    def forward(self, x):
        left = self.process(x)
        right = x if self.conv1 is None else self.conv1(x)

        return F.mish(left + right)


class cnnModule(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=128, dropout=0.2):
        super().__init__()

        self.head = nn.Sequential (
            nn.Conv1d(in_channel, hidden_channel, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2)
        )

        self.cnn = nn.Sequential (
            resBlock(hidden_channel, out_channel, use_conv1=True, strides=1, dropout=dropout),
            resBlock(out_channel, out_channel, strides=1, dropout=dropout),
            resBlock(out_channel, out_channel, strides=1, dropout=dropout),
        )

        self.bypass_1 = nn.Sequential (
             nn.Conv1d(in_channel, hidden_channel, 7, stride=1, padding=3,dilation=6),
             nn.BatchNorm1d(hidden_channel),
             nn.Mish(inplace=True),
             nn.MaxPool1d(2),
             nn.Conv1d(hidden_channel, out_channel, kernel_size=3, padding=1),
             nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        b1 = self.bypass_1(x)
        x = self.head(x)
        x = self.cnn(x)
        x = torch.cat((b1,x),2)
        return x


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """
            Applies global max pooling over timesteps dimension
        """

        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out

class PredictionHead(nn.Module):
    def __init__(self, dropout=0.2,cls=False):
        super().__init__()
        self.fc = nn.Sequential (
            nn.Linear(5012, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class DeepLPI(nn.Module):
    def __init__(self, molshape, seqshape, dropout=0.3):
        super().__init__()

        self.molshape = molshape
        self.seqshape = seqshape

        self.molcnn = cnnModule(1,64,dropout=dropout)
        self.seqcnn = cnnModule(768,64,dropout=dropout)

        self.lstm = nn.LSTM(32, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.pocket_pool = GlobalMaxPooling1D()

        self.smi_attention_poc = EncoderLayer(64, 64, 0.1, 0.1, 2)

        # (32x192896 and 96512x5012)

        # (32x71936 and 192896x5012)

        #(35968x32 and 71936x5012)

        inb = 71936

        self.mlp = nn.Sequential (
            nn.Linear(inb, 5012),
            nn.BatchNorm1d(5012),
            nn.Mish(),
        )

        # self.mlp = nn.Sequential (
        #     nn.Linear(inb, 10024),
        #     nn.BatchNorm1d(10024),
        #     nn.Mish(),

        #     nn.Linear(10024, 8012),
        #     nn.BatchNorm1d(5012),
        #     nn.Mish(),

        #     nn.Linear(8012, 5012),
        #     nn.BatchNorm1d(5012),
        #     nn.Mish(),

        #     nn.Linear(5012, 5012),
        #     nn.BatchNorm1d(5012),
        #     nn.Mish(),
        # )

        self.ki_head = PredictionHead()
        self.kd_head = PredictionHead()
        self.ic50_head = PredictionHead()
        # self.enzymatic_head = PredictionHead(cls=True)
        self.enzymatic_head = PredictionHead()

    def forward(self, ankh, ecfp):
        BATCH_SIZE = ankh.shape[0]

        molv = ecfp.reshape(-1,1,768)
        ankh = ankh.transpose(2,1)

        mol = self.molcnn(molv)

        ankh = self.seqcnn(ankh)

        mol = mol.reshape((BATCH_SIZE, -1, 64))
        ankh = ankh.reshape((BATCH_SIZE, -1, 64))

        mol_attention = mol
        mol = self.smi_attention_poc(mol, ankh)
        ankh = self.smi_attention_poc(ankh, mol_attention)

        ankh = self.pocket_pool(ankh)

        mol = mol.flatten(1)
        ankh = ankh.flatten(1)

        x = torch.cat((mol, ankh),1)
        x = x.reshape((BATCH_SIZE,-1,32))
        x, _ = self.lstm(x)
        x = x.flatten(1)

        x = self.mlp(x)

        enzyme_activity = self.enzymatic_head(x).flatten()
        enzyme_activity = F.sigmoid(enzyme_activity)
        return enzyme_activity

def create_models():
    models = []
    ensemble_size = 3
    for i in range(ensemble_size):
        model = DeepLPI(1024,768,dropout=0.2)
        model.load_state_dict(torch.load(f"ensemble_model_{i}_perf_0.1284844132615304_validation_set_3_prod.pytorch"))
        model.eval()
        model.cuda()
        models.append(model)
    return models