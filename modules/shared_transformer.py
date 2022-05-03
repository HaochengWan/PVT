import torch.nn as nn
import torch.nn.functional as F
import torch


__all__ = ['SharedTransformer']


class SharedTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
       
        self.conv1 = nn.Conv1d(in_channels,out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.sa1 = SA_Layer(out_channels)    
        self.conv_fuse = nn.Sequential(nn.Conv1d(out_channels*2, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2)
                                       )
        

    def forward(self, inputs, rel_pos):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x1 = self.sa1(x, rel_pos)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_fuse(x)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.q_conv.bias = self.k_conv.bias 

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.attn_mlp = nn.Sequential(
        nn.Linear(3, channels),
        nn.ReLU(),
        )
    def forward(self, x, rel_pos):
        n = x.shape[2]
        pos = torch.randn(1, n, 3).cuda(0)
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy + rel_pos)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

