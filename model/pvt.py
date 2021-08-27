import torch
import torch.nn as nn
from torch.nn import functional as F

from model.utils import create_pointnet_components

__all__ = ['pvt']


class pvt(nn.Module):
    blocks = ((64, 1, 30), (128, 2, 15), (512, 1, None), (1024, 1, None))

    def __init__(self, num_classes=40, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 6

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,model='PVTConv'
        )
        self.point_features = nn.ModuleList(layers)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(channels_point + concat_channels_point + channels_point, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024))

        self.linear1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, features):
        # inputs : [B, 6, N]
        num_points, batch_size = features.size(-1), features.size(0)

        coords = features[:,:3,:]
        out_features_list = []
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)

        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        out_features_list.append(
            features.mean(dim=-1, keepdim=True).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_points))

        features = torch.cat(out_features_list, dim=1)
        features = F.leaky_relu(self.conv_fuse(features))
        features = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        features = F.leaky_relu(self.bn1(self.linear1(features)))
        features = self.dp1(features)
        features = F.leaky_relu(self.bn2(self.linear2(features)))
        features = self.dp2(features)
        return self.linear3(features)
