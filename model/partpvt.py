import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np



from model.utils import create_pointnet_components, create_mlp_components

__all__ = ['pvt_partseg']

class STNbox(nn.Module):
    def __init__(self, k=6):
        super(STNbox, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class pvt_partseg(nn.Module):
    blocks = ((64, 1, 30), (128, 2, 15), (512, 1, None), (1024, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,model='PartPVTConv'
        )
        self.point_features = nn.ModuleList(layers)

        layers, _ = create_mlp_components(
            in_channels=(num_shapes + channels_point + concat_channels_point + channels_point),
            out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
            classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.stn = STNbox()

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        trans = self.stn(features)
        features = features.transpose(2, 1)
        features = torch.bmm(features, trans)
        features = features.transpose(2, 1)
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)
        b = features.size(0)

        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]

        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        out_features_list.append(features.mean(dim=-1, keepdim=True).view(b, -1).unsqueeze(-1).repeat(1, 1, num_points))

        return self.classifier(torch.cat(out_features_list, dim=1))