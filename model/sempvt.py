import torch
import torch.nn as nn


from model.utils import create_pointnet_components, create_mlp_components

__all__ = ['pvt_semseg']


class pvt_semseg(nn.Module):
    blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

    def __init__(self, seg_num_all=13, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 9
        self.num_classes = 13

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels,width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,model='SemPVTConv'
        )
        self.point_features = nn.ModuleList(layers)
        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point, out_channels=[256, 128],
            classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=(concat_channels_point + channels_cloud + channels_cloud),
            out_channels=[512, 0.3, 256, 0.3, self.num_classes],
            classifier=True, dim=2, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):

        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords = inputs[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            inputs, _ = self.point_features[i]((inputs, coords))
            out_features_list.append(inputs)
        a = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)
        b = self.cloud_features(inputs.mean(dim=-1, keepdim=False))
        out_features_list.append(a.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        out_features_list.append(b.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))

        return self.classifier(torch.cat(out_features_list, dim=1))