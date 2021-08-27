import functools

import torch.nn as nn

from modules import SharedMLP
from modules.pvtconv import PVTConv,PartPVTConv,SemPVTConv

__all__ = ['create_mlp_components', 'create_pointnet_components']


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1,model=''):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        elif model=='PVTConv':
            block = functools.partial(PVTConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                      normalize=normalize, eps=eps)
        elif model=='PartPVTConv':
            block = functools.partial(PartPVTConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                      normalize=normalize, eps=eps)
        elif model=='SemPVTConv':
            block = functools.partial(SemPVTConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                      normalize=normalize, eps=eps)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels

