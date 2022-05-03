import torch
import torch.nn as nn
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_transformer import SharedTransformer
from modules.se import SE3d
from timm.models.layers import DropPath
import numpy as np

__all__ = ['PVTConv','PartPVTConv','SemPVTConv']

def rand_bbox(size, lam):
    x = size[2]
    y = size[3]
    z = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_x = np.int(x * cut_rat)
    cut_y = np.int(y * cut_rat)
    cut_z = np.int(y * cut_rat)

    cx = np.random.randint(x)
    cy = np.random.randint(y)
    cz = np.random.randint(z)

    bbx1 = np.clip(cx - cut_x // 2, 0, x)
    bbx2 = np.clip(cx + cut_x // 2, 0, x)
    bby1 = np.clip(cy - cut_y // 2, 0, y)
    bby2 = np.clip(cy + cut_y // 2, 0, y)
    bbz1 = np.clip(cz - cut_z // 2, 0, z)
    bbz2 = np.clip(cz + cut_z // 2, 0, z)

    return bbx1, bby1, bbx2, bby2, bbz1, bbz2

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def box_partition(x, box_size):
    b = x.shape[0]
    resolution = x.shape[1]
    out_channels = x.shape[-1]
    x = torch.reshape(x, (
        b, resolution // box_size, box_size, resolution // box_size,
        box_size,
        resolution // box_size, box_size, out_channels))
    boxs = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, box_size, box_size, box_size,
                                                                  out_channels)

    return boxs

def box_reverse(boxs, box_size, resolution):
    b = int(boxs.shape[0] / (resolution **3 / box_size / box_size / box_size))
    x = torch.reshape(boxs, (
        b, resolution // box_size, resolution // box_size,
        resolution // box_size,
        box_size, box_size, box_size, -1))
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, resolution, resolution, resolution, -1)
    return x

class BoxAttention(nn.Module):
    def __init__(self, dim, box_size, num_heads, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.box_size = box_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((3 * box_size - 1) **2, num_heads))

        coords_x = torch.arange(self.box_size)
        coords_y = torch.arange(self.box_size)
        coords_z = torch.arange(self.box_size)

        coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.box_size - 1
        relative_coords[:, :, 1] += self.box_size - 1
        relative_coords[:, :, 2] += self.box_size - 1
        relative_coords[:, :, 0] *= 3 * self.box_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.box_size **3, self.box_size**3, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, box_size={self.box_size}, num_heads={self.num_heads}'

class Sboxblock(nn.Module):
    def __init__(self, out_channels, resolution, boxsize, mlp_dims, shift=True, drop_path=0., ):
        super().__init__()
        self.out_channels = out_channels
        self.resolution = resolution
        self.heads = 4
        self.dim_head = self.out_channels // self.heads
        self.box_size = boxsize
        if shift is not None:
            self.shift_size = self.box_size // 2
        else:
            self.shift_size = 0

        self.attn = BoxAttention(
            out_channels, box_size=self.box_size, num_heads=self.heads)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, self.resolution, self.resolution, self.resolution, 1))
            slices = (slice(0, -self.box_size),
                      slice(-self.box_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for x in slices:
                for y in slices:
                    for z in slices:
                        img_mask[:, x, y, z, :] = cnt
                        cnt += 1

            mask_boxs = box_partition(img_mask, self.box_size)
            mask_boxs = mask_boxs.view(-1, self.box_size **3)
            attn_mask = mask_boxs.unsqueeze(1) - mask_boxs.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp_dim = mlp_dims
        self.mlp = FeedForward(out_channels, self.mlp_dim)
        self.drop_path = DropPath(drop_path)
    def forward(self, inputs):

        shortcut = inputs
        b = inputs.shape[0]
        x = self.norm1(inputs)
        x = torch.reshape(x, (b, self.resolution, self.resolution, self.resolution, self.out_channels))
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))

        else:
            shifted_x = x
        boxs = box_partition(shifted_x, self.box_size)
        boxs = torch.reshape(boxs, (-1, self.box_size ** 3, self.out_channels))

        attn_boxs = self.attn(boxs, mask=self.attn_mask)

        boxs = torch.reshape(attn_boxs, (-1, self.box_size, self.box_size, self.box_size,
                                               self.out_channels))
        x = box_reverse(boxs, self.box_size, self.resolution)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x
        x = torch.reshape(shifted_x, (b, self.resolution**3, self.out_channels))

        x = self.drop_path(x)*0.5 + shortcut
        x = self.drop_path(self.mlp(self.norm2(x)))*0.5 + x

        return x

class Transformer(nn.Module):
    def __init__(self, out_channels, resolution,boxsize,mlp_dims,drop_path1,drop_path2):
        super().__init__()
        self.shift = None
        self.depth = 2
        self.blocks = nn.ModuleList([
            Sboxblock(out_channels, resolution, boxsize, mlp_dims, shift=None if (i % 2 == 0) else True,
                      drop_path=drop_path1 if (i % 2 == 0) else drop_path2)
            for i in range(self.depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class VoxelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution,boxsize,mlp_dims,drop_path1,drop_path2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.pos_drop = nn.Dropout(p=0.)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))
        self.voxel_Trasformer = Transformer(out_channels, resolution, boxsize,mlp_dims,drop_path1,drop_path2)

    def forward(self, inputs):
        inputs = self.voxel_emb(inputs)
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        x = inputs.permute(0, 2, 1)
        x = self.layer_norm(x)
        x += self.pos_embedding  # todo
        x = self.pos_drop(x)
        x = self.voxel_Trasformer(x)
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))
        x = x.permute(0, 4, 1, 2, 3)
        return x

class SegVoxelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution,boxsize,mlp_dims,drop_path1,drop_path2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.pos_drop = nn.Dropout(p=0.)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))
        self.voxel_Trasformer = Transformer(out_channels, resolution, boxsize,mlp_dims,drop_path1,drop_path2)
        self.beta = 1.

    def forward(self, inputs):
        inputs = self.voxel_emb(inputs)
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2, bbz1, bbz2 = rand_bbox(inputs.size(), lam)
        temp_x = inputs.clone()
        temp_x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = inputs.flip(0)[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        inputs = temp_x
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        x = inputs.permute(0, 2, 1)
        x = self.layer_norm(x)
        x += self.pos_embedding  # todo
        x = self.pos_drop(x)
        x = self.voxel_Trasformer(x)
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))
        temp_x = x.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = x.flip(0)[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :]
        x = temp_x
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PVTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.2
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution,self.boxsize,
                                          self.mlp_dims,self.drop_path1,self.drop_path2)
        self.SE = SE3d(out_channels)
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        
        pos = coords.permute(0, 2, 1)
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos = rel_pos.sum(dim=-1)  
        
        fused_features = voxel_features + self.point_features(features, rel_pos)
        return fused_features, coords

class PartPVTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.1
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution,self.boxsize,
                                             self.mlp_dims,self.drop_path1,self.drop_path2)
        self.SE = SE3d(out_channels)
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords

class SemPVTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 4
        self.mlp_dims = out_channels*4
        self.drop_path1 = 0.
        self.drop_path2 = 0.1
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = SegVoxelEncoder(in_channels, out_channels, kernel_size, resolution,self.boxsize,
                                             self.mlp_dims,self.drop_path1,self.drop_path2)
        self.SE = SE3d(out_channels)
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
