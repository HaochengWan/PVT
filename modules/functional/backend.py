import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_pvt_backend',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'interpolate/neighbor_interpolate.cpp',
                    'interpolate/neighbor_interpolate.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'sampling/sampling.cpp',
                    'sampling/sampling.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']
