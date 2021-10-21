# PVT: Point-Voxel Transformer for 3D Deep Learning
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-voxel-transformer-an-efficient-approach/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=point-voxel-transformer-an-efficient-approach) 
## Paper and Citation
The paper can be downloaded from [arXiv](https://arxiv.org/abs/2108.06076).<BR/>
If you like our work and think it helpful to your project, please cite it as follows.

```citation
@article{zhang2021point,
  title={PVT: Point-Voxel Transformer for 3D Deep Learning},
  author={Zhang, Cheng and Wan, Haocheng and Liu, Shengqiang and Shen, Xinyi and Wu, Zizhao},
  journal={arXiv preprint arXiv:2108.06076},
  year={2021}
}
```

## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.3
- [numba](https://github.com/numba/numba)
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [six](https://github.com/benjaminp/six)
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [plyfile](https://github.com/dranjan/python-plyfile)
- [h5py](https://github.com/h5py/h5py)
- [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm)

## Data Preparation

### ModelNet40

Download alignment ModelNet40 [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and save in `data/modelnet40_normal_resampled/`. 

### S3DIS

We follow the data pre-processing in [PointCNN](https://github.com/yangyanli/PointCNN).
The code for preprocessing the S3DIS dataset is located in [`data/s3dis/`](data/s3dis/prepare_data.py).
One should first download the dataset from [here](http://buildingparser.stanford.edu/dataset.html), then run 

```bash
python data/s3dis/prepare_data.py
```

### ShapeNet

We follow the data pre-processing in [PointNet2](https://github.com/charlesq34/pointnet2). Please download the dataset from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. 

### KITTI

For Frustum-PointNet backbone, we follow the data pre-processing in [Frustum-Pointnets](https://github.com/charlesq34/frustum-pointnets).
One should first download the ground truth labels from [here](http://www.cvlibs.net/download.php?file=data_object_label_2.zip), then run
```bash
unzip data_object_label_2.zip
mv training/label_2 data/kitti/ground_truth
./data/kitti/frustum/download.sh
```

## Pretrained Models

Here we provide a pretrained model on ModelNet40. The accuracy might vary a little bit compared to the paper, since we re-train some of the models for reproducibility.
The path of the model is in `./checkpoints/cls/model.t7`

## Example training and testing


```
#train
python main_cls.py --exp_name=cls --num_points=1024 --use_sgd=True --batch_size 32 --epochs 200 --lr 0.001

#test
python main_cls.py --exp_name=cls --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/cls/model.t7 --test_batch_size 32

```
