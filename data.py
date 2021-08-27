import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids
class ModelNetDataLoader(Dataset):
    def __init__(self, npoint=1024, partition='train', uniform=False, normal_channel=True, cache_size=15000):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')

        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_test.txt'))]

        assert (partition == 'train' or partition == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[partition]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(DATA_DIR, shape_names[i], shape_ids[partition][i]) + '.txt') for i
                         in range(len(shape_ids[partition]))]
        print('The size of %s data is %d'%(partition,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)
def load_data_cls(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class _ShapeNetDataset(Dataset):
    def __init__(self, num_points, partition='trainval', with_normal=True, with_one_hot_shape_id=True,
                 normalize=True, jitter=True):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(BASE_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
        self.num_points = num_points
        self.split = partition
        self.with_normal = with_normal
        self.with_one_hot_shape_id = with_one_hot_shape_id
        self.normalize = normalize
        if partition == 'trainval':
            self.jitter = jitter
        else:
            self.jitter = False

        shape_dir_to_shape_id = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for shape_id, line in enumerate(f):
                shape_name, shape_dir = line.strip().split()
                shape_dir_to_shape_id[shape_dir] = shape_id
        file_paths = []
        if self.split == 'trainval':
            split = ['train', 'val']
        else:
            split = ['test']
        for s in split:
            with open(os.path.join(self.root, 'train_test_split', f'shuffled_{s}_file_list.json'), 'r') as f:
                file_list = json.load(f)
                for file_path in file_list:
                    _, shape_dir, filename = file_path.split('/')
                    file_paths.append(
                        (os.path.join(self.root, shape_dir, filename + '.txt'),
                         shape_dir_to_shape_id[shape_dir])
                    )
        self.file_paths = file_paths
        self.num_shapes = 16
        self.num_classes = 50

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            coords, normal, label, shape_id = self.cache[index]
        else:
            file_path, shape_id = self.file_paths[index]
            data = np.loadtxt(file_path).astype(np.float32)
            coords = data[:, :3]
            if self.normalize:
                coords = self.normalize_point_cloud(coords)
            normal = data[:, 3:6]
            label = data[:, -1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (coords, normal, label, shape_id)

        choice = np.random.choice(label.shape[0], self.num_points, replace=True)
        coords = coords[choice, :].transpose()
        if self.jitter:
            coords = self.jitter_point_cloud(coords)
        if self.with_normal:
            normal = normal[choice, :].transpose()
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, normal, shape_one_hot])
            else:
                point_set = np.concatenate([coords, normal])
        else:
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, shape_one_hot])
            else:
                point_set = coords

        shape_label = np.array([1])
        shape_label = shape_label + shape_id


        return point_set, label[choice].transpose(), shape_label

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def normalize_point_cloud(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        return points / np.max(np.linalg.norm(points, axis=1))

    @staticmethod
    def jitter_point_cloud(points, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              3xN array, original batch of point clouds
            Return:
              3xN array, jittered batch of point clouds
        """
        assert (clip > 0)
        return np.clip(sigma * np.random.randn(*points.shape), -1 * clip, clip).astype(np.float32) + points

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud



class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area=5, with_normalized_coords=True):
        """
        :param root: directory path to the s3dis dataset
        :param num_points: number of points to process for each scene
        :param partition: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param test_area: which area to holdout (default: 5)
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(BASE_DIR, 'data', 's3dis','pointcnn')
        self.partition = partition
        self.num_points = num_points
        self.test_area = None if test_area is None else int(test_area)
        self.with_normalized_coords = with_normalized_coords
        # keep at most 20/30 files in memory
        self.cache_size = 20 if partition == 'train' else 30
        self.cache = {}

        # mapping batch index to corresponding file
        areas = []
        if self.partition == 'train':
            for a in range(1, 7):
                if a != self.test_area:
                    areas.append(os.path.join(self.root, f'Area_{a}'))
        else:
            areas.append(os.path.join(self.root, f'Area_{self.test_area}'))

        self.num_scene_windows, self.max_num_points = 0, 0
        index_to_filename, scene_list = [], {}
        filename_to_start_index = {}
        for area in areas:
            area_scenes = os.listdir(area)
            area_scenes.sort()
            for scene in area_scenes:
                current_scene = os.path.join(area, scene)
                scene_list[current_scene] = []
                for partition in ['zero', 'half']:
                    current_file = os.path.join(current_scene, f'{partition}_0.h5')
                    filename_to_start_index[current_file] = self.num_scene_windows
                    h5f = h5py.File(current_file, 'r')
                    num_windows = h5f['data'].shape[0]
                    self.num_scene_windows += num_windows
                    for i in range(num_windows):
                        index_to_filename.append(current_file)
                    scene_list[current_scene].append(current_file)
        self.index_to_filename = index_to_filename
        self.filename_to_start_index = filename_to_start_index
        self.scene_list = scene_list

    def __len__(self):
        return self.num_scene_windows

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        if filename not in self.cache.keys():
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data']
            scene_label = h5f['label_seg']
            scene_num_points = h5f['data_num']
            if len(self.cache.keys()) < self.cache_size:
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
            else:
                victim_idx = np.random.randint(0, self.cache_size)
                cache_keys = list(self.cache.keys())
                cache_keys.sort()
                self.cache.pop(cache_keys[victim_idx])
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
        else:
            scene_data, scene_label, scene_num_points = self.cache[filename]

        internal_pos = index - self.filename_to_start_index[filename]
        current_window_data = np.array(scene_data[internal_pos]).astype(np.float32)
        current_window_label = np.array(scene_label[internal_pos]).astype(np.int64)
        current_window_num_points = scene_num_points[internal_pos]

        choices = np.random.choice(current_window_num_points, self.num_points,
                                   replace=(current_window_num_points < self.num_points))
        data = current_window_data[choices, ...].transpose()
        label = current_window_label[choices]
        # data[9, num_points] = [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
        if self.with_normalized_coords:
            return data, label
        else:
            return data[:-3, :], label

