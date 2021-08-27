import numpy as np
import torch
from kitti_datasets.container import G
from kitti_datasets.attributes import kitti_attributes as kitti

__all__ = ['configs']
configs = G()

configs.classes = ('Car', 'Pedestrian', 'Cyclist')
configs.num_classes = len(configs.classes)

configs.num_points_per_object = 512
configs.num_heading_angle_bins = 12
configs.size_template_names = kitti.class_names
configs.num_size_templates = len(configs.size_template_names)
configs.class_name_to_size_template_id = {
    cat: cls for cls, cat in enumerate(configs.size_template_names)
}
configs.size_template_id_to_class_name = {
    v: k for k, v in configs.class_name_to_size_template_id.items()
}
configs.size_template = np.zeros((configs.num_size_templates, 3))
for i in range(configs.num_size_templates):
    configs.size_template[i, :] = kitti.class_name_to_size_template[
        configs.size_template_id_to_class_name[i]]
configs.size_templates = torch.from_numpy(configs.size_template.astype(np.float32))

configs.class_name_to_size_template_id = configs.class_name_to_size_template_id
configs.random_flip = True
configs.random_shift = True
configs.frustum_rotate = True
configs.from_rgb_detection = False
#loss
configs.box_loss_weight = 1.0
configs.corners_loss_weight = 10.0
configs.heading_residual_loss_weight = 20.0
configs.size_residual_loss_weight = 20.0
#eval
configs.eval_num_tests = 20
configs.eval_ground_truth_path = 'data/kitti/ground_truth'
configs.eval_image_id_file_path = 'data/kitti/image_sets/val.txt'


