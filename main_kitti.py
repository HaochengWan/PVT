from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from kitti_datasets.frustum import FrustumKittiDataset
from model.kitti.frustum.frustum_net import FrustumPVT
from torch.utils.data import DataLoader
from modules.frustum import FrustumPointNetLoss
from kitti_datasets.config import configs
from kitti_meters.utils.common import eval_from_files
from util import IOStream
import numpy as np
import numba
from kitti_meters.frustum import MeterFrustumKitti
import shutil


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')


def train(args, io):
    train_loader = DataLoader(FrustumKittiDataset(split='train', num_points=args.num_points, classes=configs.classes,
                 num_heading_angle_bins=configs.num_heading_angle_bins,class_name_to_size_template_id=configs.class_name_to_size_template_id,
                 from_rgb_detection=False,random_flip=True,random_shift=True,frustum_rotate=True),num_workers=16,
                 batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(FrustumKittiDataset(split='val', num_points=args.num_points, classes=configs.classes,
                 num_heading_angle_bins=configs.num_heading_angle_bins,class_name_to_size_template_id=configs.class_name_to_size_template_id,
                 from_rgb_detection=False,random_flip=True,random_shift=True,frustum_rotate=True),num_workers=16,
                 batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'pvt':
        model = FrustumPVT(num_classes=configs.num_classes, num_heading_angle_bins=configs.num_heading_angle_bins,
                 num_size_templates=configs.num_size_templates, num_points_per_object= configs.num_points_per_object,
                 size_templates=configs.size_templates, extra_feature_channels=1, width_multiplier=1,
                 voxel_resolution_multiplier=1).to(device)
    else:
        raise Exception("Not implemented")

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 10, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = FrustumPointNetLoss(num_heading_angle_bins=configs.num_heading_angle_bins, num_size_templates=configs.num_size_templates,
                 size_templates=configs.size_templates, box_loss_weight=1.0,
                 corners_loss_weight=10.0, heading_residual_loss_weight=20.0, size_residual_loss_weight=20.0)

    eval_metrics = ('acc/iou_3d_class_acc_val', 'acc/iou_3d_acc_val')
    best_metrics = {m: None for m in eval_metrics}
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        model.train()
        for data,targets in train_loader:
            for k,v in data.items():
                data[k] = v.to(device)
            for k,v in targets.items():
                targets[k] = v.to(device)
            opt.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()

        ####################
        # Test
        ####################
        model.eval()
        meters = {}
        for name, metric in [
            ('acc/iou_3d_{}', 'iou_3d'), ('acc/acc_{}', 'accuracy'),
            ('acc/iou_3d_acc_{}', 'iou_3d_accuracy'), ('acc/iou_3d_class_acc_{}', 'iou_3d_class_accuracy')
        ]:
            meters[name.format('val')] = MeterFrustumKitti(metric=metric,num_heading_angle_bins=configs.num_heading_angle_bins,
                                            num_size_templates=configs.num_size_templates,size_templates=configs.size_templates,
                                            class_name_to_class_id={cat: cls for cls, cat in enumerate(configs.classes)})
        for data, targets in test_loader:
            for k, v in data.items():
                data[k] = v.to(device)
            for k, v in targets.items():
                targets[k] = v.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            for meter in meters.values():
                meter.update(outputs, targets)
        for k, meter in meters.items():
            meters[k] = meter.compute()

        for k, meter in meters.items():
            outstr = f'Test %d, loss: %.6f,[{k}] = {meter:2f}'% (epoch,loss)

        io.cprint(outstr)

        best = {m: False for m in eval_metrics}
        for m in eval_metrics:
            if best_metrics[m] is None or best_metrics[m] < meters[m]:
                best_metrics[m], best[m] = meters[m], True
                torch.save(model.state_dict(), 'checkpoints/%s/model.t7' % args.exp_name)
            meters[m + '_best'] = best_metrics[m]



def eval(args, io):
    dataset = FrustumKittiDataset(split='val', num_points=args.num_points, classes=configs.classes,
                                                 num_heading_angle_bins=configs.num_heading_angle_bins,
                                                 class_name_to_size_template_id=configs.class_name_to_size_template_id,
                                                 from_rgb_detection=True, random_flip=True, random_shift=True,
                                                 frustum_rotate=True)
    eval_loader = DataLoader(dataset, num_workers=16,batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = FrustumPVT(num_classes=configs.num_classes, num_heading_angle_bins=configs.num_heading_angle_bins,
                 num_size_templates=configs.num_size_templates, num_points_per_object= configs.num_points_per_object,
                 size_templates=configs.size_templates, extra_feature_channels=1, width_multiplier=1,
                 voxel_resolution_multiplier=1).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    results = dict()
    for test_index in range(configs.eval_num_tests):
        predictions = np.zeros((len(dataset), 8))
        size_templates = configs.size_templates.to(device)
        heading_angle_bin_centers = torch.arange(
            0, 2 * np.pi, 2 * np.pi / configs.num_heading_angle_bins).to(device)
        current_step = 0

        with torch.no_grad():
            for data, targets in eval_loader:
                for k, v in data.items():
                    data[k] = v.to(device)
                outputs = model(data)

                center = outputs['center']  # (B, 3)
                heading_scores = outputs['heading_scores']  # (B, NH)
                heading_residuals = outputs['heading_residuals']  # (B, NH)
                size_scores = outputs['size_scores']  # (B, NS)
                size_residuals = outputs['size_residuals']  # (B, NS, 3)

                batch_size = center.size(0)
                batch_id = torch.arange(batch_size, device=center.device)
                heading_bin_id = torch.argmax(heading_scores, dim=1)
                heading = heading_angle_bin_centers[heading_bin_id] + heading_residuals[
                    batch_id, heading_bin_id]  # (B, )
                size_template_id = torch.argmax(size_scores, dim=1)
                size = size_templates[size_template_id] + size_residuals[batch_id, size_template_id]  # (B, 3)

                center = center.cpu().numpy()
                heading = heading.cpu().numpy()
                size = size.cpu().numpy()
                rotation_angle = targets['rotation_angle'].cpu().numpy()  # (B, )
                rgb_score = targets['rgb_score'].cpu().numpy()  # (B, )

                update_predictions(predictions=predictions, center=center, heading=heading, size=size,
                                   rotation_angle=rotation_angle, rgb_score=rgb_score,
                                   current_step=current_step, batch_size=batch_size)
                current_step += batch_size

        np.save('checkpoints/%s/eval.npy' % args.exp_name, predictions)
        predictions_path = 'checkpoints/%s/best_.predictions_%s' % (args.exp_name, test_index)
        image_ids = write_predictions(predictions_path, ids=dataset.data.ids,
                                      classes=dataset.data.class_names, boxes_2d=dataset.data.boxes_2d,
                                      predictions=predictions, image_id_file_path=configs.eval_image_id_file_path)
        _, current_results = eval_from_files(prediction_folder=predictions_path,
                                             ground_truth_folder=configs.eval_ground_truth_path,
                                             image_ids=image_ids, verbose=True)

        if configs.eval_num_tests == 1:
            return
        else:
            for class_name, v in current_results.items():
                if class_name not in results:
                    results[class_name] = dict()
                for kind, r in v.items():
                    if kind not in results[class_name]:
                        results[class_name][kind] = []
                    results[class_name][kind].append(r)

    for class_name, v in results.items():
        io.cprint(f'{class_name}  AP(Average Precision)')
        for kind, r in v.items():
            r = np.asarray(r)
            m = r.mean(axis=0)
            s = r.std(axis=0)
            u = r.max(axis=0)
            rs = ', '.join(f'{mv:.2f} +/- {sv:.2f} ({uv:.2f})' for mv, sv, uv in zip(m, s, u))
            io.cprint(f'{kind:<4} AP: {rs}')

@numba.jit()
def update_predictions(predictions, center, heading, size, rotation_angle, rgb_score, current_step, batch_size):
    for b in range(batch_size):
        l, w, h = size[b]
        x, y, z = center[b]  # (3)
        r = rotation_angle[b]
        t = heading[b]
        s = rgb_score[b]
        v_cos = np.cos(r)
        v_sin = np.sin(r)
        cx = v_cos * x + v_sin * z  # it should be v_cos * x - v_sin * z, but the rotation angle = -r
        cy = y + h / 2.0
        cz = v_cos * z - v_sin * x  # it should be v_sin * x + v_cos * z, but the rotation angle = -r
        r = r + t
        while r > np.pi:
            r = r - 2 * np.pi
        while r < -np.pi:
            r = r + 2 * np.pi
        predictions[current_step + b] = [h, w, l, cx, cy, cz, r, s]


def write_predictions(prediction_path, ids, classes, boxes_2d, predictions, image_id_file_path=None):
    import pathlib

    # map from idx to list of strings, each string is a line (with \n)
    results = {}
    for i in range(predictions.shape[0]):
        idx = ids[i]
        output_str = ('{} -1 -1 -10 '
                      '{:f} {:f} {:f} {:f} '
                      '{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(classes[i], *boxes_2d[i][:4], *predictions[i]))
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)

    # write txt files
    if os.path.exists(prediction_path):
        shutil.rmtree(prediction_path)
    os.mkdir(prediction_path)
    for k, v in results.items():
        file_path = os.path.join(prediction_path, f'{k:06d}.txt')
        with open(file_path, 'w') as f:
            f.writelines(v)

    if image_id_file_path is not None and os.path.exists(image_id_file_path):
        with open(image_id_file_path, 'r') as f:
            val_ids = f.readlines()
        for idx in val_ids:
            idx = idx.strip()
            file_path = os.path.join(prediction_path, f'{idx}.txt')
            if not os.path.exists(file_path):
                # print(f'warning: {file_path} doesn\'t exist as indicated in {image_id_file_path}')
                pathlib.Path(file_path).touch()
        return image_id_file_path
    else:
        image_ids = sorted([k for k in results.keys()])
        return image_ids



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='detection', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pvt', metavar='N',
                        choices=['pvt'],
                        help='Model to use, [pvt]')
    parser.add_argument('--dataset', type=str, default='kitti', metavar='N',
                        choices=['kitti'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=209, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='checkpoints/detection/model.t7', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        eval(args, io)