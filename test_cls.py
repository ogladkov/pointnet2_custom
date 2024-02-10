import argparse

import numpy as np
from tqdm import tqdm
import torch

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.pointnet2_cls_ssg import get_model
from models.pointnet_utils import inplace_relu

def test(model, loader, num_class):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1).float()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def test_one(model, loader, num_class):
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1).float()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        print(pred_choice)


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_path', default='./data/val_custom_tmp', help='dataset dir')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--num_workers', default=4, help='cpu worker number')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='./checkpoints/best_model.pth', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=2, type=int, choices=[2],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test_dataset = ModelNetDataLoader(args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    classifier = get_model(num_class=args.num_category, normal_channel=False)
    classifier.apply(inplace_relu)

    checkpoint = torch.load(args.model)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.cuda()

    with torch.no_grad():
        test_one(classifier.eval(), testDataLoader, num_class=args.num_category)
