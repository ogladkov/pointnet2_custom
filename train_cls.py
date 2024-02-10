import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm

import data_utils.augment as aug
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.pointnet2_cls_ssg import get_model, get_loss
from models.pointnet_utils import inplace_relu
from test_cls import test


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_path', default='./data/jewels_dataset_with_faces', help='dataset dir')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--num_workers', default=4, help='cpu worker number')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
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
    train_dataset = ModelNetDataLoader(args=args, split='train')
    test_dataset = ModelNetDataLoader(args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    classifier = get_model(num_class=args.num_category, normal_channel=args.use_normals)
    criterion = get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    start_epoch = 0

    '''TRANING'''
    def log_string(str):
        logger.info(str)
        print(str)

    logger = logging.getLogger("Model")
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = aug.random_point_dropout(points, max_dropout_ratio=0.3)
            points[:, :, 0:3] = aug.random_scale_point_cloud(
                points[:, :, 0:3],
                scale_low=0.9,
                scale_high=1.1
            )
            points[:, :, 0:3] = aug.shift_point_cloud(points[:, :, 0:3], shift_range=0.05)
            points[:, :, 0:3] = aug.rotate_point_cloud_z(points[:, :, 0:3])
            points[:, :, 0:3] = aug.jitter_point_cloud(points[:, :, 0:3], sigma=0.01, clip=0.05)
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(
                classifier.eval(),
                testDataLoader,
                num_class=args.num_category
            )

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str('checkpoints') + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            global_epoch += 1

    logger.info('End of training...')