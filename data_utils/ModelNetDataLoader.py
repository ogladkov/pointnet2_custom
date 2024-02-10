import os
import pickle
import random

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


def read_off(file):

    with open(file, 'r') as f:
        off_header = f.readline().strip()

        if 'OFF' == off_header:
            n_verts, n_faces, __ = tuple([int(s) for s in f.readline().strip().split(' ')])

        else:
            n_verts, n_faces, __ = tuple([int(s) for s in off_header.split(' ')])

        verts = [[float(s) for s in f.readline().strip('(').strip(')').split()] for i_vert in range(n_verts)]
        verts = np.array(verts)

        faces = [[int(s) for s in f.readline().strip('(').strip(')').split()][1:] for i_face in range(n_faces)]
        faces = np.array(faces)

    return verts, faces


def farthest_point_sample(point, npoint):
    N, D = point.shape

    if npoint <= N:
        # If fewer points are needed, sample without replacement
        indices = np.random.choice(N, npoint, replace=False)
        return point[indices]

    else:
        # Generate initial set of points
        interpolated_points = np.zeros((npoint, D))
        interpolated_points[:N, :] = point

        # Start interpolation to generate new points
        for i in range(N, npoint):
            # Randomly select two points to interpolate between
            p1, p2 = point[np.random.choice(N, 2, replace=False), :]
            # Generate a random interpolation factor between 0 and 1
            alpha = np.random.rand()
            # Interpolate new point
            new_point = p1 + alpha * (p2 - p1)
            interpolated_points[i, :] = new_point

        return interpolated_points


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class PointSamplerEven(object):

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, verts, faces):
        verts = np.array(verts)
        total_area = sum(self.triangle_area(verts[face[0]], verts[face[1]], verts[face[2]]) for face in faces)

        sampled_points = []

        for face in faces:
            area = self.triangle_area(verts[face[0]], verts[face[1]], verts[face[2]])
            points_in_face = int(np.round(self.output_size * (area / total_area)))

            for _ in range(points_in_face):
                cl = face[-1]
                sampled_points.append(self.sample_point(verts[face[0]], verts[face[1]], verts[face[2]]))

        # In case rounding errors result in a different number of points, we adjust the number of sampled points
        while len(sampled_points) > self.output_size:
            sampled_points.pop()

        while len(sampled_points) < self.output_size:
            face = random.choice(faces)
            sampled_points.append(self.sample_point(verts[face[0]], verts[face[1]], verts[face[2]]))

        return np.array(sampled_points)

class ModelNetDataLoader(Dataset):

    def __init__(self, args, split='train'):
        self.sampler = PointSamplerEven(1024)
        self.root = args.data_path
        self.npoints = args.num_point
        self.process_data = args.process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.catfile = os.path.join(self.root, 'shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.datapath = []

        for cl in self.cat:
            folder = os.path.join(self.root, cl, split)

            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                self.datapath.append([cl, fpath])

        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.save_path = os.path.join(self.root, '%s_%dpts_fps.dat' % (split, self.npoints))

        # Process data (points sampling)
        if self.process_data:

            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    verts, faces = read_off(fn[1])
                    point_set = self.sampler(verts, faces)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)

        print('Load processed data from %s...' % self.save_path)

        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        # Normalize
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        return point_set, label[0]
