import os
import json
import numpy as np
import h5py
import open3d

from .dataset import Dataset

DB_ROOT = os.environ['DB_ROOT']

class ImageListRelevants(Dataset):
    """ A dataset composed by a list of images, a list of indices used as queries,
        and for each query a list of relevant an    d junk indices (ie. Oxford-like GT format)

        Input: path to the pickle file
    """
    def __init__(self, root=None, img_dir='', ext='.jpg', train=True):
        self.img_dir = img_dir

        if train:
            self.root = root + '/train'
        else:
            self.root = root + '/test'

        sub_dir_list = os.listdir(self.root)

        self.imgs = []
        self.pc_datas = []
        self.camera_paths = []
        self.gt_paths = []
        for sub_dir in sub_dir_list:
            sub_imgs = os.listdir(self.root + '/' + sub_dir + '/images')
            sub_imgs = [sub_dir + '/images/' + a for a in sub_imgs]
            self.imgs.extend(sub_imgs)

            path = self.root + '/' + sub_dir + '/camera_parameters.txt'
            self.camera_paths.append(path)

            if train:
                path = self.root + '/' + sub_dir + '/groundtruth.hdf5'
                self.gt_paths.append(path)

                sub_pc = os.listdir(self.root + '/' + sub_dir + '/pointclouds_data')
                sub_pc = [sub_dir + '/pointclouds_data/' + a for a in sub_pc]
                self.pc_datas.extend(sub_pc)

        self.nimg = len(self.imgs)

    def get_image(self, img_idx):
        from PIL import Image
        img = Image.open(self.get_filename(img_idx)).convert('RGB')
        resize = (int(img.width / 2), int(img.height / 2))
        if resize:
            img = img.resize(resize, Image.ANTIALIAS if np.prod(resize) < np.prod(img.size) else Image.BICUBIC)
        return img

    def get_pointclouds(self, timestamp_list, date):
        pcd_list = []
        for pc_path in self.pc_datas:
            if date in pc_path and 'lidar0' in pc_path:
                for timestamp in timestamp_list:
                    if str(timestamp) in pc_path:
                        pcd_list.append(self.root + '/' + pc_path)

        return pcd_list

    def get_parameters(self, cameraname, date):
        for path in self.camera_paths:
            if date in path:
                with open(path) as f:
                    line = f.readline()
                    while line:
                        parameter_line = line.strip()
                        if cameraname in parameter_line:
                            parameter = parameter_line.split()
                            return parameter

                        line = f.readline()

    def get_groundtruth(self, posename, stampname, timestamp, date):
        timestamp = int(timestamp)
        
        for path in self.gt_paths:
            if date in path:
                with h5py.File(path, "r") as f:
                    group_key = list(f.keys())
                    pose_idx = 0
                    if stampname in group_key:
                        data = list(f[stampname])
                        for i, d in enumerate(data):
                            if d[0] == timestamp:
                                pose_idx = i
                                break
                    
                    if posename in group_key:
                        data = list(f[posename])
                        return data[pose_idx]


    def get_relevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            rel = self.relevants[qimg_idx]
        elif mode == 'easy':
            rel = self.easy[qimg_idx]
        elif mode == 'medium':
            rel = self.easy[qimg_idx] + self.hard[qimg_idx]
        elif mode == 'hard':
            rel = self.hard[qimg_idx]
        return rel

    def get_junk(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            junk = self.junk[qimg_idx]
        elif mode == 'easy':
            junk = self.junk[qimg_idx] + self.hard[qimg_idx]
        elif mode == 'medium':
            junk = self.junk[qimg_idx]
        elif mode == 'hard':
            junk = self.junk[qimg_idx] + self.easy[qimg_idx]
        return junk

    def get_query_filename(self, img_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.imgs[img_idx])

    def get_query_roi(self, qimg_idx):
        return self.qroi[qimg_idx]

    def get_key(self, i):
        return self.imgs[i]

    def get_query_key(self, i):
        return self.qimgs[i]

    def get_query_db(self):
        return ImageListROIs(self.root, self.img_dir, self.qimgs, self.qroi)

    def get_query_groundtruth(self, query_idx, what='AP', mode='classic'):
        # negatives
        res = -np.ones(self.nimg, dtype=np.int8)
        # positive
        res[self.get_relevants(query_idx, mode)] = 1
        # junk
        res[self.get_junk(query_idx, mode)] = 0
        return res

    def eval_query_AP(self, query_idx, scores):
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_average_precision
        if self.relevants:
            gt = self.get_query_groundtruth(query_idx, 'AP')  # labels in {-1, 0, 1}
            assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
            assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
            keep = (gt != 0)  # remove null labels

            gt, scores = gt[keep], scores[keep]
            gt_sorted = gt[np.argsort(scores)[::-1]]
            positive_rank = np.where(gt_sorted == 1)[0]
            return compute_average_precision(positive_rank)
        else:
            d = {}
            for mode in ('easy', 'medium', 'hard'):
                gt = self.get_query_groundtruth(query_idx, 'AP', mode)  # labels in {-1, 0, 1}
                assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
                assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
                keep = (gt != 0)  # remove null labels
                if sum(gt[keep] > 0) == 0:  # exclude queries with no relevants from the evaluation
                    d[mode] = -1
                else:
                    gt2, scores2 = gt[keep], scores[keep]
                    gt_sorted = gt2[np.argsort(scores2)[::-1]]
                    positive_rank = np.where(gt_sorted == 1)[0]
                    d[mode] = compute_average_precision(positive_rank)
            return d

class Hyundai1f(ImageListRelevants):
    def __init__(self, train):
        ImageListRelevants.__init__(self, root=os.path.join(DB_ROOT, '1f'), train=train)

class Hyundaib1(ImageListRelevants):
    def __init__(self, train):
        ImageListRelevants.__init__(self, root=os.path.join(DB_ROOT, 'b1'), train=train)


