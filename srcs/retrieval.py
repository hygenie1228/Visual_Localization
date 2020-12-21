import sys, os
import os.path as osp
import pdb, json, tqdm, numpy as np, torch
import torch.nn.functional as F
from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader
import dirtorch.test_dir as test
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl
import pickle as pkl, hashlib
from torchvision import transforms
from PIL import Image

def load_db(db_paths, db_list):
    db = []
    end_points = []
    end = 0
    for dbpath in db_paths:
        bdescs = np.load(dbpath)
        bdescs = torch.tensor(bdescs)
        db.append(bdescs)
        end = end + len(bdescs)
        end_points.append(end)

    db = torch.cat(db, dim=0)
    dataset_list = []
    for name in db_list:
        dataset_cmd = 'datasets.' + name + '(' + str(True) + ')'
        dataset = eval(dataset_cmd)
        dataset_list.append(dataset)

    return (db, end_points, dataset_list)


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = (nets.create_model)(pretrained='', **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


def get_topk_images(dataset_list, db_set, net, img, topk=30):
    if hasattr(net, 'eval'):
        net.eval()
    from dirtorch.utils import transforms
    trf_chain = (transforms.create)('', to_tensor=True, **net.preprocess)
    img = trf_chain(img)
    _, H, W = img.shape
    img = img.view(1, 3, H, W)
    img = common.variables([img], net.iscuda)[0]
    desc = net(img)
    desc = desc.cpu()
    db, end_idxs, db_list = db_set
    smiliarity = torch.matmul(db, desc)
    max_idx = torch.argmax(smiliarity)
    _, idxs = torch.topk(smiliarity, topk)
    max_idxs = []
    db_idxs = []
    for idx in idxs:
        db_name = ''
        for i, end in enumerate(end_idxs):
            if idx >= end:
                idx = idx - end
            else:
                db_idx = i
                break

        max_idxs.append(idx)
        db_idxs.append(db_idx)

    file_paths = []
    for i, (db_idx, img_idx) in enumerate(zip(db_idxs, max_idxs)):
        dataset = dataset_list[db_idx]
        img = dataset.get_image(img_idx)
        file_paths.append(dataset.get_query_filename(img_idx))

    return (db_idxs, file_paths)


def one_image_retrieval(net, db_list, img, topk=30):
    db_paths = ['db/' + name + '.npy' for name in db_list]
    db, end_idxs, dataset_list = load_db(db_paths, db_list)
    db_set = [db, end_idxs, db_list]
    db_idxs, file_paths = get_topk_images(dataset_list, db_set, net, img, topk=topk)
    return (db_idxs, dataset_list, file_paths)


def visualize_retrieval_results(file_paths, img, result_path):
    img.save(result_path + 'query.jpg')
    for i, path in enumerate(file_paths[:10]):
        img = Image.open(path)
        img.save(result_path + 'retrieval_result' + str(i) + '.jpg')