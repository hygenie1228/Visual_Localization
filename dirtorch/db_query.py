import sys
import os
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader

import dirtorch.test_dir as test
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl

import pickle as pkl
import hashlib
from torchvision import transforms

def create_db(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
                     threads=8, batch_size=16, output=None, dbg=()):
    """ 
        Extract features from trained model (network) on a given dataset.
    """
    print("\n>> Create DB...")

    bdescs = []
    trfs_list = [trfs] if isinstance(trfs, str) else trfs

    for trfs in trfs_list:
        kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
        bdescs.append(test.extract_image_features(db, trfs, net, desc="DB", **kw))

    bdescs = F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1)
    np.save(output, bdescs.cpu().numpy())

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

    db = torch.cat(db, dim = 0)

    dataset_list = []
    for name in db_list:
        dataset_cmd = "datasets." + name + "(" + str(True) + ")"
        dataset = eval(dataset_cmd)
        dataset_list.append(dataset)

    return db, end_points, dataset_list

def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


def get_topk_images(db_set, net, img, topk=30):
    
    if hasattr(net, 'eval'):
        net.eval()
    
    from dirtorch.utils import transforms
    trf_chain = transforms.create('', to_tensor=True, **net.preprocess)

    img = trf_chain(img)
    _, H, W = img.shape
    img = img.view(1, 3, H, W)
    img = common.variables([img], net.iscuda)[0]
    desc = net(img)
    desc = desc.cpu()

    db, endpoints, db_list = db_set
    smiliarity = torch.matmul(db, desc)
    max_idx = torch.argmax(smiliarity)

    _, idxs = torch.topk(smiliarity, topk)
    max_idxs = []
    db_idxs = []
    for idx in idxs:
        db_name = ''
        for i, end in enumerate(endpoints):
            if idx >= end:
                idx = idx - end
            else:
                db_idx = i
                break
        max_idxs.append(idx)
        db_idxs.append(db_idx)

    return max_idxs, db_idxs
    
    
def save_results(db_idxs, max_idxs):
    for i, (db_idx, img_idx) in enumerate(zip(db_idxs, max_idxs)):
        dataset = dataset_list[db_idx]
        img = dataset.get_image(img_idx)
        img.save("results/"+ str(i) + ".jpg")

    print("Images saved!!")


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--db', type=str, default="", help='path to output features')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')
    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = arg_parser()
    args.iscuda = common.torch_set_gpu(args.gpu)

    train = False
    db_list = ["Hyundai1f", "Hyundaib1"]
    topk = 30
    visualize = True
    db_save_path = "db/Hyundai1f.npy"


    dataset_cmd = "datasets." + args.dataset + "(" + str(train) + ")"
    dataset = eval(dataset_cmd)
    print("Dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None



    db_paths = ["db/" + name + ".npy" for name in db_list]
    test_idx = 20


    if train:
        create_db(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                            threads=args.threads, dbg=args.dbg, whiten=args.whiten, output=db_save_path)
    else:
        db, end_points, dataset_list = load_db(db_paths, db_list)
        db_set = [db, end_points, db_list]

        img = dataset.get_image(test_idx)
        max_idxs, db_idxs = get_topk_images(db_set, net, img, topk=topk)
        
        if visualize:
            img.save("results/query.jpg")
            save_results(db_idxs, max_idxs)