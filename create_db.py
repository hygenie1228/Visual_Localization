import json
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils import common
from dirtorch.utils.common import pool

import dirtorch.test_dir as test
import dirtorch.nets as nets
import dirtorch.datasets as datasets

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

def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')
    parser.add_argument('--db', type=str, required=True, help='path to output features')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parser()
    args.iscuda = common.torch_set_gpu(args.gpu)

    dataset_cmd = "datasets." + args.dataset + "(True)"
    dataset = eval(dataset_cmd)
    print("Dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda)
    net.pca = None  

    create_db(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                threads=args.threads, dbg=args.dbg, whiten=args.whiten, output=args.db)
