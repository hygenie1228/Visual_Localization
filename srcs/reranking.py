import numpy as np
import torch

from models.matching import Matching
from models.utils import read_image, make_matching_plot
from pathlib import Path
import matplotlib.cm as cm

def load_matching_model(device):
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    matching = Matching(config).eval().to(device)

    return matching

def re_ranking(img_path, file_paths, matching, device, topk=10):
    image0, inp0, scales0 = read_image(
            img_path, device, [640, 480], 0, False)

    mconf_list = []
    for img in file_paths:
        image1, inp1, scales1 = read_image(
            img, device, [640, 480], 0, False)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        mconf_list.append(mconf)

    threshold=0.7
    while True:
        conf_list = []
        for mconf in mconf_list:
            if len(mconf) != 0:
                conf_list.append(np.count_nonzero(mconf> threshold))
            else:
                conf_list.append(0.0)
                
        conf_list = np.array(conf_list) 
        topk_idxs = np.argsort(conf_list)
        topk_idxs = topk_idxs[::-1]
        topk_idxs = topk_idxs[:topk]

        inliers = conf_list[topk_idxs]
        if inliers[0] > 180 or threshold < 0:
            break
        else:
            threshold = threshold - 0.1
    
    target_img_paths = []
    for i in topk_idxs:
        target_img_paths.append(file_paths[i])
    
    return topk_idxs[0], target_img_paths[0]


def visual_matching_result(matching_net, img_path, target_img_path, device, result_path):
    resize = [640, 480]
    resize_float = False
    rot0, rot1 = 0, 0

    image0, inp0, scales0 = read_image(
        img_path, device, resize, rot0, resize_float)
    image1, inp1, scales1 = read_image(
        target_img_path, device, resize, rot1, resize_float)

    # Perform the matching.
    pred = matching_net({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]

    # Display extra parameter info.
    k_thresh = matching_net.superpoint.config['keypoint_threshold']
    m_thresh = matching_net.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]
    output_dir = Path(result_path)
    viz_path = output_dir / 'match_result.png'

    show_keypoints = False
    fast_viz= False
    opencv_display = False

    make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        text, viz_path, show_keypoints,
        fast_viz, opencv_display, 'Matches', small_text)