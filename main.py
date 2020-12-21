import json
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils import common
import dirtorch.datasets as datasets

from models.matching import Matching
from models.utils import read_image, make_matching_plot
from pathlib import Path
import matplotlib.cm as cm
import open3d
import cv2
from pyquaternion import Quaternion

from retrieval import load_model, one_image_retrieval, visualize_retrieval_results
from reranking import load_matching_model, re_ranking, visual_matching_result
from projection import get_match_kpt, get_pcd_list, get_camera_timestammp, get_lidar_timestamp, get_pose_matrix, project_pcd_points, find_closest_points, visualize_projection_result
from pose_estimation import get_intrinsic_matrix, estimate_pose_matrix

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')
    parser.add_argument('--result', type=str, required=True, help='path to save results files')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--db', type=str, default="", help='path to output features')

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

    db_list = ["Hyundai1f", "Hyundaib1"]
    visualize = False
    result_path = args.result + '/'
    json_data = []
    json_log = []

    # load test dataset
    if args.dataset == "Hyundaib1":
        dataset_name = db_list[1]
    elif args.dataset == "Hyundai1f":
        dataset_name = db_list[0]
    else:
        print("ERROR : Invalid dataset!")

    dataset_cmd = "datasets." + dataset_name + "(False)"
    dataset = eval(dataset_cmd)
    print("Dataset:", dataset)

    # load - retrieval network
    retrieval_net = load_model(args.checkpoint, args.iscuda)
    retrieval_net.pca = None

    # load - matching network
    device = 'cuda' if args.iscuda else 'cpu'
    matching_net = load_matching_model(device)
    
    #for test_idx in range(dataset.nimg):
    for test_idx in range(1911, 1912):
        print("idx : ", test_idx)

        # get test image
        img = dataset.get_image(test_idx)
        query_img_path = dataset.get_query_filename(test_idx)

        # retrieval - get topk images
        db_idxs, dataset_list, file_paths = one_image_retrieval(retrieval_net, db_list, img, topk=30)

        # visualize retireved / matching images
        if visualize:
            visualize_retrieval_results(file_paths, img, result_path)

        # re-ranking
        topk_idx, target_img_path = re_ranking(query_img_path, file_paths, matching_net, device, topk=5)
        db_idx = db_idxs[topk_idx]
        
        query_dataset = dataset
        target_dataset = dataset_list[db_idx]

        # get pcd list
        pcd_paths, range_i = get_pcd_list(target_img_path, target_dataset)
        
        # visualize matching result
        if visualize:
            visual_matching_result(matching_net, query_img_path, target_img_path, device, result_path)

        # camera parameter, pose
        query_camera, _, query_date = get_camera_timestammp(query_img_path)
        target_camera, img_timestamp, target_date = get_camera_timestammp(target_img_path)
        query_parameter = query_dataset.get_parameters(query_camera, query_date)
        target_parameter = target_dataset.get_parameters(target_camera, target_date)

        target_K = get_intrinsic_matrix(target_parameter)
        query_K = get_intrinsic_matrix(query_parameter)

        target_camera_pose = target_dataset.get_groundtruth(target_camera+ '_pose', target_camera+ '_stamp', img_timestamp, target_date)
        target_camera_pose = get_pose_matrix(target_camera_pose)

        image0, inp0, scales0 = read_image(
            query_img_path, device, [640, 480], 0, False)

        image1, inp1, scales1 = read_image(
            target_img_path, device, [640, 480], 0, False)

        #get matched keypoints
        mkpts0, mkpts1 = get_match_kpt(matching_net, inp0, inp1, scales0, scales1)
        match_points = mkpts0.shape[0]

        image0 = cv2.imread(query_img_path)
        image1 = cv2.imread(target_img_path)

        # project pcd point
        pcd_2ds, pcd_3ds = project_pcd_points(pcd_paths, image1.shape, target_K, target_camera_pose, target_dataset, target_date)
        pcd_points = pcd_2ds.shape[0]

        # find closest points
        new_mkpts0, new_mkpts1, new_pcd_2ds, new_pcd_3ds = find_closest_points(mkpts0, mkpts1, pcd_2ds, pcd_3ds)
        final_points = new_mkpts0.shape[0]
        #print("Estimate pose...")
        if final_points >= 10:
            quaternion_R, solvtt, inliers, distance = estimate_pose_matrix(new_pcd_3ds, new_mkpts0, query_K)
            if quaternion_R == 0:
                qw, qx, qy, qz = 0, 0, 0, 0
                x, y, z = 0, 0, 0
                inliers = 0
                distance = 1000
            else :
                qw, qx, qy, qz = quaternion_R.elements
                x, y, z = solvtt
        else:
            qw, qx, qy, qz = 0, 0, 0, 0
            x, y, z = 0, 0, 0
            inliers = 0
            distance = 1000
        
        if final_points == 0:
            final_points = 1
        print("Inliers / Total points : ", inliers, " / ", final_points)
        

        # Report result & log
        if dataset_name == "Hyundaib1":
            floor = "b1"
        else:
            floor = "1f"
        
        s = query_img_path.find('images/')
        name = query_img_path[s+7:]
        
        data = {
            "floor": floor,
            "name": name,
            "qw": float(qw),
            "qx": float(qx),
            "qy": float(qy),
            "qz": float(qz),
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }

        json_data.append(data)

        data = {
            "idx": test_idx,
            "match_points": match_points,
            "pcd_points": pcd_points,
            "final_points": final_points,
            "inliers": inliers,
            "epsilon": inliers / final_points,
            "range_i": range_i,
            "distance": distance
        }
        
        json_log.append(data)
        
        if visualize:
            visualize_projection_result(image0, image1, mkpts0, mkpts1, new_mkpts0, new_mkpts1, pcd_2ds, new_pcd_3ds, result_path)
        
        file_path = "results/result.json"  
        with open(file_path, 'w') as outfile:
            json.dump(json_data, outfile)

        file_path = "results/log.json"
        with open(file_path, 'w') as outfile:
            json.dump(json_log, outfile)