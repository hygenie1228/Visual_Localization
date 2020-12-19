import json
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils import common
import dirtorch.datasets as datasets

import pickle as pkl
import hashlib
from torchvision import transforms
from PIL import Image

from models.matching import Matching
from models.utils import read_image, make_matching_plot
from pathlib import Path
import matplotlib.cm as cm
import open3d
import cv2
from pyquaternion import Quaternion

from retrieval import load_model, one_image_retrieval, visualize_retrieval_results
from reranking import load_matching_model, re_ranking, visual_matching_result
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def get_camera_timestammp(img_path):
    s = img_path.find('images/')
    t = img_path.find('.jpg')
    camera_timestamp = img_path[s+7:t]
    k = camera_timestamp.find('_')
    return camera_timestamp[:k], camera_timestamp[k+1:], img_path[s-20:s-1]

def get_lidar_timestamp(pcd_path):
    s = pcd_path.find('pointclouds_data/')
    t = pcd_path.find('.pcd')
    pcd_timestamp = pcd_path[s+17:t]
    s = pcd_timestamp.find('_')
    return pcd_timestamp[:s], pcd_timestamp[s+1:]

def get_match_kpt(matching_net, inp0, inp1, scales0, scales1):
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

    # update scale
    threshold = 0.7    
    while True:
        if np.count_nonzero(mconf> threshold) > 150 or threshold < 0:
            break
        else:
            threshold = threshold - 0.1
        
    new_mkpts0 = []
    new_mkpts1 = []
    for i, c in enumerate(mconf):
        if c > threshold:
            point = [mkpts0[i][0] * scales0[0], mkpts0[i][1] * scales0[1]]
            new_mkpts0.append(point)
            point = [mkpts1[i][0] * scales1[0], mkpts1[i][1] * scales1[1]]
            new_mkpts1.append(point)

    new_mkpts0 = np.asarray(new_mkpts0)
    new_mkpts1 = np.asarray(new_mkpts1)
    return new_mkpts0, new_mkpts1

def get_pose_matrix(pose):  
    P = np.identity(4)
    P[:3, 3] = pose[:3]
    P[:3, :3] = Quaternion(pose[3:]).rotation_matrix

    return P

def get_intrinsic_matrix(parameter):
    parameter = parameter[3:7]
    parameter = [float(x) for x in parameter]
    K = np.identity(3)
    K[0][0] = parameter[0]
    K[1][1] = parameter[1]
    K[0][2] = parameter[2]
    K[1][2] = parameter[3]
    return K

def get_pcd_list(path):
    range_i = 1

    while True:
        image_paths = []
        pcd_paths = []

        target_camera, timestamp, date = get_camera_timestammp(path)
        timestamp = int(timestamp[:-5])
        timestamp_list = []
        for i in range(1, 2*range_i + 1):
            timestamp_list.append(timestamp + range_i - i)
        
        pcd_list = target_dataset.get_pointclouds(timestamp_list, date)

        if len(pcd_list) > 7:
            return pcd_list, range_i
        range_i = range_i + 1
    

def project_pcd_points(pcd_paths, img_shape, K, target_camera_pose):
    pcd_2ds = []
    pcd_3ds = []
    
    for pcd_path in pcd_paths[:8]:
        lidar, pcd_timestamp = get_lidar_timestamp(pcd_path)
        target_lidar_pose = target_dataset.get_groundtruth(lidar+ '_pose', lidar+ '_stamp', pcd_timestamp, target_date)
        target_lidar_pose = get_pose_matrix(target_lidar_pose)

        pc = open3d.read_point_cloud(pcd_path) 
        pc_array = np.asarray(pc.points)
        o = np.ones((pc_array.shape[0], 1))
        pc_array = np.concatenate((pc_array, o), axis=1)

        pcd_world = np.matmul(target_lidar_pose, pc_array.T)
        pcd_camera = np.matmul(np.linalg.inv(target_camera_pose)[:3], pcd_world)
        pcd_pixel = np.matmul(K, pcd_camera)
        pcd_pixel = pcd_pixel / pcd_pixel[2]
        pcd_pixel = pcd_pixel[:2].T

        new_pcd_2d = []
        new_pcd_3d = []

        for p, p3 in zip(pcd_pixel, pcd_world.T[:, :3]):
            if p[0] >= 0 and p[0] <= img_shape[1]:
                if p[1] >= 0 and p[1] <= img_shape[0]:
                    new_pcd_2d.append(p)
                    new_pcd_3d.append(p3)

        pcd_2ds.extend(new_pcd_2d)
        pcd_3ds.extend(new_pcd_3d)

    pcd_2ds = torch.tensor(pcd_2ds)
    pcd_3ds = torch.tensor(pcd_3ds)

    return pcd_2ds, pcd_3ds

def find_closest_points(mkpts0, mkpts1, pcd_2ds, pcd_3ds):
    new_mkpts0 = []
    new_mkpts1 = []
    new_pcd_2ds = []
    new_pcd_3ds = []
    threshold = 7

    for mkpt0, mkpt1 in zip(mkpts0, mkpts1):
        distance = pcd_2ds - mkpt1
        distance = torch.norm(distance, dim=1)

        min_dist, min_idx = torch.topk(distance, 1, largest=False) 
        if min_dist[0] < threshold:
            new_mkpts0.append(mkpt0)
            new_mkpts1.append(mkpt1)
            new_pcd_2ds.append(pcd_2ds[min_idx[0]].numpy())
            new_pcd_3ds.append(pcd_3ds[min_idx[0]].numpy())

    return np.asarray(new_mkpts0), np.asarray(new_mkpts1), np.asarray(new_pcd_2ds), np.asarray(new_pcd_3ds)

def visualize_projection_result(image0, image1, mkpts0, mkpts1, new_mkpts0, new_mkpts1, pcd_2ds, new_pcd_3ds, result_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = new_pcd_3ds[:, 0]
    y = new_pcd_3ds[:, 1]
    z = new_pcd_3ds[:, 2]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    plt.savefig('results/3dplot.png')

    color = (0, 0, 255)
    for mkpt in mkpts0:
        mkpt = (int(mkpt[0]), int(mkpt[1]))
        image0 = cv2.line(image0, mkpt, mkpt, color, 25)
    
    color = (0, 255, 0)
    for mkpt in new_mkpts0:
        mkpt = (int(mkpt[0]), int(mkpt[1]))
        image0 = cv2.line(image0, mkpt, mkpt, color, 15)

    cv2.imwrite(result_path + '/0.jpg', image0)

    color = (255, 0, 0)
    for mkpt in pcd_2ds:
        mkpt = (int(mkpt[0]), int(mkpt[1]))
        image1 = cv2.line(image1, mkpt, mkpt, color, 5)

    color = (0, 0, 255)
    for mkpt in mkpts1:
        mkpt = (int(mkpt[0]), int(mkpt[1]))
        image1 = cv2.line(image1, mkpt, mkpt, color, 25)

    color = (0, 255, 0)
    for mkpt in new_mkpts1:
        mkpt = (int(mkpt[0]), int(mkpt[1]))
        image1 = cv2.line(image1, mkpt, mkpt, color, 15)

    cv2.imwrite(result_path + '/1.jpg', image1)


def estimate_pose_matrix(points_3D, points_2D, LCam_K):
    distCoeffs = np.zeros((4, 1), dtype='float32')
    print(points_3D.shape)
    _,solvR,solvt,inlierR = cv2.solvePnPRansac(points_3D.astype("float64"), \
                                                points_2D.astype("float64"), \
                                                LCam_K.astype("float64"), \
                                                distCoeffs.astype("float64"), \
                                                iterationsCount=50000, \
                                                useExtrinsicGuess = True, \
                                                confidence = 0.9, \
                                                reprojectionError = 16, \
                                                flags = cv2.SOLVEPNP_P3P)

    
    solvRR,_ = cv2.Rodrigues(solvR)
    solvRR_inv = np.linalg.inv(solvRR)
    solvtt = -np.matmul(solvRR_inv, solvt)

    

    quaternion_R = Quaternion(matrix=solvRR_inv) 

    inlierR = inlierR.flatten()
    points_3D = points_3D[inlierR]
    points_2D = points_2D[inlierR]

    o = np.ones((points_3D.shape[0], 1))
    test_pcd_3d = np.concatenate((points_3D, o), axis=1)
    #M = np.concatenate((solvRR_inv, solvtt), axis=1)
    M = np.concatenate((solvRR, solvt), axis=1)

    projected_points = np.matmul(M, test_pcd_3d.T)
    projected_points = np.matmul(LCam_K, projected_points)
    projected_points = projected_points / projected_points[2]
    
    distance = projected_points[:2].T - points_2D
    distance = np.linalg.norm(distance, axis=1)
    inlier_count = len(inlierR)
    distance = np.average(distance)


    return quaternion_R, solvtt, inlier_count, distance



def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

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
    result_path = "results/"
    json_data = []
    json_log = []

    # load test dataset
    dataset_name = db_list[1]
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
    for test_idx in range(444, 1000):
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
        pcd_paths, range_i = get_pcd_list(target_img_path)
        
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
        pcd_2ds, pcd_3ds = project_pcd_points(pcd_paths, image1.shape, target_K, target_camera_pose)
        pcd_points = pcd_2ds.shape[0]

        # find closest points
        new_mkpts0, new_mkpts1, new_pcd_2ds, new_pcd_3ds = find_closest_points(mkpts0, mkpts1, pcd_2ds, pcd_3ds)
        final_points = new_mkpts0.shape[0]
        #print("Estimate pose...")
        if final_points >= 8:
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
        print(inliers, " / ", final_points)

        if dataset_name == "Hyundaib1":
            floor = "b1"
        else:
            floor = "1f"
        
        s = target_img_path.find('images/')
        name = target_img_path[s+7:]
        

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
    
        file_path = "results/result_b1_444-999.json"
        with open(file_path, 'w') as outfile:
            json.dump(json_data, outfile)

        file_path = "results/log_b1_444-999.json"
        with open(file_path, 'w') as outfile:
            json.dump(json_log, outfile)
