import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.cm as cm
import open3d
import cv2
from pyquaternion import Quaternion

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
        if np.count_nonzero(mconf> threshold) > 180 or threshold < 0:
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

def get_pcd_list(path, target_dataset):
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

        if len(pcd_list) > 15:
            return pcd_list, range_i
        range_i = range_i + 1
    

def project_pcd_points(pcd_paths, img_shape, K, target_camera_pose, target_dataset, target_date):
    pcd_2ds = []
    pcd_3ds = []
    
    for pcd_path in pcd_paths[:16]:
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
    threshold = 7.5

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