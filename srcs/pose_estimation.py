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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def get_intrinsic_matrix(parameter):
    parameter = parameter[3:7]
    parameter = [float(x) for x in parameter]
    K = np.identity(3)
    K[0][0] = parameter[0]
    K[1][1] = parameter[1]
    K[0][2] = parameter[2]
    K[1][2] = parameter[3]
    return K

def estimate_pose_matrix(points_3D, points_2D, LCam_K):
    distCoeffs = np.zeros((4, 1), dtype='float32')
    
    _,solvR,solvt,inlierR = cv2.solvePnPRansac(points_3D.astype("float64"), \
                                                points_2D.astype("float64"), \
                                                LCam_K.astype("float64"), \
                                                distCoeffs.astype("float64"), \
                                                iterationsCount=50000, \
                                                useExtrinsicGuess = True, \
                                                confidence = 0.99, \
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
    M = np.concatenate((solvRR, solvt), axis=1)

    projected_points = np.matmul(M, test_pcd_3d.T)
    projected_points = np.matmul(LCam_K, projected_points)
    projected_points = projected_points / projected_points[2]
    
    distance = projected_points[:2].T - points_2D
    distance = np.linalg.norm(distance, axis=1)
    inlier_count = len(inlierR)
    distance = np.average(distance)

    return quaternion_R, solvtt, inlier_count, distance