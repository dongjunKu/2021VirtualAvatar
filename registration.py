import open3d as o3d
import teaserpp_python
import numpy as np
import copy
from numpy.linalg import inv
from scipy.spatial import cKDTree
import cv2
import os
import glob


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T


def pcd2rgb(pcd):
    return np.asarray(pcd.colors)


def extract_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd.orient_normals_towards_camera_location()

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def teaser(A_pcd_raw, B_pcd_raw, voxel_size):
    # voxel downsample both clouds
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=voxel_size)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=voxel_size)

    A_xyz = pcd2xyz(A_pcd)  # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd)  # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd, voxel_size)
    B_feats = extract_fpfh(B_pcd, voxel_size)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)

    """
    # GU consideing color
    A_rgb = pcd2rgb(A_pcd)
    B_rgb = pcd2rgb(B_pcd)

    A_corr_rgb = A_rgb[corrs_A]
    B_corr_rgb = B_rgb[corrs_B]

    dist_rgb = np.sum((A_corr_rgb - B_corr_rgb)**2, axis=-1)**0.5

    thres = 0.1
    inliers = dist_rgb < thres

    corrs_A = corrs_A[inliers]
    corrs_B = corrs_B[inliers]
    # GU
    """

    A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]

    # robust global registration using TEASER++
    NOISE_BOUND = voxel_size
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser, t_teaser)

    # local refinement using ICP
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp = icp_sol.transformation

    return T_icp


def registrate_pcd(pcd_list, voxel_size, extrinsic_list=None):

    if not extrinsic_list:
        extrinsic_list = [np.identity(4)]

        # 변환 행렬 계산
        for i in range(0, len(pcd_list)-1):
            pcd_ = copy.deepcopy(pcd_list[i])
            pcd__ = copy.deepcopy(pcd_list[i+1])
            T_icp = teaser(pcd_, pcd__, voxel_size)
            extrinsic_list.append(T_icp)

        # 변환 행렬 연쇄
        for i in range(1, len(extrinsic_list)):
            prev_extrinsic = extrinsic_list[i-1]
            extrinsic = extrinsic_list[i]
            extrinsic = np.matmul(extrinsic, prev_extrinsic)
            extrinsic_list[i] = extrinsic

    target_temp = copy.deepcopy(pcd_list)
    for i in range(1, len(pcd_list)):
        target_temp[i] = target_temp[i].transform(inv(extrinsic_list[i]))

    return target_temp, extrinsic_list


def merge_pcd(pcd_list):

    np_pcd_list = []
    np_pcdc_list = []
    np_pcdn_list = []

    for pcd in pcd_list:

        np_pcd_list.append(np.asarray(pcd.points))
        # color
        np_pcdc_list.append(np.asarray(pcd.colors))
        # normal
        np_pcdn_list.append(np.asarray(pcd.normals))

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(
        np.concatenate(np_pcd_list, axis=0))
    # color
    new_pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate(np_pcdc_list, axis=0))
    # normal
    new_pcd.normals = o3d.utility.Vector3dVector(
        np.concatenate(np_pcdn_list, axis=0))

    return new_pcd


if __name__ == "__main__":

    folder_path = "./input/pcds"
    source_idx = 0

    # 1. load rgb-data
    ply_list = glob.glob(os.path.join(folder_path, "*.ply"))
    ply_list = sorted(ply_list)

    full_body_ply_list = [ply for ply in ply_list if 'full' in ply]
    part_body_ply_list = [ply for ply in ply_list if 'full' not in ply]

    print(full_body_ply_list)
    print(part_body_ply_list)

    full_body_pcd_list = [o3d.io.read_point_cloud(
        ply) for ply in full_body_ply_list]
    part_body_pcd_list = [o3d.io.read_point_cloud(
        ply) for ply in part_body_ply_list]

    voxel_size = 0.02

    # full body
    pcd_list, extrinsic_list = registrate_pcd(full_body_pcd_list, voxel_size)
    result = merge_pcd(pcd_list[::2])
    o3d.visualization.draw_geometries([result])
    # o3d.io.write_point_cloud("./input/pcd_.ply", result)
    result = result.voxel_down_sample(voxel_size=0.02)
    # o3d.io.write_point_cloud("./input/full_body_pcd.ply", result)

    # part body using extrinsic_list of full body
    pcd_list, extrinsic_list = registrate_pcd(
        part_body_pcd_list, voxel_size, extrinsic_list)
    result = merge_pcd(pcd_list[::4])
    o3d.visualization.draw_geometries([result])
    # o3d.io.write_point_cloud("./input/pcd_.ply", result)
    result = result.voxel_down_sample(voxel_size=0.02)
    o3d.io.write_point_cloud("./input/part_body_pcd.ply", result)
