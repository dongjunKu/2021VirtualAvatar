import json
import os
import os.path as osp
import pickle
import numpy as np
import glob

from utils import *
from create_pcd_full_body import create_human_pcd

# SMPL
from smpl.serialization import load_model
from smplify_core import run_single_fit

# Trimesh
import trimesh

# Teaser
import open3d as o3d
from registration import registrate_pcd, merge_pcd

# clothes
from clothes import create_body_texture, make_avatar, display
from utils import reconstruct_mesh_uv, saveMeshObjFile

import matplotlib.pyplot as plt

"""

   Sinle Image Fitting Function

"""


def main(base_dir,
         out_dir,
         n_betas=10,
         gender='male',  # male, female, neutral
         viz=True):
    """
    Set up dataset dependent paths to image and joint data, saves results.

    :param base_dir: folder containing LSP images and data
    :param out_dir: output folder
    :param n_betas: number of shape coefficients considered during optimization
    :param use_neutral: boolean, if True enables uses the neutral gender SMPL model
    :param viz: boolean, if True enables visualization during optimization
    """

    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # load models
    sph_regs = None
    if gender == 'male':
        model = load_model(MODEL_MALE_PATH)
    elif gender == 'female':
        model = load_model(MODEL_FEMALE_PATH)
    else:
        model = load_model(MODEL_NEUTRAL_PATH)

    # load images
    im_color_file_list = glob.glob(osp.join(imageFileDir, "color/*.png"))
    im_color_file_list = sorted(im_color_file_list)
    im_depth_file_list = glob.glob(osp.join(imageFileDir, "depth/*.png"))
    im_depth_file_list = sorted(im_depth_file_list)
    im_segment_file_list = glob.glob(osp.join(imageFileDir, "segment/*.png"))
    im_segment_file_list = sorted(im_segment_file_list)

    im_color_list = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                     for f in im_color_file_list]
    im_depth_list = [cv2.imread(f, cv2.IMREAD_UNCHANGED)
                     for f in im_depth_file_list]
    im_segment_list = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                       for f in im_segment_file_list]

    human_pcd_list, human_uv_list = [], []
    for c, d, s in zip(im_color_list, im_depth_list, im_segment_list):
        pcd, uv = create_human_pcd(c, d, s, intrinsic)
        human_pcd_list.append(pcd)
        human_uv_list.append(uv)
    clothes_mask = [2, 4, 10, 13, 14, 15, 16, 17]
    clothes_pcd_list, clothes_uv_list = [], []
    for c, d, s in zip(im_color_list, im_depth_list, im_segment_list):
        pcd, uv = create_human_pcd(c, d, s, intrinsic, clothes_mask)
        clothes_pcd_list.append(pcd)
        clothes_uv_list.append(uv)

    o3d.visualization.draw_geometries(human_pcd_list)

    # registration
    voxel_size = 0.02
    sample_voxel_size = 0.04

    human_pcd_list, extrinsic_list = registrate_pcd(human_pcd_list, voxel_size)
    human_pcd = merge_pcd(human_pcd_list[::2])
    o3d.visualization.draw_geometries([human_pcd])

    clothes_pcd_list, extrinsic_list = registrate_pcd(
        clothes_pcd_list, voxel_size, extrinsic_list)
    o3d.visualization.draw_geometries(clothes_pcd_list)

    # SMPLify
    target = trimesh.PointCloud(np.asarray(
        human_pcd_list[0].points), np.asarray(human_pcd_list[0].colors))

    with open(jsonFilePath, "r") as f:
        jsonstr = f.read()
    joints_json = json.loads(jsonstr)
    j3d = joints_json['people'][0]['pose_keypoints_3d']
    j3d = np.array(j3d).reshape(-1, 3)
    j3d = j3d / 1000  # kinect

    params = run_single_fit(
        target,
        j3d,
        model,
        regs=sph_regs,
        n_betas=n_betas,
        viz=viz,
        out_dir=out_dir)

    # create clothes mesh and clothes texture

    # method 1
    clothes_pcd = merge_pcd([pcd.voxel_down_sample(
        voxel_size=sample_voxel_size) for pcd in clothes_pcd_list])
    clothes_mesh = create_poisson_mesh(clothes_pcd, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([clothes_mesh])

    clothes_mesh = clothes_mesh.simplify_quadric_decimation(
        target_number_of_triangles=10000)
    o3d.visualization.draw_geometries([clothes_mesh])
    # method 1 end

    clothes_mesh = o3d.geometry.TriangleMesh(
        clothes_mesh.vertices, clothes_mesh.triangles)  # FIXME 새로 안만들면 색깔이 어두워짐;;

    def l_i(l, idx=[0, 4, 3, 5]):  # for mode priority
        return [l[i] for i in idx]

    mesh_uv, face_label = reconstruct_mesh_uv(
        np.asarray(clothes_mesh.vertices), np.asarray(clothes_mesh.triangles), l_i(clothes_pcd_list), l_i(clothes_uv_list))
    clothes_mesh.triangle_material_ids = o3d.utility.IntVector(
        list(face_label))
    clothes_mesh.textures = [o3d.geometry.Image(
        img) for img in l_i(im_color_list)]

    clothes_mesh.triangle_uvs = o3d.utility.Vector2dVector(mesh_uv)

    o3d.io.write_triangle_mesh("./clothes_mesh.obj", clothes_mesh, write_ascii=True, compressed=False,
                               write_vertex_normals=True, write_vertex_colors=True,
                               write_triangle_uvs=True, print_progress=False)

    # create body texture
    body_mesh = create_body_texture(
        model, l_i(human_pcd_list), l_i(human_uv_list), l_i(im_color_list), params)

    # show time
    cps_t2m, coeff = make_avatar(
        model, clothes_mesh, params)

    display(model, clothes_mesh, cps_t2m, coeff,
            body_mesh, params)

    with open(osp.join(out_dir, 'output.pkl'), 'wb') as outf:  # 'wb' for python 3?
        pickle.dump(params, outf)


if __name__ == '__main__':

    """  Parsing the arguments and load the SMPL specific model files    """

    base_dir = "."
    out_dir = osp.join(base_dir, 'results')
    n_betas = 10
    gender = 'male'
    viz = True

    # 1. load SMPL models (independent upon dataset)
    MODEL_DIR = osp.join(base_dir, 'models')
    MODEL_NEUTRAL_PATH = osp.join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = osp.join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = osp.join(
        MODEL_DIR, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    imageFileDir = osp.join(base_dir, 'input/images')
    plyFileDir = osp.join(base_dir, 'input/pcds')
    jsonFilePath = osp.join(base_dir, 'input/joints/joint0.json')

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        640, 576, 502.724945, 502.854401, 323.970764, 326.964050)

    # 3. call the  main function
    main(base_dir, out_dir, n_betas, gender, viz)
