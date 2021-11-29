import open3d as o3d
import numpy as np
import os.path as osp
import cv2
import json
import matplotlib.pyplot as plt
"""
Note:
{
  "Color_intrinsics" : [1280.000000, 720.000000, 605.927002, 605.799927, 639.942627, 367.209930],
  "Depth_intrinsics" : [640.000000, 576.000000, 502.724945, 502.854401, 323.970764, 326.964050]
}
(width, height, fx, fy, cx, cy)

"""

cihppgn_label_dict = {0: 'background',
                      1: 'hat',
                      2: 'hair',
                      3: 'glove',
                      4: 'sunglasses',    # skin
                      5: 'upperclothes',
                      6: 'dress',
                      7: 'coat',
                      8: 'socks',
                      9: 'pants',
                      10: 'torso-skin',   # skin
                      11: 'scarf',
                      12: 'skirt',
                      13: 'face',         # skin
                      14: 'leftArm',      # skin
                      15: 'rightArm',     # skin
                      16: 'leftLeg',      # skin
                      17: 'rightLeg',     # skin
                      18: 'leftShoe',
                      19: 'rightShoe'}


def create_human_pcd(im_color, im_depth, im_segment, intrinsic, mask_parts=None, extrinsic=None):

    im_color, im_depth = im_color.copy(), im_depth.copy()

    if not extrinsic:
        extrinsic = np.identity(4)

    height, width = im_color.shape[:2]

    mask = (im_segment != 0)

    if mask_parts:
        for part in mask_parts:
            mask *= (im_segment != part)

    # rgba -> rgb
    if im_color.shape[-1] == 4:
        im_color = im_color[:, :, :3]

    im_color *= np.expand_dims(mask, axis=-1)  # masking
    im_color = o3d.geometry.Image(im_color.data)

    im_depth *= mask  # masking
    im_depth = o3d.geometry.Image(im_depth.data)

    im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_color, im_depth, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=im_rgbd, intrinsic=intrinsic, extrinsic=extrinsic)

    # for uv map
    im_range = np.array(range(height * width),
                        dtype=np.uint32).reshape(height, width)
    u = (im_range % width).astype(np.uint16)
    v = (im_range // width).astype(np.uint16)
    im_u = o3d.geometry.Image(u.data)
    im_v = o3d.geometry.Image(v.data)

    im_ud = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_u, im_depth, convert_rgb_to_intensity=True)
    im_vd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_v, im_depth, convert_rgb_to_intensity=True)
    pcd_ud = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=im_ud, intrinsic=intrinsic, extrinsic=extrinsic)
    pcd_vd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=im_vd, intrinsic=intrinsic, extrinsic=extrinsic)

    pcd_u = np.asarray(pcd_ud.colors)[:, 0]
    pcd_v = np.asarray(pcd_vd.colors)[:, 0]
    pcd_uv_ = np.stack([pcd_u, pcd_v, np.zeros_like(pcd_u)],
                       axis=-1) / np.array([width, height, 1])
    pcd_uv_[:, 1] = pcd_uv_[:, 1]
    # normals는 float type을 허용하므로 잠시 여기에 보관해둔다...
    pcd.normals = o3d.utility.Vector3dVector(pcd_uv_)

    # masking
    pcd_points, pcd_colors, pcd_uv_ = np.asarray(
        pcd.points), np.asarray(pcd.colors), np.asarray(pcd.normals)
    non_black = np.where(np.sum(pcd_colors, axis=-1) != 0)
    pcd_points, pcd_colors, pcd_uv_ = pcd_points[non_black], pcd_colors[non_black], pcd_uv_[
        non_black]
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd.normals = o3d.utility.Vector3dVector(pcd_uv_)

    # dbscan clustering
    labels = np.array(pcd.cluster_dbscan(
        eps=0.04, min_points=30, print_progress=True))
    pcd_points = np.asarray(pcd.points)[labels == 0]
    pcd_colors = np.asarray(pcd.colors)[labels == 0]
    pcd_uv_ = np.asarray(pcd.normals)[labels == 0]

    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # estimate normal
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_towards_camera_location()

    # uv
    pcd_uv = pcd_uv_[:, :2]

    return pcd, pcd_uv


if __name__ == '__main__':

    DATA_DIR = "/home/gu/workspace/Gu/CapstoneDesign/input"

    for id in range(8):

        # inputs
        seg_pth = osp.join(
            DATA_DIR, f"images/segmented/color_to_depth{id}.png")
        color_pth = osp.join(DATA_DIR, f"images/color/color_to_depth{id}.png")
        u_pth = osp.join(DATA_DIR, f"images/texture/color_to_depth{id}_u.png")
        v_pth = osp.join(DATA_DIR, f"images/texture/color_to_depth{id}_v.png")
        depth_pth = osp.join(DATA_DIR, f"images/depth/depth{id}.png")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 576, 502.724945, 502.854401, 323.970764, 326.964050)
        extrinsic = np.identity(4)

        output_pth = osp.join(DATA_DIR, f"pcds/pcd_full{id}.ply")
        output_u_pth = osp.join(DATA_DIR, f"pcds/texture/pcd_full{id}_u.ply")
        output_v_pth = osp.join(DATA_DIR, f"pcds/texture/pcd_full{id}_v.ply")

        seg = cv2.imread(seg_pth, cv2.IMREAD_UNCHANGED)
        mask = (seg != 0)

        color = o3d.io.read_image(color_pth)
        color_data = np.asarray(color)
        color_data *= np.expand_dims(mask, axis=-1)
        # if rgba: rgba -> rgb
        if color_data.shape[-1] == 4:
            # o3d agrb 순으로 읽음. 현재 배열은 rgba. 따라서 rgba 에서 rgrb
            color_data[:, :, 1:] = np.concatenate(
                [color_data[:, :, 1:2], color_data[:, :, 0:1], color_data[:, :, 2:3]], axis=-1)

        im_range = np.array(range(
            color_data.shape[0] * color_data.shape[1]), dtype=np.uint32).reshape(*color_data.shape[:2])
        u = (im_range % color_data.shape[1]).astype(np.uint16)
        v = (im_range // color_data.shape[1]).astype(np.uint16)
        cv2.imwrite(u_pth, u)
        cv2.imwrite(v_pth, v)

        u = o3d.io.read_image(u_pth)
        v = o3d.io.read_image(v_pth)

        depth = o3d.io.read_image(depth_pth)
        depth_data = np.asarray(depth)
        depth_data *= mask

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        ud = o3d.geometry.RGBDImage.create_from_color_and_depth(
            u, depth, convert_rgb_to_intensity=True)
        vd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            v, depth, convert_rgb_to_intensity=True)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd, intrinsic=intrinsic, extrinsic=extrinsic)
        u_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=ud, intrinsic=intrinsic, extrinsic=extrinsic)
        u_data = np.asarray(u_pcd.colors)
        u_data /= color_data.shape[1]  # 0 ~ 1로 맵핑
        v_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=vd, intrinsic=intrinsic, extrinsic=extrinsic)
        v_data = np.asarray(v_pcd.colors)
        v_data /= color_data.shape[0]  # 0 ~ 1로 맵핑

        for _pcd in [pcd, u_pcd, v_pcd]:
            # delete masked black color
            pcd_points = np.asarray(_pcd.points)
            pcd_colors = np.asarray(_pcd.colors)
            non_black = np.where(np.sum(pcd_colors, axis=-1) != 0)
            pcd_points = pcd_points[non_black]
            pcd_colors = pcd_colors[non_black]
            _pcd.points = o3d.utility.Vector3dVector(pcd_points)
            _pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

            # dbscan clustering
            # import matplotlib.pyplot as plt
            labels = np.array(_pcd.cluster_dbscan(
                eps=0.04, min_points=40, print_progress=True))
            pcd_points = np.asarray(_pcd.points)[labels == 0]
            pcd_colors = np.asarray(_pcd.colors)[labels == 0]
            _pcd.points = o3d.utility.Vector3dVector(pcd_points)
            _pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

            # estimate normal
            _pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            _pcd.orient_normals_towards_camera_location()

            # o3d.visualization.draw_geometries([_pcd], width=600, height=600)

        u_pcd.normals = u_pcd.colors
        v_pcd.normals = v_pcd.colors

        o3d.io.write_point_cloud(output_pth, pcd, write_ascii=True)
        o3d.io.write_point_cloud(output_u_pth, u_pcd, write_ascii=True)
        o3d.io.write_point_cloud(output_v_pth, v_pcd, write_ascii=True)
