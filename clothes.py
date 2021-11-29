# basics
import numpy as np
import chumpy as ch
from PIL import Image
from scipy.spatial import cKDTree
import trimesh
import open3d as o3d
from copy import deepcopy

from utils import *

# SMPL
#from smpl_webuser.verts import verts_decorated
from smpl.verts import verts_decorated

from playbvh.smplclothbvhplayer import playBVH, openBVH, getBVHFrame


def create_body_texture(model, pcd_list, uv_list, img_list, init):

    betas = ch.zeros(10)
    init_pose = ch.zeros(72)

    sv = verts_decorated(
        scale=ch.ones(1),
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :10],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    if init:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]

    body_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(deepcopy(sv.r)), o3d.utility.Vector3iVector(deepcopy(sv.f)))
    mesh_uv, face_label = reconstruct_mesh_uv(sv.r, sv.f, pcd_list, uv_list)
    body_mesh.triangle_material_ids = o3d.utility.IntVector(
        list(face_label))
    body_mesh.textures = [o3d.geometry.Image(img) for img in img_list]
    body_mesh.triangle_uvs = o3d.utility.Vector2dVector(mesh_uv)

    o3d.io.write_triangle_mesh("./body_mesh.obj", body_mesh, write_ascii=True, compressed=False,
                               write_vertex_normals=True, write_vertex_colors=True,
                               write_triangle_uvs=True, print_progress=False)

    return body_mesh


def make_avatar(model,
                clothes,
                init=None,
                viz=True):

    betas = ch.zeros(10)
    init_pose = ch.zeros(72)

    sv = verts_decorated(
        scale=ch.ones(1),
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :10],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    if init:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]

    # make clothes mesh
    sv_tree = cKDTree(sv[:])
    axis_x, axis_y, axis_z = setup_vertex_local_coord(
        sv.f, sv.r[:])  # (6890, 3)

    cps_t2m = sv_tree.query(clothes.vertices, k=1, p=2, n_jobs=-1)[1]

    displacement = clothes.vertices[:] - sv[cps_t2m]
    coeff_x = np.einsum('ij,ij->i', axis_x[cps_t2m], displacement)
    coeff_y = np.einsum('ij,ij->i', axis_y[cps_t2m], displacement)
    coeff_z = np.einsum('ij,ij->i', axis_z[cps_t2m], displacement)
    coeff_z[coeff_z < 0.01] = 0.01  # 메쉬가 모델 내부로 파고드는 것 방지

    coeff = np.stack([coeff_x, coeff_y, coeff_z], axis=-1)

    return cps_t2m, coeff


def display(model, clothes, cps_t2m, coeff, body_with_texture, init):

    betas = ch.zeros(10)
    init_pose = ch.zeros(72)

    sv = verts_decorated(
        scale=ch.ones(1),
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :10],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    if init:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]

    def update_target_pcd():
        # FIXME GPU
        axis = np.stack(setup_vertex_local_coord(sv.f, sv.r[:]), axis=-1)
        clothes_vertices[:, :] = sv[cps_t2m] + \
            np.einsum('ij,ikj->ik', coeff, axis[cps_t2m])

    input("start?")

    total_frame = openBVH("./playbvh/gangnam.bvh")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(body_with_texture)
    vis.add_geometry(clothes)

    body_vertices = np.asarray(body_with_texture.vertices)
    clothes_vertices = np.asarray(clothes.vertices)

    for i in range(total_frame):
        sv.pose[:] = getBVHFrame(i)
        body_vertices[:, :] = sv.r
        update_target_pcd()

        vis.update_geometry(body_with_texture)
        vis.update_geometry(clothes)
        vis.poll_events()
        vis.update_renderer()
        if True:
            vis.capture_screen_image("./results/temp_%04d.jpg" % i)

    vis.destroy_window()


def display_physics(model, clothes, cps_t2m, coeff, body_with_texture, init):

    betas = ch.zeros(10)
    init_pose = ch.zeros(72)

    sv = verts_decorated(
        scale=ch.ones(1),
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :10],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    if init:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]

    def update_target_pcd():
        # FIXME GPU
        axis = np.stack(setup_vertex_local_coord(sv.f, sv.r[:]), axis=-1)
        clothes_vertices[:, :] = sv[cps_t2m] + \
            np.einsum('ij,ikj->ik', coeff, axis[cps_t2m])

    input("start?")

    total_frame = openBVH("./playbvh/gangnam.bvh")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(body_with_texture)
    vis.add_geometry(clothes)

    body_vertices = np.asarray(body_with_texture.vertices)
    clothes_vertices = np.asarray(clothes.vertices)

    for i in range(total_frame):
        print(i)
        sv.pose[:] = getBVHFrame(i)
        body_vertices[:, :] = sv.r
        update_target_pcd()

        vis.update_geometry(body_with_texture)
        vis.update_geometry(clothes)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
