import os
import open3d as o3d
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.stats import mode


def normalize_v3(arr):
    # Normalize a numpy array of 3 component vectors shape=(n,3)
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def find2ndaxis(faces, v_normal, v_ref):

    debug = True
    n_vertex = v_ref.shape[0]

    # 1. first find the smallest-indexed neighbor vertex
    # @TODO any way to speed up this step?
    ngbr_vertex = n_vertex*np.ones(v_ref.shape[0], dtype=np.int64)
    for fidx, fv in enumerate(faces):
        v0, v1, v2 = fv
        if ngbr_vertex[v0] > min(v1, v2):
            ngbr_vertex[v0] = min(v1, v2)  # short-form by Matiur
        if ngbr_vertex[v1] > min(v0, v2):
            ngbr_vertex[v1] = min(v0, v2)
        if ngbr_vertex[v2] > min(v1, v0):
            ngbr_vertex[v2] = min(v1, v0)

    # check results
    if debug:
        for idx in range(n_vertex):
            if ngbr_vertex[idx] >= n_vertex:
                print('This vertex has no neighbor hood:',  idx)

    # 2. compute the tangential vector component
    #    vec -   dot(normal, vec) * normal
    from numpy import dot
    from numpy.linalg import norm

    vec1 = v_ref[ngbr_vertex] - v_ref       # get the edge vector
    # print('shape comp: ',  v_normal.shape, vec1.shape)
    coefs = np.sum(v_normal*vec1, axis=1)  # coef = dot(v_normal, vec1)
    vec2 = vec1 - coefs[:, None]*v_normal  # remove the normal components

    axis = normalize_v3(vec2)

    return axis


def setup_vertex_local_coord(faces, vertices):

    # 1.1 normal vectors (1st axis) at each vertex
    _, axis_z = calc_normal_vectors(vertices, faces)
    # 1.2 get 2nd axis
    axis_x = find2ndaxis(faces, axis_z, vertices)
    # 1.3 get 3rd axis
    # matiur contribution. np.cross support row-vectorization
    axis_y = np.cross(axis_z[:, :], axis_x[:, :])

    return axis_x, axis_y, axis_z


def estimate_normals(vertices, radius, orient=None, kmean=0):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    if orient:
        pcd.orient_normals_to_align_with_direction(orient)  # FIXME

    normals = np.asarray(pcd.normals)

    if kmean > 0:
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(vertices.astype(
            np.float32), kmean, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        displacement = vertices - centers[labels.flatten()]
        inner = np.sum(normals * displacement, axis=1)
        normals[inner < 0] *= -1

    return normals


def calc_normal_vectors(vertices, faces):

    # Create a zero array with the same type and shape as our vertices i.e., per vertex normal
    _norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    #n = norm(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    _norm[faces[:, 0]] += n
    _norm[faces[:, 1]] += n
    _norm[faces[:, 2]] += n
    normalize_v3(_norm)
    # norm(_norm)

    return n, _norm


def create_poisson_mesh(pcd, voxel_size=0.02, out_dir=None, viz=False):

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_neighbor_list = [pcd_tree.search_radius_vector_3d(
        mesh.vertices[idx], 2*voxel_size)[0] for idx in range(len(np.asarray(mesh.vertices)))]
    num_neighbor_list = np.array(num_neighbor_list)
    vertices_mask = num_neighbor_list < np.median(num_neighbor_list) / 4  # 2
    mesh.remove_vertices_by_mask(vertices_mask)

    if viz:
        print(len(np.asarray(mesh.vertices)))
        o3d.visualization.draw_geometries([mesh])

    if out_dir:
        o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.obj"), mesh)

    return mesh


def reconstruct_mesh_uv(target_vertices, target_faces, pcd_list, uv_list):

    points_list = [np.asarray(pcd.points) for pcd in pcd_list]
    points = np.concatenate(points_list, axis=0)
    labels = np.concatenate([i * np.ones(len(points), dtype=int)
                             for i, points in enumerate(points_list)])
    tree = cKDTree(points)
    indices = tree.query(target_vertices, k=1, p=2, n_jobs=-1)[1]

    labels = np.stack([labels[idx] for idx in indices], axis=0)

    uvs = []
    for uv, pcd in zip(uv_list, pcd_list):
        points = np.asarray(pcd.points)
        tree = cKDTree(points)
        indices = tree.query(target_vertices, k=1, p=2, n_jobs=-1)[1]
        new_uv = uv[indices]
        uvs.append(new_uv)
    uvs = np.stack(uvs, axis=0)  # (num_pcd_list, num_target_vertices, 2)

    target_faces = target_faces.reshape(-1)  # (num_face, 3) -> (3 * num_face)
    label_faces = labels[target_faces].reshape(-1, 3)  # (num_face, 3)
    mode_label_faces = mode(label_faces, axis=1)[0]  # mode, (num_face, 1)
    label_faces[:, :] = mode_label_faces[:, :]
    label_faces = label_faces.reshape(-1)
    triangle_uv = uvs[label_faces, target_faces]

    return triangle_uv, mode_label_faces.reshape(-1)


def saveMeshObjFile(objFilePath, mtlfilePath, uvfilePath, vertices, vts, faces):

    #saveObjfile(filename, v_cur.r, smpl_model.f, projected_v_old)
    with open(objFilePath, 'w') as fp:
        fp.write('mtllib {}\n'.format(mtlfilePath))
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for vt in vts:
            fp.write('vt %f %f\n' % (vt[0], vt[1]))
        fp.write('usemtl {}\n'.format("material_0"))
        for f in faces+1:  # Faces are 1-based, not 0-based in obj files
            # polygons for 3D and textture
            fp.write('f %d/%d %d/%d %d/%d\n' %
                     (f[0], f[0], f[1], f[1], f[2], f[2]))

    with open(mtlfilePath, 'w') as fp:
        fp.write('newmtl material_0\n')
        fp.write('# shader_type beckmann\n')
        fp.write('map_Kd {}\n'.format(uvfilePath))
