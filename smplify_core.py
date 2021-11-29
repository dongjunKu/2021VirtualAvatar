"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

About this Script:
============
This is a demo version of the algorithm implemented in the paper,
which fits the SMPL body model to the image given the joint detections.
The code is organized to be run on the LSP dataset.
See README to see how to download images and the detected joints.


==============
Modification : 9

For Windows Enviroment, we use only Camera module from OpenDR and replay the Rendering with PyRedner


dependent 
===============

numpy    (pip install numpy) 
opencv   (pip install opencv-python)
chumpy   (pip install chumpy)
smpl     (modified by seoultech)
opendr   (modified by SeoulTech) 
smplify  (named for package) 


"""
import os
import logging
# import cPickle as pickle   # Python 2
import _pickle as pickle   # Python 3
from time import time

# basics
import numpy as np
import chumpy as ch
from scipy.spatial import cKDTree

from utils import *

# SMPL
from smpl.lbs import global_rigid_transformation
#from smpl_webuser.verts import verts_decorated
from smpl.verts import verts_decorated

# SMPLIFY
from smplify.robustifiers import GMOf
#from lib.sphere_collisions import SphereCollisions
from smplify.sphere_collisions import SphereCollisions
#from lib.max_mixture_prior import MaxMixtureCompletePrior
from smplify.max_mixture_prior import MaxMixtureCompletePrior

# Trimesh
import trimesh

_LOGGER = logging.getLogger(__name__)

# Mapping from Kinnect joints to SMPL joints.
# kinnect                   smpl
# 24    Right ankle         8
# 23    Right knee          5
# 22    Right hip           2
# 18    Left hip            1
# 19    Left knee           4
# 20    Left ankle          7
# 14    Right wrist         21
# 13    Right elbow         19
# 12    Right shoulder      17
# 5     Left shoulder       16
# 6     Left elbow          18
# 7     Left wrist          20

# 15    Right hand          21
# 8     Left hand           22
# 25    Right foot          11
# 21    Left foot           10
# 3     Neck                12
# 26    Head                15

# --------------------Core optimization --------------------


def optimize_j3d(j3d,
                 model,
                 prior,
                 n_betas=10,
                 target=None,
                 regs=None,
                 conf=None,
                 viz=False,
                 cam_pose=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param j3d: 14x3 array of CNN joints
    :param model: SMPL model
    :param prior: mixture of gaussians pose prior
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: 18D vector storing the confidence values from the CNN
    :param viz: boolean, if True enables visualization during optimization
    :returns: the optimized model
    """

    t0 = time()
    # define the mapping Kinnect joints -> SMPL joints
    # cids are joints ids for Kinnect:
    cids = [24, 23, 22, 18, 19, 20, 14, 13, 12,
            5, 6, 7, 15, 8, 25, 21, 3, 26]

    # joint ids for SMPL
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17,
                16, 18, 20, 21, 22, 11, 10, 12, 15]

    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and Kinnect is significantly different so set
    base_weights = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1], dtype=np.float64)
    # [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # initialize the pose by using the optimized body orientation and the
    # pose prior
    init_pose = np.hstack(([0, 0, 1], prior.weights.dot(prior.means)))

    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    sv = verts_decorated(
        scale=ch.ones(1),
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                       for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + \
        model.J_regressor.dot(model.v_template.r)

    # get joint positions as a function of model pose, betas, scale and trans
    (_, A_global) = global_rigid_transformation(
        sv.pose, J_onbetas, model.kintree_table, xp=ch)
    Jtr = sv.scale * ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

    # update the weights using confidence values
    weights = base_weights * conf[cids] if conf is not None else base_weights

    # obj1. data term: distance between observed and estimated joints in 2D
    ###########################

    def obj_j3d(w, sigma):
        return (w * weights.reshape((-1, 1)) * GMOf((j3d[cids] - Jtr[smpl_ids]), sigma))

    # obj2: mixture of gaussians pose prior
    ###########################

    def pprior(w): return w * prior(sv.pose)
    # obj3: joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    def my_exp(x): return alpha * ch.exp(x)

    def obj_angle(w): return w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
        58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

    if viz is True:
        import smpl_pyrender
        import pyrender

        if cam_pose is None:
            cam_pose = np.array([
                [1.0,  0.0,  0.0,  0.0],   # zero-rotation
                [0.0, -1.0,  0.0,  0.0],   #
                [0.0,  0.0, -1.0, -1.0],   # (0, 0, 2.0) translation
                [0.0,  0.0,  0.0,  1.0]
            ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)

        if type(target) == trimesh.base.Trimesh:
            target_pyrender = pyrender.Mesh.from_trimesh(target, smooth=False)
        if type(target) == trimesh.points.PointCloud:
            target_pyrender = pyrender.Mesh.from_points(
                target.vertices, target.colors)
            print(target.vertices)
            print(target.colors)

        smpl_pyrender.smplMeshNode = None

        # show joints
        sm = trimesh.creation.uv_sphere(radius=0.02)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0]
        tfs = np.tile(np.eye(4), (len(j3d[cids]), 1, 1))
        tfs[:, :3, 3] = j3d[cids]
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        def on_step(_):
            smpl_pyrender.updateSMPL(scene, viewer, sv[:], sv.f)

            viewer.render_lock.acquire()
            scene.add(m)
            viewer.render_lock.release()

            if target is not None:
                viewer.render_lock.acquire()
                scene.add(target_pyrender)
                viewer.render_lock.release()
    else:
        on_step = None

    # obj5: interpenentration
    ###########################
    if regs is not None:
        # interpenetration term
        sp = SphereCollisions(
            pose=sv.pose, betas=sv.betas, model=model, regs=regs)
        sp.no_hands = True

    #############################################
    # 5. optimize
    #############################################
    # weight configuration used in the paper, with joints + confidence values from the CNN
    # (all the weights used in the code were obtained via grid search, see the paper for more details)
    # the first list contains the weights for the pose priors,
    # the second list contains the weights for the shape prior
    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78, 0.75],
                      [1e2, 5 * 1e1, 1e1, .5 * 1e1, .5 * 1e1])

    # run the optimization in 4 stages, progressively decreasing the
    # weights for the priors
    for stage, (w, wbetas) in enumerate(opt_weights):
        _LOGGER.info('stage %01d', stage)
        objs = {}
        objs['j3d'] = 10 * obj_j3d(1., 100)
        objs['pose'] = pprior(w)
        if stage != 4:
            objs['pose_exp'] = obj_angle(0.317 * w)
        objs['betas'] = wbetas * betas
        if regs is not None:
            objs['sph_coll'] = 1e3 * sp

        if stage == 0:
            t, b, p = ch.minimize(
                objs,                   # objective functions
                x0=[sv.trans, sv.betas, sv.pose],  # free-variables
                method='dogleg',
                callback=on_step,
                options={'maxiter': 100, 'e_3': .0001, 'disp': 0})
            s = sv.scale.r
        else:
            s, t, b, p = ch.minimize(
                objs,                   # objective functions
                x0=[sv.scale, sv.trans, sv.betas, sv.pose],  # free-variables
                method='dogleg',
                callback=on_step,
                options={'maxiter': 100, 'e_3': .0001, 'disp': 0})

        # checking optimized pose and shape
        print('   scale :', s)
        print('   trans :', t)
        print('   betas;', b)
        print('   pose :', p.reshape((-1, 3))[0])

    t1 = time()
    _LOGGER.info('elapsed %.05f', (t1 - t0))

    if viz is True:
        viewer.close_external()

    return sv

# --------------------Core optimization --------------------


def optimize_icp(target,
                 model,
                 prior,
                 n_betas=10,
                 regs=None,
                 init=None,
                 maxiter=30,
                 viz=False,
                 save_dir=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param target: trimesh mesh or pointcloud
    :param model: SMPL model
    :param prior: mixture of gaussians pose prior
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param init: initial parameter dictionary
    :param maxiter: maxiter
    :param viz: boolean, if True enables visualization during optimization
    :returns: the optimized model
    """

    t0 = time()

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # initialize the pose by using the optimized body orientation and the
    # pose prior
    init_pose = np.hstack(([0, 0, 0], prior.weights.dot(prior.means)))

    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    sv = verts_decorated(
        # GU; result = scale * result + tr; result.scale = scale (verts_decorated, verts.py)
        scale=ch.ones(1),
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    target_tree = cKDTree(target.vertices)
    target_normal = estimate_normals(
        target.vertices, radius=40, orient=[0, 0, -10])  # FIXME

    def pprior(w): return w * prior(sv.pose)
    # obj3: joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    def my_exp(x): return alpha * ch.exp(x)

    if regs is not None:
        # interpenetration term
        sp = SphereCollisions(
            pose=sv.pose, betas=sv.betas, model=model, regs=regs)
        sp.no_hands = True

    #############################################
    # 5. optimize
    #############################################

    if viz is True:

        import smpl_pyrender
        import pyrender

        cam_pose = np.array([
            [1.0,  0.0,  0.0,  0.0],   # zero-rotation
            [0.0,  -1.0, 0.0, -0.5],   #
            [0.0,  0.0, -1.0,  0.0],   # (0, 0, 2.0) translation
            [0.0,  0.0,  0.0,  1.0]
        ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)

        if type(target) == trimesh.base.Trimesh:
            target_pyrender = pyrender.Mesh.from_trimesh(target, smooth=False)
        if type(target) == trimesh.points.PointCloud:
            target_pyrender = pyrender.Mesh.from_points(
                target.vertices, target.colors)

        smpl_pyrender.smplMeshNode = None

        def on_step(_):
            smpl_pyrender.updateSMPL(scene, viewer, sv[:], sv.f)

            viewer.render_lock.acquire()
            scene.add(target_pyrender)
            viewer.render_lock.release()

    else:
        on_step = None

    if init is not None:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]

    for i in range(maxiter):

        _LOGGER.info('Iteration %01d', i)

        sv_tree = cKDTree(sv[:])
        _, sv_normal = calc_normal_vectors(sv[:], sv.f)

        cps_m2t = target_tree.query(sv[:], k=1, p=2)[1]  # (dist, idx)
        cps_t2m = sv_tree.query(target.vertices, k=1, p=2, n_jobs=-1)[1]

        def dist_m2t(thres=0):
            # 노말벡터가 비슷한 방향인 경우
            mask = np.sum(
                sv_normal[:] * target_normal[cps_m2t], axis=-1) > thres
            distv = sv[:] - target.vertices[cps_m2t]
            dist = ch.sum(distv**2, axis=-1)**0.5
            return dist[mask]**2 + ch.abs(dist[mask])

        def dist_t2m(thres=0):
            # 노말벡터가 비슷한 방향인 경우
            mask = np.sum(sv_normal[cps_t2m] *
                          target_normal[:], axis=-1) > thres
            distv = sv[cps_t2m] - target.vertices[:]
            dist = ch.sum(distv**2, axis=-1)**0.5
            return dist[mask]**2 + ch.abs(dist[mask])

        def penalty_m2t(w, thres=0):
            distv = sv[:] - target.vertices[cps_m2t]
            mask = np.sum(target_normal[cps_m2t] * distv.r, axis=-1) > thres
            dist = w * ch.sum(distv**2, axis=-1)**0.5
            # 노말벡터 기준으로 모델이 타겟보다 앞에 있을 경우 엄청난 페널티
            return ch.exp(dist[mask]) / w

        def penalty_t2m(w, thres=0):
            distv = sv[cps_t2m] - target.vertices[:]
            mask = np.sum(target_normal[:] * distv.r, axis=-1) > thres
            dist = w * ch.sum(distv**2, axis=-1)**0.5
            # 노말벡터 기준으로 모델이 타겟보다 앞에 있을 경우 엄청난 페널티
            return ch.exp(dist[mask]) / w

        w = 0.75 * 1e-1  # 0.75
        wbetas = 0.5 * 1e0  # 5.0

        objs = {}
        objs['dist_m2t'] = dist_m2t() / 2
        objs['dist_t2m'] = dist_t2m() / 2
        objs['penalty_m2t'] = penalty_m2t(10) / 2
        objs['penalty_t2m'] = penalty_t2m(10) / 2
        objs['pose'] = pprior(w)
        # objs['pose_exp'] = obj_angle(0.317 * w)
        objs['betas'] = wbetas * betas
        if regs is not None:
            objs['sph_coll'] = 1e-1 * sp

        s, t, b, p = ch.minimize(
            objs,                   # objective functions
            x0=[sv.scale, sv.trans, sv.betas, sv.pose],  # free-variables
            method='dogleg',
            callback=on_step,
            options={'maxiter': 1, 'e_3': .01, 'disp': 0})

        # checking optimized pose and shape
        print('   scale :', s)
        print('   trans :', t)
        print('   betas;', b)
        print('   pose :', p.reshape((-1, 3))[0])

        params = {'scale': sv.scale.r,
                  'trans': sv.trans.r,
                  'betas': sv.betas.r,
                  'pose': sv.pose.r}

        if save_dir is not None:
            with open(os.path.join(save_dir, 'output' + str(i) + '.pkl'), 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(params, outf)
            with open(os.path.join(save_dir, 'output_icp.pkl'), 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(params, outf)

    if viz is True:
        viewer.close_external()

    t1 = time()
    _LOGGER.info('elapsed %.05f', (t1 - t0))

    return sv


def run_single_fit(target,
                   j3d,
                   model,
                   regs=None,
                   n_betas=10,
                   viz=False,  # optimize_on_joints
                   out_dir=None):
    """Run the fit for one specific image.
    :param target: trimesh pcd
    :param j3d: 3-D joints coordinate, N * 3 array
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """

    ###################################
    # 1. prior setting
    ###################################
    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))
    print(' gmm: w=', prior.weights.r)   # pose GMM statistics chumpy
    print(' gmm: m=', prior.means.shape)  # pose GMM statistics numpy
    print(" inital pose:", init_pose.reshape((-1, 3)))  # numpy

    ###################################
    # 3. fit
    ####################################

    choice = 'n'
    if os.path.isfile(os.path.join(out_dir, 'output_j3d.pkl')):
        choice = input("skip j3d? [y/n] ")

    if choice == 'n' or choice == 'N':
        sv = optimize_j3d(
            j3d,
            model,
            prior,  # priors
            n_betas=n_betas,  # shape params size
            target=target,
            viz=viz,     # visualizing or not
            regs=regs
        )

        params = {'scale': sv.scale.r,
                  'trans': sv.trans.r,
                  'betas': sv.betas.r,
                  'pose': sv.pose.r}

        with open(os.path.join(out_dir, 'output_j3d.pkl'), 'wb') as outf:  # 'wb' for python 3?
            pickle.dump(params, outf)
    else:
        with open(os.path.join(out_dir, 'output_j3d.pkl'), 'rb') as p:
            params = pickle.load(p)

    choice = 'n'
    if os.path.isfile(os.path.join(out_dir, 'output_icp.pkl')):
        choice = input("skip icp? [y/n] ")

    if choice == 'n' or choice == 'N':
        sv = optimize_icp(
            target,
            model,
            prior,  # priors
            n_betas=n_betas,  # shape params size
            maxiter=30,
            regs=regs,
            init=params,
            viz=viz      # visualizing or not
        )

        # 5. return resultant fit parameters  (pose, shape) is what we want but camera needed
        # save all Camera parameters and SMPL paramters
        params = {'scale': sv.scale.r,
                  'trans': sv.trans.r,
                  'betas': sv.betas.r,
                  'pose': sv.pose.r}

        with open(os.path.join(out_dir, 'output_icp.pkl'), 'wb') as outf:  # 'wb' for python 3?
            pickle.dump(params, outf)

    else:
        with open(os.path.join(out_dir, 'output_icp.pkl'), 'rb') as p:
            params = pickle.load(p)

    return params
