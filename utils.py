# -*- coding: utf-8 -*-
# @Time    : 10/21/2022 11:45 AM
# @Author  : Mingdian Liu
# @Email   : mingdian@iastate.edu lmdvigor@gmail.com
# @File    : utils.py
# @Software: PyCharm
import numpy as np
from scipy.spatial.transform import Rotation as R


def gen_mask(vol_rec, x_min, x_max, magnification, projs_cols, det_spacing_x, src_x_det_crd, rot_x_det_crd):
    vol_num = vol_rec.shape[1]
    effective_det_range = projs_cols * det_spacing_x / 2 - (abs(src_x_det_crd) + abs(rot_x_det_crd))*4
    effective_det_range = effective_det_range - 5 # minus 1 to remove the internal edge ring
    effective_recon_range = effective_det_range / magnification
    effective_recon_pixel = effective_recon_range/x_max * vol_num / 2
    mask = np.ones((vol_num, vol_num))

    for i in range(vol_num):
        for j in range(vol_num):
            if pow(i +1 - vol_num/2,2) + pow(j +1 - vol_num/2,2) >= pow(effective_recon_pixel,2):
                mask[i,j] = 0

    return mask



def remove_edge_noise(vol_rec, mask):
    vol_num = vol_rec.shape[0]
    min_value = np.min(vol_rec)
    # vol_rec = np.array([np.multiply(vol_rec[i], mask) for i in range(vol_num)], dtype=np.float32)
    vol_rec = np.array([np.multiply(vol_rec[i], mask) + (1-mask)*min_value for i in range(vol_num)], dtype=np.float32)

    return vol_rec

def remove_edge_noise_2d(vol_rec, mask):
    vol_num = vol_rec.shape[0]
    min_value = np.min(vol_rec)
    # vol_rec = np.array([np.multiply(vol_rec[i], mask) for i in range(vol_num)], dtype=np.float32)
    vol_rec = np.array(np.multiply(vol_rec, mask) + (1-mask)*min_value, dtype=np.float32)

    return vol_rec



def cal_vecs_3d(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles,
             det_spacing_x, det_spacing_y):

    print([src_x_det_crd, src_y_det_crd, src_z_det_crd])
    print([rot_x_det_crd, rot_y_det_crd, rot_z_det_crd])
    src_pos = [-rot_x_det_crd, -(src_z_det_crd - rot_z_det_crd),
               -src_y_det_crd * (src_z_det_crd - rot_z_det_crd) / src_z_det_crd]
    det_pos = [-src_x_det_crd-rot_x_det_crd, rot_z_det_crd, src_y_det_crd * rot_z_det_crd / src_z_det_crd]
    print(src_pos)
    print(det_pos)


    # # legacy code
    # flip_src_pos_z = False    # CStalk
    #
    # if flip_src_pos_z:
    #     src_pos[2] = -src_pos[2]
    #     det_pos[2] = -det_pos[2]

    alpha = 0
    theta = 0
    gmma = 0

    u = [det_spacing_x, 0, 0]
    v = [0, 0, det_spacing_y]

    r = R.from_rotvec([alpha, 0, 0])
    u = r.apply(u)
    v = r.apply(v)
    r = R.from_rotvec([0, theta, 0])
    u = r.apply(u)
    v = r.apply(v)
    r = R.from_rotvec([0, 0, gmma])
    u = r.apply(u)
    v = r.apply(v)

    vectors = np.zeros((len(angles), 12))

    for i in range(len(angles)):
        # source
        vectors[i, 0] = np.cos(angles[i]) * src_pos[0] - np.sin(angles[i]) * src_pos[1]
        vectors[i, 1] = np.sin(angles[i]) * src_pos[0] + np.cos(angles[i]) * src_pos[1]
        vectors[i, 2] = src_pos[2]

        # center of detector
        vectors[i, 3] = np.cos(angles[i]) * det_pos[0] - np.sin(angles[i]) * det_pos[1]
        vectors[i, 4] = np.sin(angles[i]) * det_pos[0] + np.cos(angles[i]) * det_pos[1]
        vectors[i, 5] = det_pos[2]

        # vector from detector pixel (0,0) to (0,1), U
        vectors[i, 6] = np.cos(angles[i]) * u[0]
        vectors[i, 7] = np.sin(angles[i]) * u[0]
        vectors[i, 8] = u[2]

        # vector from detector pixel (0,0) to (1,0), V
        vectors[i, 9] = np.cos(angles[i]) * v[0]
        vectors[i, 10] = np.sin(angles[i]) * v[0]
        vectors[i, 11] = v[2]

    return vectors


def cal_vecs_2d(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles,
             det_spacing_x, det_spacing_y):
    # dist_orgin_detector, dist_origin_detector
    src_pos = [-rot_x_det_crd, -(src_z_det_crd - rot_z_det_crd),
               -src_y_det_crd * (src_z_det_crd - rot_z_det_crd) / src_z_det_crd]
    det_pos = [-src_x_det_crd-rot_x_det_crd, rot_z_det_crd, src_y_det_crd * rot_z_det_crd / src_z_det_crd]

    print(src_pos)
    print(det_pos)

    # # legacy code
    # flip_src_pos_z = False    # CStalk
    #
    # if flip_src_pos_z:
    #     src_pos[2] = -src_pos[2]
    #     det_pos[2] = -det_pos[2]

    alpha = 0
    theta = 0
    gmma = 0

    u = [det_spacing_x, 0, 0]
    v = [0, 0, det_spacing_y]

    # r = R.from_rotvec([alpha, 0, 0])
    # u = r.apply(u)
    # v = r.apply(v)
    # r = R.from_rotvec([0, theta, 0])
    # u = r.apply(u)
    # v = r.apply(v)
    # r = R.from_rotvec([0, 0, gmma])
    # u = r.apply(u)
    # v = r.apply(v)

    vectors = np.zeros((len(angles), 6), dtype=np.float32)

    for i in range(len(angles)):
        # source
        vectors[i, 0] = np.cos(angles[i]) * src_pos[0] - np.sin(angles[i]) * src_pos[1]
        vectors[i, 1] = np.sin(angles[i]) * src_pos[0] + np.cos(angles[i]) * src_pos[1]

        # center of detector
        vectors[i, 2] = np.cos(angles[i]) * det_pos[0] - np.sin(angles[i]) * det_pos[1]
        vectors[i, 3] = np.sin(angles[i]) * det_pos[0] + np.cos(angles[i]) * det_pos[1]

        # vector from detector pixel 0 to 1, U
        vectors[i, 4] = np.cos(angles[i]) * u[0]
        vectors[i, 5] = np.sin(angles[i]) * u[0]

    return vectors

