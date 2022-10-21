import numpy as np


def gen_mask(vol_rec, x_min, x_max, magnification, projs_cols, det_spacing_x, src_x_det_crd, rot_x_det_crd):
    _,vol_num,_ = vol_rec.shape
    effective_det_range = projs_cols * det_spacing_x / 2 - (abs(src_x_det_crd) + abs(rot_x_det_crd))*4
    effective_det_range = effective_det_range - 1 # minus 1 to remove the internal edge ring
    effective_recon_range = effective_det_range / magnification
    effective_recon_pixel = effective_recon_range/x_max * vol_num / 2
    mask = np.ones((vol_num, vol_num))

    for i in range(vol_num):
        for j in range(vol_num):
            if pow(i +1 - vol_num/2,2) + pow(j +1 - vol_num/2,2) >= pow(effective_recon_pixel,2):
                mask[i,j] = 0

    return mask



def remove_edge_noise(vol_rec, mask):
    vol_num,_,_ = vol_rec.shape
    min_value = np.min(vol_rec)
    vol_rec = np.array([np.multiply(vol_rec[i], mask) for i in range(vol_num)], dtype=np.float32)
    # vol_rec = np.array([np.multiply(vol_rec[i], mask) + (1-mask)*min_value for i in range(vol_num)], dtype=np.float32)

    return vol_rec



