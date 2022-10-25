# -*- coding: utf-8 -*-
# @Time    : 10/25/2022 10:35 AM
# @Author  : Mingdian Liu
# @Email   : mingdian@iastate.edu lmdvigor@gmail.com
# @File    : center_of_rotation_determination.py
# @Software: PyCharm

"""
Inspection Information:
  - Facility: ASCII_274_PE2923
    - Detector: PE2923_Rm157
      - Resolution [x , y]:  [3888 , 3072]
      - Size [x , y]:        [0.2908 , 0.2298] [0:m:1]
      - Pixel Pitch [x , y]: [7.47942e-05 , 7.48047e-05] [0:m:1]
      - Bit Depth:           16
    - Source: ASCII_157_Feifocus_225.48
      - Position [x , y , z]:            [-0.00104712 , -0.0385992 , 0.985] [0:m:1]
      - Beam CL Orientation [x , y , z]: [0 , 0 , 0] [0:radian:1]
      - Sample rotation vector - X: 1 0 0
      - Sample rotation vector - Y: 0 1 0
      - Sample rotation vector - Z: 0 0 1
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


# # example code for Reading a RAW file:
# input_file = './CStalk/CrnStlk0.raw'
# npimg = np.fromfile(input_file, dtype=np.uint16)
# print(type(npimg))
# print(npimg.shape)
# imageSize = (3888, 3072)
# npimg = npimg.reshape(imageSize)
#
# # print(npimg[1944:1954, 1536:1546])
# # print(np.max(npimg))
# # print(np.min(npimg))
#
# imgplot = plt.imshow(npimg, cmap='gray')
# plt.show()




import numpy as np
import astra
import os
import imageio.v2 as imageio
import time
import utils
import matplotlib.pyplot as plt


#### user defined settings #####################################################

# sub-sampling rate of raw data
angluar_sub_sampling = 1

## configuration of reconstruction space

x_min, x_max = -18.175, 18.175 # [mm] reconstruction volume range
y_min, y_max = -18.175, 18.175 # [mm] reconstruction volume range
z_min, z_max = -18.175, 18.175  # [mm] reconstruction volume range

x_vol_sz, y_vol_sz, z_vol_sz = 1500, 1500, 1500 # volex number for x, y, z. Please make sure x_vol_sz=y_vol_sz
# x_vol_sz, y_vol_sz, z_vol_sz = 3000, 3000, 3000 # volex number for x, y, z. Please make sure x_vol_sz=y_vol_sz


## file path and prefix configuration
# raw data path
data_path = './CStalk/' # the path for input data (.raw)
# reconstruction path
recon_path = './CStalk_reconstruction'
# prefix of raw data
projs_name = 'CrnStlk{}.raw'
# number of projections
n_pro = 360 # CrnStlk

# set True when you already use Matlab script to rotate raw data by 90 degrees
ard_rot = False  # CStalk

# detector configuration
projs_cols = 3888 # the number of column in detector array
projs_rows = 3072 # the number of row in detector array
det_spacing_x = 7.47942e-02  # [mm] the spanning distance between columns
det_spacing_y = 7.48047e-02  # [mm] the spanning distance between rows
det_lim = 16383 # the maximum detected value for each pixel. It is commonly 2^n -1.


# distance of source to detector
dist_source_detector = 985
# distance of source to object
dist_source_origin = 123.125  # [mm]
# distance of object to detector
dist_origin_detector = dist_source_detector - dist_source_origin  # [mm]
magnification = dist_source_detector / dist_source_origin # [mm]

# source position
src_x_det_crd = -1.04712  # [mm]
src_y_det_crd = -38.5992  # [mm] # CSStlk
src_z_det_crd = dist_source_detector # [mm]

# center of rotation offset
offset_start = 0
offset_end = 1.2
offset_slice_num = 32
offset_arr = np.linspace(offset_start, offset_end, num=int(offset_slice_num+2))
offset_arr = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1]
offset_arr = [tmp+0.46329 for tmp in offset_arr]

rot_x_det_crd = 0  # [mm] # CStalk
rot_y_det_crd = 0  # [mm]
rot_z_det_crd = dist_origin_detector # [mm]



#### load data #################################################################

t_start=time.time()
t = time.time()
print('load data', flush=True)


angles = np.linspace(0, 2 * np.pi, num=n_pro, endpoint=False)

# create the numpy array which will receive projection data from raw data
projs = np.zeros((projs_rows, n_pro, projs_cols), dtype=np.float32) # (n_pro, projs_rows, projs_cols)
sinogram = np.zeros((n_pro, projs_cols), dtype=np.float32) # (n_pro, projs_cols)

# load projection data
imageSize = (projs_cols, projs_rows)
if ard_rot:
    imageSize = list(reversed(imageSize))

for i in range(n_pro):
    npimg = np.fromfile(os.path.join(data_path, projs_name.format(i)), dtype=np.uint16)
    npimg = npimg.reshape(imageSize)
    ## check raw image
    # print(type(npimg))
    # plt.imshow(npimg)
    # plt.show()
    if ard_rot:
        npimg = np.rot90(npimg)

    # print(type(npimg))
    # plt.imshow(npimg)
    # plt.show()

    npimg = np.flipud(npimg)
    npimg = np.transpose(npimg)
    # print(type(npimg))
    # plt.imshow(npimg)
    # plt.show()
    sinogram[i, :] = npimg[int(projs_rows/2-1-src_y_det_crd/det_spacing_y)] # (n_pro, projs_cols)


print('plot sinogram')
plt.imshow(sinogram)
plt.show()

print(np.round_(time.time() - t, 3), 'sec elapsed')


### pre-process data ###########################################################

t = time.time()
print('pre-process data', flush=True)
# take the negative log to linearize the data according to the Beer-Lambert law

print(f'projs: {np.max(projs)} {np.min(projs)}')

sinogram /= det_lim*4
np.log(sinogram, out=sinogram)
np.negative(sinogram, out=sinogram)
sinogram = np.ascontiguousarray(sinogram)
print(np.round_(time.time() - t, 3), 'sec elapsed')

print('plot sinogram')
plt.imshow(sinogram)
plt.show()

### compute FDK reconstruction #################################################

t = time.time()
print('compute reconstruction', flush=True)


# numpy array holding the reconstruction volume
vol_geom = astra.create_vol_geom(x_vol_sz, y_vol_sz)
vol_geom['option']['WindowMinX'] = x_min
vol_geom['option']['WindowMaxX'] = x_max
vol_geom['option']['WindowMinY'] = y_min
vol_geom['option']['WindowMaxY'] = y_max

vol_rec = np.zeros([x_vol_sz, y_vol_sz], dtype=np.float32)



angles = np.linspace(0, 2*np.pi, n_pro, False)

for rot_x_det_crd in offset_arr:

    vecs = utils.cal_vecs_2d(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles, det_spacing_x, det_spacing_y)

    # proj_geom = astra.create_proj_geom('fanflat', det_spacing_x, projs_cols, angles, dist_source_origin, dist_origin_detector)
    # print(proj_geom.keys())
    # proj_geom_vec = astra.geom_2vec(proj_geom)
    # print(proj_geom_vec['Vectors'].shape)
    # print(proj_geom_vec['Vectors'])
    # proj_geom['Vectors']=vecs
    # vecs=proj_geom['Vectors']
    proj_geom = astra.create_proj_geom('fanflat_vec', projs_cols, vecs)
    proj_id_new = astra.create_projector('line_fanflat', proj_geom, vol_geom)


    # register both volume and projection geometries and arrays to ASTRA
    vol_id  = astra.data2d.link('-vol', vol_geom, vol_rec)
    proj_id = astra.data2d.link('-sino', proj_geom, sinogram)


    # ## FBP
    # cfg_fdk = astra.astra_dict('FBP_CUDA')
    #
    # cfg_fdk['ProjectionDataId'] = proj_id
    # cfg_fdk['ReconstructionDataId'] = vol_id
    # alg_id = astra.algorithm.create(cfg_fdk)
    # astra.algorithm.run(alg_id, 1)

    # ## SIRT
    # cfg_fdk = astra.astra_dict('SIRT_CUDA')
    #
    # cfg_fdk['ProjectionDataId'] = proj_id
    # cfg_fdk['ReconstructionDataId'] = vol_id
    # alg_id = astra.algorithm.create(cfg_fdk)
    # astra.algorithm.run(alg_id, 200)

    ## SART
    cfg_fdk = astra.astra_dict('SART_CUDA')

    cfg_fdk['ProjectionDataId'] = proj_id
    cfg_fdk['ReconstructionDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg_fdk)
    astra.algorithm.run(alg_id, 500)


    # ## CGLS
    # cfg_fdk = astra.astra_dict('CGLS_CUDA')
    #
    # cfg_fdk['ProjectionDataId'] = proj_id
    # cfg_fdk['ReconstructionDataId'] = vol_id
    # alg_id = astra.algorithm.create(cfg_fdk)
    # astra.algorithm.run(alg_id, 1)



    # ## CPU algorithms
    # cfg_fdk = astra.astra_dict('CGLS')
    # cfg_fdk['ProjectionId'] = proj_id_new
    # cfg_fdk['ProjectionDataId'] = proj_id
    # cfg_fdk['ReconstructionDataId'] = vol_id
    # alg_id = astra.algorithm.create(cfg_fdk)
    # astra.algorithm.run(alg_id, 1)

    # ## template
    # cfg_fdk = astra.astra_dict('SIRT_CUDA')
    #
    # cfg_fdk['ProjectionDataId'] = proj_id
    # cfg_fdk['ReconstructionDataId'] = vol_id
    # # cfg_fdk['option'] = { 'FilterType': 'Ram-Lak' }
    # # cfg_fdk['option'] = {}
    # cfg_fdk['option']['ShortScan'] = False
    # alg_id = astra.algorithm.create(cfg_fdk)
    # astra.algorithm.run(alg_id, 1)


    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    ### remove edge noise ########################################################
    mask = utils.gen_mask(vol_rec, x_min, x_max, magnification, projs_cols, det_spacing_x, src_x_det_crd, rot_x_det_crd)
    vol_rec = utils.remove_edge_noise_2d(vol_rec, mask)

    print(np.max(vol_rec))
    print(np.min(vol_rec))
    print(np.mean(vol_rec))

    plt.imshow(vol_rec)
    plt.show()
    slice_path = '{:.6f}.png'.format(rot_x_det_crd)
    imageio.imwrite(slice_path, vol_rec)


print(np.round_(time.time() - t_start, 3), 'sec in total')