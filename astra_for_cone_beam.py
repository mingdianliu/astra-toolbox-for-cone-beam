
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
import astra
import os
import imageio.v2 as imageio
import time
import utils


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



#### user defined settings #####################################################

## configuration of reconstruction space

x_min, x_max = -20, 20 # [mm] reconstruction volume range
y_min, y_max = -20, 20 # [mm] reconstruction volume range
z_min, z_max = -20, 20  # [mm] reconstruction volume range

x_vol_sz, y_vol_sz, z_vol_sz = 1500, 1500, 1500 # volex number for x, y, z. Please make sure x_vol_sz=y_vol_sz


## file path and prefix configuration
# raw data path
data_path = 'H:\Reconstructed\CStalk'
# reconstruction path
recon_folder = 'recon'
# prefix of raw data
projs_name = 'CrnStlk{}.raw'
# number of projections
n_pro = 360

# set True when you already use Matlab script to rotate raw data by 90 degrees
ard_rot = True

# detector configuration
projs_cols = 3888
projs_rows = 3072
det_spacing_x = 7.47942e-02  # [mm]
det_spacing_y = 7.48047e-02  # [mm]


# distance of object to detector
dist_orgin_detector = 985
# distance of source to object
dist_source_origin = 123.125  # [mm]
# distance of object to detector
dist_origin_detector = dist_orgin_detector - dist_source_origin  # [mm]
magnification = dist_orgin_detector / dist_source_origin # [mm]

# source position
src_x_det_crd = -1.04712  # [mm]
src_y_det_crd = -38.5992  # [mm]
src_z_det_crd = dist_orgin_detector # [mm]

# center of rotation offset
rot_x_det_crd = 0.46329  # [mm]
rot_y_det_crd = 0  # [mm]
rot_z_det_crd = dist_origin_detector # [mm]


#### load data #################################################################

t = time.time()
print('load data', flush=True)


angles = np.linspace(0, 2 * np.pi, num=n_pro, endpoint=False)

# create the numpy array which will receive projection data from raw data
projs = np.zeros((projs_rows, n_pro, projs_cols), dtype=np.float32) # (n_pro, projs_rows, projs_cols)


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

    npimg = np.flipud(npimg)
    npimg = np.transpose(npimg)
    projs[:, i, :] = npimg # (projs_rows, n_pro, projs_cols)


print(np.round_(time.time() - t, 3), 'sec elapsed')


### pre-process data ###########################################################

t = time.time()
print('pre-process data', flush=True)
# take the negative log to linearize the data according to the Beer-Lambert law

projs /= 65535

np.log(projs, out=projs)
np.negative(projs, out=projs)
projs = np.ascontiguousarray(projs)
print(np.round_(time.time() - t, 3), 'sec elapsed')

### compute FDK reconstruction #################################################

t = time.time()
print('compute reconstruction', flush=True)


# numpy array holding the reconstruction volume
vol_sz = [x_vol_sz, y_vol_sz, z_vol_sz] # CStalk
vol_rec = np.zeros(vol_sz, dtype=np.float32)

# we need to specify the details of the reconstruction space to ASTRA
# this is done by a "volume geometry" type of structure, in the form of a Python dictionary
# by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
vol_geom = astra.create_vol_geom(vol_sz)
vol_geom['option']['WindowMinX'] = x_min
vol_geom['option']['WindowMaxX'] = x_max
vol_geom['option']['WindowMinY'] = y_min
vol_geom['option']['WindowMaxY'] = y_max
vol_geom['option']['WindowMinZ'] = z_min
vol_geom['option']['WindowMaxZ'] = z_max



angles = np.linspace(0, 2*np.pi, n_pro, False)

def cal_vecs(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles,
             det_spacing_x, det_spacing_y):

    src_pos = [-rot_x_det_crd, -(src_z_det_crd - rot_z_det_crd),
               -src_y_det_crd * (src_z_det_crd - rot_z_det_crd) / src_z_det_crd]
    det_pos = [-src_x_det_crd-rot_x_det_crd, rot_z_det_crd, src_y_det_crd * rot_z_det_crd / src_z_det_crd]

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

vecs = cal_vecs(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles, det_spacing_x, det_spacing_y)
proj_geom = astra.create_proj_geom('cone_vec',  projs_rows, projs_cols, vecs)

# register both volume and projection geometries and arrays to ASTRA
vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)
proj_id = astra.data3d.link('-sino', proj_geom, projs)

# finally, create an ASTRA configuration.
# this configuration dictionary setups an algorithm, a projection and a volume
# geometry and returns a ASTRA algorithm, which can be run on its own
cfg_fdk = astra.astra_dict('FDK_CUDA')

cfg_fdk['ProjectionDataId'] = proj_id
cfg_fdk['ReconstructionDataId'] = vol_id
cfg_fdk['option'] = {}
cfg_fdk['option']['ShortScan'] = False
alg_id = astra.algorithm.create(cfg_fdk)

# run FDK algorithm
astra.algorithm.run(alg_id, 1)

# release memory allocated by ASTRA structures
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)


print(np.round_(time.time() - t, 3), 'sec elapsed')



### remove edge noise ########################################################
mask = utils.gen_mask(vol_rec, x_min, x_max, magnification, projs_cols, det_spacing_x, src_x_det_crd, rot_x_det_crd)
vol_rec = utils.remove_edge_noise(vol_rec, mask)



### save reconstruction ########################################################

t = time.time()
print('save results', flush=True)


print(f'vol_rec {vol_rec.shape} {np.max(vol_rec)} {np.min(vol_rec)}')


recon_path = os.path.join(data_path, recon_folder)
print(recon_path)
# create the directory in case it doesn't exist yet
if not os.path.exists(recon_path):
    os.makedirs(recon_path)

# Save every slice in  the volume as a separate tiff file
for i in range(vol_sz[0]):
    slice_path = os.path.join(recon_path, 'recon_{:03}.tiff'.format(i))
    imageio.imwrite(slice_path, vol_rec[i,...])


print(np.round_(time.time() - t, 3), 'sec elapsed')