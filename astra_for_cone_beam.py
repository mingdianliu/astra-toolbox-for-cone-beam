
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
import matplotlib.pyplot as plt


#### user defined settings #####################################################

# define a sub-sampling factor in angular direction
# (all reference reconstructions are computed with full angular resolution)
angluar_sub_sampling = 1
# select of voxels per mm in one direction (higher = larger res) for volume reconstruction
# (all reference reconstructions are computed with 10)
voxel_per_mm = 45

# we enter here some intrinsic details of the dataset needed for our reconstruction scripts
# set the variable "data_path" to the path where the dataset is stored on your own workstation
data_path = './raw/'

# set the variable "recon_path" to the path where you would like to store the
# reconstructions you compute
recon_path = './recon'

# the prefix of raw data
projs_name = 'CrnStlk{}.raw'

# number of projections
n_pro = 360

# size of the reconstruction volume in voxels
vol_sz = 3*(1500 + 1,)

# set true if you already rotate raw image by 90 degree
ard_rot = False


#### load data #################################################################

t = time.time()
print('load data', flush=True)


# the configuration of detector
projs_cols = 3888
projs_rows = 3072

det_spacing_x = 7.47942e-02  # [mm]
det_spacing_y = 7.48047e-02  # [mm]

angles = np.linspace(0, 2 * np.pi, num=n_pro, endpoint=False)

# create the numpy array which will receive projection data from raw data
projs = np.zeros((projs_rows, n_pro, projs_cols), dtype=np.float32) # (n_pro, projs_rows, projs_cols)


# load projection data
imageSize = (3888, 3072)

for i in range(n_pro):

    npimg = np.fromfile(os.path.join(data_path, projs_name.format(i)), dtype=np.uint16)
    if ard_rot:
        imageSize = list(reversed(imageSize))

    npimg = npimg.reshape(imageSize)
    ## you can check the orientation of raw image by plotting it
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


# size of a cubic voxel in mm
vox_sz  = 1/voxel_per_mm

# numpy array holding the reconstruction volume
vol_rec = np.zeros(vol_sz, dtype=np.float32)

# we need to specify the details of the reconstruction space to ASTRA
# this is done by a "volume geometry" type of structure, in the form of a Python dictionary
# by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
vol_geom = astra.create_vol_geom(vol_sz)
vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz


# set up configuration
dist_source_detector = 985  # [mm]
dist_origin_detector = 861.875  # [mm]
dist_source_origin = dist_source_detector - dist_origin_detector  # [mm]

# source position in the coordinate of detector
src_x_det_crd = -1.04712  # [mm] # CStlk
src_y_det_crd = -38.5992  # [mm] # CStlk
src_z_det_crd = dist_source_detector

# rotation center position in the coordinate of detector
rot_x_det_crd = 0.46329  # [mm] # CStalk
rot_y_det_crd = 0  # [mm]
rot_z_det_crd = dist_origin_detector

angles = np.linspace(0, 2*np.pi, n_pro, False)

def cal_vecs(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles,
             det_spacing_x, det_spacing_y):

    src_pos = [-rot_x_det_crd, -(src_z_det_crd - rot_z_det_crd),
               -src_y_det_crd * (src_z_det_crd - rot_z_det_crd) / src_z_det_crd]
    det_pos = [-src_x_det_crd-rot_x_det_crd, rot_z_det_crd, src_y_det_crd * rot_z_det_crd / src_z_det_crd]

    # flip_src_pos_z = False    # CStalk
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


### save reconstruction ########################################################

t = time.time()
print('save results', flush=True)

# create the directory in case it doesn't exist yet
if not os.path.exists(recon_path):
    os.makedirs(recon_path)

# Save every slice in  the volume as a separate tiff file
for i in range(vol_sz[0]):
    slice_path = os.path.join(recon_path, 'fdk_ass{}_vmm{}_{:03}.tiff'.format(
                                  angluar_sub_sampling, voxel_per_mm, i))
    imageio.imwrite(slice_path, vol_rec[i,...])

print(np.round_(time.time() - t, 3), 'sec elapsed')
