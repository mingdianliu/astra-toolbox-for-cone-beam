# Astra-toolbox-for-cone-beam
This is a collection of Python scripts for implementing [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) 
for cone-beam X-ray CT reconstruction. The complete scripts consist of raw data loading, configuration loading, center 
of rotation determination, back-propagation/iterative reconstruction, edge noise removal, and 3D volume generation. The 
current script includes all functions but 3D volume generation. We will add denoising, 3D volume generation, and 
detailed tutorial later on.

Before executing the script, please download the dataset from [here](https://drive.google.com/file/d/1MB4gLI_lRbVqmQA0ofnqwM9qFJ1joQwE/view) 
and save it in `./raw` file. The image for determining the center of rotation will be saved to `offset_determination`.
After the code is executed, the reconstructed data will be saved to `./recon`. You can refer to the comments in the 
script to modify the configuration setting in terms of your source and detector context.


## Requirements

* All scripts make use of the development version (more recent than 1.9.0dev) of the 
* [ASTRA toolbox](https://www.astra-toolbox.com/). If you are using conda, this is available through the `astra-toolbox/label/dev` channel.

## Setup environment

Update apt-get

```shell
sudo apt update
```

Setup conda env:
```shell
conda env create -f environment.yml
conda activate astra-cone-beam
```

## Contributors

Mingdian Liu (lmdvigor@gmail.com)
