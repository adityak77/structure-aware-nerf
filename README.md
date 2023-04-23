# Structure-Aware NeRF

Clone the repo as follows:

```
git clone https://github.com/adityak77/structure-aware-nerf.git
```

To install the environment,

```
conda env create --file environment.yml
conda activate structure-nerf

# install nerfstudio
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

To install segment-anything

```
cd segment-anything
pip install -e .
mkdir models && cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

To run segment-anything demo for panoptic segmentation (our use case), run `python demo/demo.py  --img demo/desk.jpg`. For more examples see `segment-anything/notebooks/automatic_mask_generator_example.ipynb`.

To install a separate environment for Omni-3D/Cube-RCNN

```
cd omni3d
conda env create --file environment_cubercnn.yml
conda activate cubercnn3

pip install cython opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

## Extract 6D Object Poses

Run the following

```
cd omni3d
python demo/demo_scannet.py --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml --scannet-folder /path/to/scans/scene[spaceid]_[scanid]/ MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth OUTPUT_DIR output/test
```

## Download and Process Scannet

Run the following commands to download. To download a single scene, add the `--id scene%04d_%02d` argument. The format is ` scene[spaceid]_[scanid]` where there are approximately 
2-3 scans per space. For example, `scene0000_00`, `scene0000_01`, `scene0002_00` are valid scenes.

```
python download-scannet.py -o /scannet/root/folder --type .sens
python download-scannet.py -o /scannet/root/folder --type _2d-instance-filt.zip
python download-scannet.py -o /scannet/root/folder --type .aggregation.json
```

Run the following to extract data locally
```
# unzip instance files, extract images
sh extract-scannet.sh /scannet/root/folder

# extract object pose using Cube-RCNN model
python demo/demo_scannet.py --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml --scannet-folder "/scannet/root/folder/scene[space_id]_[scan_id]/" MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth OUTPUT_DIR output/test
```

To save the relevant information in a pickle file, run the following:
```
python process-scannet.py --scannet_dir /scannet/root/folder --output_dir /output/dir
```

This will save a pickle with instance/class labels, camera intrinsics/extrinsics, and segmentations (label maps) of all 2D images. Note that instance labels for objects are 1-indexed; 
there will be 0s in the label map that correspond to unannotated pixels.

## Running NeRF experiments (ScanNet)

### Run on all images and all objects in a scene

```
ns-train nerfacto scannet-data --data /path/to/scans/scene[spaceid]_[scanid]/

# Run longer with depth-supervised NeRF (much higher quality)
ns-train depth-nerfacto-big scannet-data --data /path/to/scans/scene[spaceid]_[scanid]/
```

### Run on single object

Below, object instance is the id of the object (corresponds to the value of the object in the mask; equal 1 + id in `.aggreggation.json` file). If you set object instance to be 0, that corresponds to the background. The `fraction_nonmask_pixel_sample` argument corresponds to the fraction of pixels that are sampled from the background. Usually values in [0.3, 0.5] work well.

```
ns-train depth-nerfacto-big custom-scannet-data --data /path/to/scans/scene[spaceid]_[scanid]/ --object_instance 54 --fraction_nonmask_pixel_sample 0.5
```