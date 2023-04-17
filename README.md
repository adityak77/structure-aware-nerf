# Structure-Aware NeRF

Clone the repo as follows:

```
git clone https://github.com/adityak77/structure-aware-nerf.git --recurse-submodules
```

If you forget to add `--recurse-submodules`, do `git submodule init` and then `git submodule update`.

To install the environment,

```
conda create -n structure-nerf python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # note this installs pytorch 2.0
conda env update --file environment.yaml --prune
```

To install segment-anything

```
cd segment-anything
pip install -e .
mkdir models && cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

To run segment-anything demo for panoptic segmentation (our use case), run `python demo/demo.py  --img demo/desk.jpg`. For more examples see `segment-anything/notebooks/automatic_mask_generator_example.ipynb`.


## Download and Process Scannet

Run the following commands to download. To download a single scene, add the `--id scene%04d_%02d` argument. The format is ` scene[spaceid]_[scanid]` where there are approximately 
2-3 scans per space. For example, `scene0000_00`, `scene0000_01`, `scene0002_00` are valid scenes.

```
python download-scannet.py -o /scannet/root/folder --type .sens
python download-scannet.py -o /scannet/root/folder --type _2d-instance-filt.zip
# python download-scannet.py -o /scannet/root/folder --type _2d-label-filt.zip
python download-scannet.py -o /scannet/root/folder --type .aggregation.json
```

Run the following to extract data locally
```
# unzip instance files, extract images
sh extract-scannet.sh /scannet/root/folder
```

To save the relevant information in a pickle file, run the following:
```
python process-scannet.py --scannet_dir /scannet/root/folder --output_dir /output/dir
```

This will save a pickle with instance/class labels, camera intrinsics/extrinsics, and segmentations (label maps) of all 2D images. Note that instance labels for objects are 1-indexed; 
there will be 0s in the label map that correspond to unannotated pixels.