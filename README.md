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


## Download Scannet

Run the following to download:

```
python download-scannet.py -o /mnt/sda/akannan2/scannet --id scene0000_00 --type .sens
python download-scannet.py -o /mnt/sda/akannan2/scannet --id scene0000_00 --type _2d-instance-filt.zip
python download-scannet.py -o /mnt/sda/akannan2/scannet --id scene0000_00 --type _2d-label-filt.zip
```

Run the following to extract data locally
```
# unzip instance and label files
sh unzip-scannet.sh /mnt/sda/akannan2/scannet/scans

# Read .sens files
python ScanNet/SensReader/python/reader.py --filename /mnt/sda/akannan2/scannet/scans/scene0000_00/scene0000_00.sens --output_path /mnt/sda/akannan2/scannet/scans/scene0000_00/ --export_color_images --export_pose --export_intrinsics --frame_skip 10
```
