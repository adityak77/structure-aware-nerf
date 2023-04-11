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