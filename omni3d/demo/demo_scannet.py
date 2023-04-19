# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import glob
import json
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

def map_scannet_category_to_omni(cat, omni_cats):
    if cat in omni_cats:
        return cat
    elif 'cabinet' in cat:
        return 'cabinet'
    elif 'door' in cat:
        return 'door'
    elif 'chair' in cat:
        return 'chair'
    elif cat == 'trash can':
        return 'bin'
    elif cat == 'shelf':
        return 'shelves'
    elif cat == 'bookshelf':
        return 'bookcase'
    elif cat == 'nightstand':
        return 'night stand'
    else:
        return None

def do_test(args, cfg, model, intrinsics):

    list_of_ims = util.list_files(os.path.join(args.scannet_folder, 'color', ''), '*')
    list_of_masks = []
    for im_path in list_of_ims:
        im_name = int(util.file_parts(im_path)[1])
        mask_path = os.path.join(args.scannet_folder, 'instance-filt', f'{im_name}.png')
        list_of_masks.append(mask_path)
    
    json_path = glob.glob(f'{args.scannet_folder}/*.aggregation.json')[0]
    with open(json_path, 'r') as f:
        instance_info = json.load(f)

    index_to_scannet_label = {elem['id'] + 1 : elem['label'] for elem in instance_info['segGroups']}

    model.eval()
    
    # thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    cats_to_ind = {cat: i for i, cat in enumerate(cats)}

    all_object_poses = {}
    for path, mask_path in tqdm(zip(list_of_ims, list_of_masks), total=len(list_of_ims)):

        im_name = util.file_parts(path)[1]
        im = util.imread(path)
        
        image_shape = im.shape[:2]  # h, w

        K = intrinsics

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]  # 0 is the not annotated

        instance_classes_scannet = [index_to_scannet_label[instance_id] for instance_id in instance_ids]
        instance_classes_omni = [map_scannet_category_to_omni(cat, cats) for cat in instance_classes_scannet]

        instance_ids_scannet = []
        instance_ids_omni = []
        for i, cat in enumerate(instance_classes_omni):
            if cat is not None:
                instance_ids_omni.append(cats_to_ind[cat])
                instance_ids_scannet.append(instance_ids[i])

        if len(instance_ids_scannet) == 0:
            all_object_poses[im_name] = {}
            continue

        # this is a list of (left, top, right, bottom) tuples
        instance_coords = [np.where(mask == instance_id) for instance_id in instance_ids_scannet]
        instance_bbox = torch.tensor([
            [min(val[1]), min(val[0]), max(val[1]), max(val[0])] for val in instance_coords
        ])

        scale_x, scale_y = image.shape[1] / im.shape[1], image.shape[0] / im.shape[0]
        instance_bbox[:, [0, 2]] = (instance_bbox[:, [0, 2]] * scale_x).to(torch.int64)
        instance_bbox[:, [1, 3]] = (instance_bbox[:, [1, 3]] * scale_y).to(torch.int64)

        instance_bbox = instance_bbox.cuda()
        oracle_dict = {
            'gt_bbox2D': instance_bbox,
            'gt_classes': torch.tensor(instance_ids_omni).to(torch.int64).cuda()
        }

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K,
            'oracle2D': oracle_dict
        }]

        dets = model(batched)[0]['instances']
        n_det = len(dets)

        if n_det != len(instance_ids_scannet):
            all_object_poses[im_name] = {}
            continue

        meshes = []
        meshes_text = []
        bbox_3d = []

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):
                cat = cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)

                bbox_3d.append(bbox3D + pose.reshape(-1).tolist()) # X, Y, Z, L, W, H, 9D rotation matrix
        
        if args.display:
            print('File: {} with {} dets'.format(im_name, len(meshes)))

            if len(meshes) > 0:
                im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
                
                if args.display:
                    im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
                    vis.imshow(im_concat)

                util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
                util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))
            else:
                util.imwrite(im, os.path.join(output_dir, im_name+'_boxes.jpg'))

        # find correspondences between segmented objects and 3d bboxes here
        poses = {
            str(instance_ids_scannet[idx]): bbox_3d[idx] for idx in range(len(instance_ids_scannet))
        }

        # save as a json file
        all_object_poses[im_name] = poses

    save_dir = os.path.join(args.scannet_folder, 'object_poses.json')
    with open(save_dir, 'w') as f:
        json.dump(all_object_poses, f)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    intrinsics_file = os.path.join(args.scannet_folder, 'intrinsic', 'intrinsic_color.txt')
    intrinsics = np.loadtxt(intrinsics_file)
    intrinsics = intrinsics[:3, :3]

    with torch.no_grad():
        do_test(args, cfg, model, intrinsics)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--scannet-folder',  type=str, help='list of image folders to process', required=True)
    # parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )