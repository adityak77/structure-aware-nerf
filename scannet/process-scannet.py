import argparse
import cv2
import json
import os
import pickle
import numpy as np

def load_mat_from_txt(path):
    return np.loadtxt(path)

def process_one(scan_dir_path):
    scan_name = os.path.basename(scan_dir_path)
    print(f'Processing scan {scan_name}')

    save_path = os.path.join(scan_dir_path, f"{scan_name}.pkl")
    if os.path.exists(save_path):
        print(f"Skipping {scan_name} because it already exists")
        return

    frame_idxs = [os.path.splitext(os.path.split(f)[1])[0] for f in os.listdir(os.path.join(scan_dir_path, 'color'))]
    num_frames = len(frame_idxs)

    with open(os.path.join(scan_dir_path, f'{scan_name}.aggregation.json'), 'r') as f:
        instance_info = json.load(f)

    with open(os.path.join(scan_dir_path, 'object_poses.json'), 'r') as f:
        object_poses = json.load(f)

    distinct_objects = set()
    for d in object_poses:
        distinct_objects.update(object_poses[d].keys())
    distinct_objects = sorted([int(elem) for elem in distinct_objects])
    
    instance_data = []
    for instance in instance_info['segGroups']:
        instance_dict = {}

        instance_dict['class'] = instance['label']
        instance_dict['instance_id'] = instance['id'] + 1
        instance_data.append(instance_dict)

    save_dict = {
        'instance_data': instance_data,
        'objects_with_poses': distinct_objects,
        'num_frames': num_frames,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

def process_all(args):
    scan_names = os.listdir(args.scannet_dir)
    for scan_name in scan_names:
        scan_dir_path = os.path.join(args.scannet_dir, scan_name)
        process_one(scan_dir_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_dir', required=True, help='path to scannet directory')
    args = parser.parse_args()

    process_all(args)

if __name__ == '__main__':
    main()