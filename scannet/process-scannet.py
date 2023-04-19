import argparse
import cv2
import json
import os
import pickle
import numpy as np

def load_mat_from_txt(path):
    return np.loadtxt(path)

def process_one(scan_dir_path, output_dir):
    scan_name = os.path.basename(scan_dir_path)
    print(f'Processing scan {scan_name}')

    save_path = os.path.join(output_dir, f"{scan_name}.pkl")
    if os.path.exists(save_path):
        print(f"Skipping {scan_name} because it already exists")
        return

    frame_idxs = [os.path.splitext(os.path.split(f)[1])[0] for f in os.listdir(os.path.join(scan_dir_path, 'color'))]
    num_frames = len(frame_idxs)

    with open(os.path.join(scan_dir_path, f'{scan_name}.aggregation.json'), 'r') as f:
        instance_info = json.load(f)
    
    intrinsics = load_mat_from_txt(os.path.join(scan_dir_path, 'intrinsic', 'intrinsic_color.txt'))
    extrinsics = [
        load_mat_from_txt(os.path.join(scan_dir_path, 'pose', f'{i}.txt')) for i in frame_idxs
    ]

    label_map = [
        cv2.imread(os.path.join(scan_dir_path, 'instance-filt', f'{i}.png'), cv2.IMREAD_GRAYSCALE) for i in frame_idxs
    ]

    instance_data = []
    for instance in instance_info['segGroups']:
        instance_dict = {}

        instance_dict['class'] = instance['label']
        instance_dict['instance_id'] = instance['id'] + 1
        instance_data.append(instance_dict)
    
    camera_data = {
        'intrinsics': intrinsics,
        'extrinsics': np.array(extrinsics),
    }

    save_dict = {
        'instance_data': instance_data,
        'label_map': np.array(label_map),
        'camera_data': camera_data,
        'num_frames': num_frames,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

def process_all(args):
    scan_names = os.listdir(args.scannet_dir)
    for scan_name in scan_names:
        scan_dir_path = os.path.join(args.scannet_dir, scan_name)
        process_one(scan_dir_path, args.output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_dir', required=True, help='path to scannet directory')
    parser.add_argument('--output_dir', required=True, help='path to output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_all(args)

if __name__ == '__main__':
    main()