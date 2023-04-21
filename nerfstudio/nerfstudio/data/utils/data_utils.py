# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union
import json
import cv2
import numpy as np
import torch
from PIL import Image


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor

def get_scannet_image_mask_tensor_from_path(
        filepath: Path, 
        instance_id: int, 
        all_instances: List[int], 
        scale_factor: float = 1.0
    ) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    assert instance_id in all_instances or instance_id == 0 # instance mask or background mask
    mask = cv2.imread(filepath.as_posix(), cv2.IMREAD_GRAYSCALE)
    if instance_id != 0:
        binary_mask = (mask == instance_id).astype(np.uint8)
    else: # return all values in mask that do not belong to any instance
        binary_mask = np.isin(mask, all_instances, invert=True).astype(np.uint8)
    if scale_factor != 1.0:
        width, height = binary_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        binary_mask = cv2.resize(binary_mask, newsize, interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor

def get_obj_poseBox_tensor_from_path(filepath: Path):
    with open(filepath,'r') as f:
        lines = f.readlines()
    line = lines[0].split(' ')
    dim = np.array([float(line[8]),float(line[9]),float(line[10])])
    assert(dim.min()>=0)
    location = np.array([float(line[11]),float(line[12]),float(line[13])])
    box = torch.tensor([[location[0]-dim[0],location[1]-dim[1],location[2]-dim[2]],
                   [location[0]+dim[0],location[1]+dim[1],location[2]+dim[2]]])
    
    return box

def get_obj_poseBox_tensor_from_json(filepath: Path, frame_index: int, instance_id: int):
    if instance_id == 0:
        return torch.zeros((2, 3)) # dummy pose, include whole scene
    
    with open(filepath,'r') as f:
        data = json.load(f)

    print(frame_index, instance_id)    
    object_pose = data[str(frame_index)][str(instance_id)]
    center = object_pose[:3]
    dim = [elem / 2 for elem in object_pose[3:6]]

    box = torch.tensor([[center[0]-dim[0],center[1]-dim[1],center[2]-dim[2]],
                   [center[0]+dim[0],center[1]+dim[1],center[2]+dim[2]]])
    
    return box


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype="int64").view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])
