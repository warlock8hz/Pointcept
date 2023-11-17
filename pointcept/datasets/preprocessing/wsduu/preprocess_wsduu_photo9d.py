"""
Preprocessing Script for LiDARist 12/13-dimensional HDF5 training/validation data

Author: Wenzheng Fan (vincent.fan@lidarist.com)
"""

import warnings

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os
import argparse
import glob
import json
# import plyfile
import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Load external constants
# from meta_data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20

# SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
# AGGREGATIONS_FILE_PFIX = ".aggregation.json"
# CLASS_IDS200 = VALID_CLASS_IDS_200
# CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1

def _load_data_file(name):
    f = h5py.File(name, "r")
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def loadFileList():

    return None

def loadHD5():

    return None

def savePth():

    return None


def handle_process(
    # scene_path, output_path, labels_pd, train_scenes, val_scenes, parse_normals=True
    scene_path, output_path, train_scenes, val_scenes, parse_normals=True
):
    scene_id = os.path.basename(scene_path)
    cloud_path = scene_path + ".h5"
    # segments_file = os.path.join(
    #     scene_path, f"{scene_id}{SEGMENTS_FILE_PFIX}"
    # )
    # aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    # info_file = os.path.join(scene_path, f"{scene_id}.txt")

    if scene_id in train_scenes:
        output_file = os.path.join(output_path, "train", f"{scene_id}.pth")
        split_name = "train"
    elif scene_id in val_scenes:
        output_file = os.path.join(output_path, "val", f"{scene_id}.pth")
        split_name = "val"
    else:
        output_file = os.path.join(output_path, "test", f"{scene_id}.pth")
        split_name = "test"

    print(f"Processing: {scene_id} in {split_name}")

    points, labels = _load_data_file(cloud_path)
    coords = points[:, :3]
    colors = points[:, 3:6]
    save_dict = dict(coord=coords, color=colors, scene_id=scene_id)

    # # Rotating the mesh to axis aligned
    # info_dict = {}
    # with open(info_file) as f:
    #     for line in f:
    #         (key, val) = line.split(" = ")
    #         info_dict[key] = np.fromstring(val, sep=' ')
    #
    # if 'axisAlignment' not in info_dict:
    #     rot_matrix = np.identity(4)
    # else:
    #     rot_matrix = info_dict['axisAlignment'].reshape(4, 4)
    # r_coords = coords.transpose()
    # r_coords = np.append(r_coords, np.ones((1, r_coords.shape[1])), axis=0)
    # r_coords = np.dot(rot_matrix, r_coords)
    # coords = r_coords

    # Parse Normals
    if parse_normals:
        save_dict["normal"] = points[:, 6:9]

    # Load segments file
    # no segment information
    if split_name != "test":
        # with open(segments_file) as f:
        #     segments = json.load(f)
        #     seg_indices = np.array(segments["segIndices"])

        # Load Aggregations file
        # with open(aggregations_file) as f:
        #     aggregation = json.load(f)
        #     seg_groups = np.array(aggregation["segGroups"])

        # Generate new labels
        semantic_gt20 = np.ones((points.shape[0])) * IGNORE_INDEX
        semantic_gt200 = np.ones((points.shape[0])) * IGNORE_INDEX
        instance_ids = np.ones((points.shape[0])) * IGNORE_INDEX
        # for group in seg_groups:
        #     point_idx, label_id20, label_id200 = point_indices_from_group(
        #         seg_indices, group, labels_pd
        #     )

        #     semantic_gt20[point_idx] = label_id20
        #     semantic_gt200[point_idx] = label_id200
        #     instance_ids[point_idx] = group["id"]

        # semantic_gt20 = semantic_gt20.astype(int)
        # semantic_gt200 = semantic_gt200.astype(int)
        # instance_ids = instance_ids.astype(int)

        save_dict["semantic_gt"] = labels # semantic_gt20
        # save_dict["semantic_gt200"] = labels # semantic_gt200
        save_dict["instance_gt"] = instance_ids

        # Concatenate with original cloud
        processed_vertices = np.hstack((semantic_gt200, instance_ids))

        if np.any(np.isnan(processed_vertices)) or not np.all(
            np.isfinite(processed_vertices)
        ):
            raise ValueError(f"Find NaN in Scene: {scene_id}")

    # Save processed data
    torch.save(save_dict, output_file)



if __name__ == "__main__":
    print("Hello LiDarist!")
    loadFileList()
    loadHD5()
    savePth()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the wsduu dataset with each file as hdf5 file",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    # normal already created and saved in the last three columns
    parser.add_argument(
        "--parse_normals", default=True, type=bool, help="Whether parse point normals"
    )
    config = parser.parse_args()

    # Load label map
    # label saved in HDF5 file too
    # labels_pd = pd.read_csv(
    #     "pointcept/datasets/preprocessing/scannet/meta_data/scannetv2-labels.combined.tsv",
    #     sep="\t",
    #     header=0,
    # )

    # Load train/val splits
    # no load splits, just load all data
    # random 80/20 selection here for train and validation
    with open(
            "/mnt/d/_DataGeneration/HDF5Tools/202311021922_single_layer_photogrammetry_las12_9D_LAB_Train/filename.txt"
    ) as hdf5_file:
        hdf5_scenes = hdf5_file.read().splitlines()
    # with open(
    #         "pointcept/datasets/preprocessing/scannet/meta_data/scannetv2_val.txt"
    # ) as val_file:
    #     val_scenes = val_file.read().splitlines()

    # random shuffle, 80 train, 20 validation
    np.random.shuffle(hdf5_scenes)
    train_scenes = hdf5_scenes[:int(len(hdf5_scenes)*0.8)]
    val_scenes = hdf5_scenes[int(len(hdf5_scenes)*0.8):]

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    # scene_paths = sorted(glob.glob(config.dataset_root))
    scene_paths = []
    for scene in hdf5_scenes:
        scene_paths.append(os.path.join(config.dataset_root, scene))

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            #repeat(labels_pd),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(config.parse_normals),
        )
    )

    print("Done!")