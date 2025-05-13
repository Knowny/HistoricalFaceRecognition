#!/usr/bin/env python3
"""
Utility for extracting images from CASIA WebFace dataset.
Saves as .jpg files with following data folder structure:
casia_images/<identity>/<id>.jpg

file: extract_casia.py
author: Tereza Magerkova, xmager00
"""

# overcome a hiccup with mxnet
import numpy as np
np.bool = bool

import cv2
import os
import mxnet as mx
from tqdm import tqdm
from collections import defaultdict
from itertools import islice

IMAGES_PER_ID = 15

def load_mx_rec(idx_path, rec_path, output_dir = 'casia_images'):
    """
    Loads the MXNet record file (.rec) and saves images aj .jpg files
    """
    os.makedirs(output_dir, exist_ok=True)

    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])

    label_to_indices = defaultdict(list)
    for idx in tqdm(range(1, max_idx + 1), desc = "Indexing  samples"):
        try:
            rec = imgrec.read_idx(idx)
            header, _ =  mx.recordio.unpack(rec)
            label =  int(header.label)
            label_to_indices[label].append(idx)
        except Exception:
            continue
    print(f"Found {len(label_to_indices)} identities, extracting {IMAGES_PER_ID} per ID")

    for label, indices in tqdm(islice(label_to_indices.items(), 1000), desc = "Extracting images"):
        identity = str(label).zfill(6)
        dst_dir = os.path.join(output_dir, identity)
        os.makedirs(dst_dir, exist_ok=True)
        n_images = len(indices)
        for i in range(IMAGES_PER_ID):
            rec_idx = indices[i % n_images]
            rec = imgrec.read_idx(rec_idx)
            header, img = mx.recordio.unpack_img(rec)
            sample_name = f"{i:02d}.jpg"
            sample_path = os.path.join(dst_dir, sample_name)
            cv2.imwrite(sample_path, img)

if __name__ == "__main__":
    # paths to the .idx and .rec files
    idx_path = 'train.idx'
    rec_path = 'train.rec'

    load_mx_rec(idx_path, rec_path)
