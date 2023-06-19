#!/usr/bin/env python3
# %%
from script_util import root, fname2id, id2, prepare
from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.segmentation import slic
import sys
import argparse
from argparse import Namespace
from pathlib import Path
import csv

def main(args: Namespace) -> None:
    with open(args.csvpath, "r") as f:
        reader = csv.reader(f)
        file_paths = [Path(line[0]) for line in reader]
    for i, fname in enumerate(tqdm(file_paths, dynamic_ncols=True)):
        _id = fname2id(fname)
        for p in [1, 30]:
            dst = f"whitewaist_slic{p}"
            if i == 0: prepare(dst)
            flat = io.imread(id2.flatten(_id))
            mask = io.imread(id2.whitewaist_mask(_id)) >= 128
            out = np.full((512,512,3), 255, dtype="uint8")
            if p == 1:
                index = np.where(mask)
                out[index] = np.median(flat[index], axis=0)
            else:
                seg = slic(flat, n_segments=p, mask=mask)
                for j in range(seg.max()+1):
                    index = np.where(seg == j)
                    out[index] = np.median(flat[index], axis=0)
            io.imsave(id2[dst](_id), out, check_contrast=False)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath", type=str, default="/data/natsuki/danbooru2020/filepath.csv")
    parser.add_argument("--root", type=str, default="/data/natsuki/danbooru2020")
    args = parser.parse_args()
    main(args)