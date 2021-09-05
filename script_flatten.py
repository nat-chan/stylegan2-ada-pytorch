#!/usr/bin/env python3
# %%
from script_util import root, fname2id, id2, prepare
from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.segmentation import slic
import sys
# id2 = CurriedFunc(lambda *args, **kwargs: id2fname(*args, **kwargs, root="./root"))
# _id = random.choice(list(d.indices))
# %%

def check():
    from dotdict import dotdict
    d = dotdict()
    for key in ["whitewaist_mask", "whitewaist", "indices", "flatten", "v4", "v4_val"]:
        d[key] = set(map(fname2id, (root/f"danbooru2020/{key}").glob("**/*.p*")))
        print(key, len(d[key]))

if __name__ == "__main__":
    for i, fname in enumerate(tqdm(list(map(lambda x: x.strip(), sys.stdin)), dynamic_ncols=True)):
        _id = fname2id(fname)
        for p in [1] + list(range(2, 20, 2)) + list(range(20, 100+1, 10)):
            dst = f"flatten_/slic{p}"
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
