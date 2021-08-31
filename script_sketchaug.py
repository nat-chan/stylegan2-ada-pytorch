#!/usr/bin/env python3
# %%
from script_util import root, fname2id, id2, prepare
import cv2
from scipy import ndimage
import numpy as np
import sys
from tqdm import tqdm
from math import gcd
# %%
#sketch = list(map(str, (root/"danbooru2020/whitewaist_sketch/0000/").glob("*.png")))
#sim = list(map(str, (root/"danbooru2020/whitewaist_sim/0000/").glob("*.png")))
# %%
def distance_field_map(arr_in):
    arr_in = (255*(arr_in >= 250)).astype("uint8")
    arr_out = ndimage.distance_transform_edt(arr_in)
    return arr_out, cv2.equalizeHist(
        (255*arr_out/arr_out.max()).astype("uint8")
    )
    

def line_width(arr_in, dilation=True, k=2): # erosion
    if k == 1:
        arr_out = arr_in
    else:
        f = [ndimage.grey_erosion, ndimage.grey_dilation]
        arr_in = 255-arr_in
        arr_out = f[dilation](arr_in, size=(k, k))
        arr_out = 255-arr_out
    return arr_out

if __name__ == "__main__":
    for i, _id in enumerate(tqdm(list(map(fname2id, sys.stdin)))):
        for src in ["whitewaist_sketch", "whitewaist_sim"]:
            arr_in = cv2.imread(id2(src)(_id), 0)
            for k1 in range(1, 6):
                arr_k1 = line_width(arr_in, dilation=True, k=k1)
                for k2 in range(1, 6):
                    if k1 == 1 and k2 > 2: continue
                    if k1 == 2 and k2 > 4: continue
                    if k1 == 3 and k2 > 4: continue
                    if gcd(k1, k2) != 1: continue
                    arr_k2 = line_width(arr_k1, dilation=False, k=k2)
                    dst = f"{src}_/d{k1}e{k2}"
                    if i == 0: prepare(dst)
                    cv2.imwrite(id2(dst)(_id), arr_k2)
                    arr_raw, arr_df = distance_field_map(arr_k2)
                    dst += "_df"
                    if i == 0: prepare(dst)
                    cv2.imwrite(id2(dst)(_id), arr_df)
                    np.save(id2(dst, ext=lambda _: "npy")(_id), arr_raw)