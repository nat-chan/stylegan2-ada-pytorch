# %%
import cv2
import numpy as np
from script_util import fname2id, id2, prepare
import sys
from tqdm import tqdm
# %%
# https://qiita.com/Shirataki2/items/813fdade850cc69d1882
# ガウシアン差分フィルタリング
def DoG(img,size, sigma, k=1.6, gamma=1):
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    return g1 - gamma*g2

# 閾値で白黒化するDoG
def thres_dog(img, size, sigma, eps, k=1.6, gamma=0.98):
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    d *= 255
    d = np.where(d >= eps, 255, 0)
    return d

# 拡張ガウシアン差分フィルタリング
"""
AlacGAN
φ=10^9
σ=0.3, 0.4, 0.5
additionaly
τ=0.95
κ=4.5
"""

params = [
    dict(sigma=1.4, eps=10, phi=3, gamma=0.98),
    dict(sigma=0.8, eps=10, phi=3, gamma=0.98),
    dict(sigma=1.4, eps=30, phi=10, gamma=0.98),
    dict(sigma=1.9, eps=0, phi=25, gamma=0.98),
]
def xdog(img, size=5, sigma=1.9, eps=0, phi=25, k=1.6, gamma=0.98):
    eps /= 255
    d = DoG(img, size, sigma, k, gamma)
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e


# %%
if __name__ == "__main__":
    fnames = sys.stdin
    for i, _id in enumerate(tqdm(list(map(fname2id, fnames)))):
        for src in ["whitechest", "whitewaist"]:
            fname = id2[src](_id)
            L = cv2.cvtColor( cv2.imread(str(fname)), cv2.COLOR_BGR2LAB)[:,:,0]
            for size in range(3, 11+1, 2):
                for p in range(len(params)):
                    dst = f"{src}_xdog/k{size}p{p}"
                    if i == 0: prepare(dst)
                    dname = id2[dst](_id)
                    arr_out = (255*xdog(L, size=size, **params[p])).astype("uint8")
                    cv2.imwrite(dname, arr_out)