# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# %%
import pickle
import torch
import numpy as np
from PIL import Image
from pathlib import Path
# python projector.py --outdir /data/natsuki/danbooru2020/shizuku --target /data/natsuki/danbooru2020/shizuku/muzu2.png --network 
fm = "/data/natsuki/training116/00023-white_yc05_yw04-mirror-auto4-gamma10-noaug/network-snapshot-021800.pkl"
fm = Path("/data/natsuki/training116/00030-v4-mirror-auto4-gamma100-batch64-noaug-resumecustom/network-snapshot-011289.pkl")
assert fm.is_file()
# %%

fw1 = "/data/natsuki/danbooru2020/shizuku/projected_w.npz"
fw2 = "/data/natsuki/danbooru2020/shizuku/maid_projected_w.npz"
fw3 = "/data/natsuki/danbooru2020/shizuku/muzu_projected_w.npz"
# %%
with open(fm, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()
def synth(w):
    synth_image = G.synthesis(w, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    img = Image.fromarray(synth_image, 'RGB')
    return img
# %%
with np.load(fw1) as data:
    w1 = torch.from_numpy(data["w"]).cuda()
# %%
with np.load(fw2) as data:
    w2 = torch.from_numpy(data["w"]).cuda()
# %%
with np.load(fw3) as data:
    w3 = torch.from_numpy(data["w"]).cuda()
# %%
p = 0.4
q = 0.5
a = w1*p+w2*(1-p)
a = a*q+w3*(1-q)
synth(a)
# %%
(a**2).mean()**.5
# %%
synth
# %%
