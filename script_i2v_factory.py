# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda"
import pickle
import torch
import torchvision
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from nokogiri.working_dir import working_dir
from typing import List, Tuple
import heapq # (-pred, seed)を保存していくヒープ
# %%
#fm = Path("/data/natsuki/training116/00030-v4-mirror-auto4-gamma100-batch64-noaug-resumecustom/network-snapshot-011289.pkl")
fm = Path("/data/natsuki/training116/00023-white_yc05_yw04-mirror-auto4-gamma10-noaug/network-snapshot-021800.pkl")
assert fm.is_file()
with open(fm, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
with working_dir("/home/natsuki/illustration2vec"):
    import i2v.pytorch_i2v
    model_tag = i2v.pytorch_i2v.PytorchI2V(
                "illust2vec_tag_ver200.pth", "tag_list.json").eval().requires_grad_(False).to(device)
to224 = torchvision.transforms.Resize((224, 224))
mean = torch.from_numpy(model_tag.mean[None,:,None,None]).float().to(device)
# %%
root = Path(f"/data/natsuki/factory")
root.mkdir(exist_ok=True)
master: List[List[Tuple[float, int]]] = [list() for _ in model_tag.tags]
def save(skip):
    for k, tag in enumerate(model_tag.tags):
        with open(root/f"{k}.pkl", "wb") as f:
            pickle.dump(master[k], f)
    with open(root/f"skip.txt", "w") as f:
        f.write(str(skip))
def lload():
    try:
        for k, tag in enumerate(model_tag.tags):
            with open(root/f"{k}.pkl", "rb") as f:
                master[k] = pickle.load(f)
        with open(root/f"skip.txt", "r") as f:
            skip = int(f.read())
    except:
        for k, tag in enumerate(model_tag.tags):
            master[k] = list()
        skip = 0
    return skip
skip = lload()
# %%
threshold=0.9
batch = 32
sample_num = 1 * 10**6
#sample_num = 1 * 10**5
#save_freq = 2500
save_freq = 25
assert sample_num % batch == 0
assert (sample_num//batch)%save_freq == 0
label = torch.zeros([1, G.c_dim], device=device)
pbar = tqdm(range(sample_num//batch), dynamic_ncols=True)
#preds = list()
for i in pbar:
    synth_image = G.synthesis(
        G.mapping(
            torch.from_numpy(np.array([np.random.RandomState(batch*i+j+skip).randn(G.z_dim) for j in range(batch)])).to(device),
            label
        ),
        noise_mode='const'
    )
    synth_image = (synth_image + 1) * (255/2)
    pros_image = to224(synth_image)[:,[2,1,0],:,:] - mean
    pred = model_tag(pros_image)[:,:,0,0]
#    preds.append(pred.cpu().numpy())
    for j in range(batch):
        seed = batch*i+j+skip
        for k, tag in enumerate(model_tag.tags):
            score = float(pred[j][k])
            if threshold < score:
                heapq.heappush(master[k], (-score, seed))
    pbar.set_description(f"skip={skip}, now={seed+1}")
    if (i+1)%save_freq == 0:
        print(f"saving...{sum(len(m) for m in master)}")
        save(seed+1)
#        preds = [np.vstack(preds)]
#        with open(root/"preds.npy", "wb" ) as f:
#            np.save(f, preds[0])