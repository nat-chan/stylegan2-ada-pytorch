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
from script_util import csv_read, csv_write
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
threshold=0.9
batch = 32
sample_num = 10**6 #[9:38:32<00:00,  1.11s/it]
#sample_num = 1 * 10**5 #1時間程度
#save_freq = 2500
save_freq = 25
start = 10**6
end = start+sample_num
def save(skip):
    for k, tag in enumerate(model_tag.tags):
        csv_write(k, root, master, start=start, end=end)
    (root/"skip").mkdir(exist_ok=True)
    with open(root/f"skip/{start}_{end}.txt", "w") as f:
        f.write(str(skip))
def lload():
    try:
        for k, tag in enumerate(model_tag.tags):
            master[k] = csv_read(k, root, start=start, end=end)
        with open(root/f"skip/{start}_{end}.txt", "r") as f:
            skip = int(f.read())
    except:
        for k, tag in enumerate(model_tag.tags):
            master[k] = list()
        skip = 0
    return skip
#skip = lload()
skip = 0 # TODO resumeまだできてない
assert sample_num % batch == 0
assert (sample_num//batch)%save_freq == 0
label = torch.zeros([1, G.c_dim], device=device)
pbar = tqdm(range(sample_num//batch), dynamic_ncols=True)
#preds = list()
for i in pbar:
    synth_image = G.synthesis(
        G.mapping(
            torch.from_numpy(np.array([np.random.RandomState(batch*i+j+start).randn(G.z_dim) for j in range(batch)])).to(device),
            label
        ),
        noise_mode='const'
    )
    synth_image = (synth_image + 1) * (255/2)
    pros_image = to224(synth_image)[:,[2,1,0],:,:] - mean
    pred = model_tag(pros_image)[:,:,0,0]
#    preds.append(pred.cpu().numpy())
    for j in range(batch):
        seed = batch*i+j+start
        for k, tag in enumerate(model_tag.tags):
            score = float(pred[j][k])
            if threshold < score:
                master[k].append( (score, seed) )
#                heapq.heappush(master[k], (-score, seed))
    pbar.set_description(f"{start=} {end=} {skip=} {seed=}")
    if (i+1)%save_freq == 0:
        print(f"saving...{sum(len(m) for m in master)}")
        save(i+1)
#        preds = [np.vstack(preds)]
#        with open(root/"preds.npy", "wb" ) as f:
#            np.save(f, preds[0])