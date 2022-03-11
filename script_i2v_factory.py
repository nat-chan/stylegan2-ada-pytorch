import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda"
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from nokogiri.working_dir import working_dir

fm = Path("/data/natsuki/training116/00030-v4-mirror-auto4-gamma100-batch64-noaug-resumecustom/network-snapshot-011289.pkl")
#fm = Path("/data/natsuki/training116/00023-white_yc05_yw04-mirror-auto4-gamma10-noaug/network-snapshot-021800.pkl")
assert fm.is_file()
with open(fm, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
with working_dir("/home/natsuki/illustration2vec"):
    import i2v
    model_tag = i2v.make_i2v_with_chainer("illust2vec_tag_ver200.caffemodel", "tag_list.json")

query = "hatsune miku"
threshold=0.9
skip = 0
batch = 32
sample_num = 1 * 10**7
assert sample_num % batch == 0
label = torch.zeros([1, G.c_dim], device=device)
root = Path(f"/data/natsuki/fact_{query.replace(' ', '_')}_{fm.parent.stem}")
root.mkdir(exist_ok=True)
pbar = tqdm(range(sample_num//batch), dynamic_ncols=True)
ok = 0
for j in pbar:
    synth_image = G.synthesis(
        G.mapping(
            torch.from_numpy(np.array([np.random.RandomState(batch*j+i+skip).randn(G.z_dim) for i in range(batch)])).to(device),
            label
        ),
        noise_mode='const'
    )
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    imgs = [Image.fromarray(synth_image[i], 'RGB') for i in range(batch)]
    pred_tag = model_tag.estimate_plausible_tags(imgs, threshold=threshold)
    for i in range(batch):
        seed = batch*j+i+skip
        pbar.set_description(f"{ok=} {skip=} {seed=} {sample_num=}")
        filtered = next(filter(lambda x: x[0]==query, pred_tag[i]["character"]), None)
        if filtered:
            _, score = filtered
            txt = f"{score:.8f}_"[2:]+f"{seed}.png"
            imgs[i].save(root/txt)
            ok += 1