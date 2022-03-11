import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda"
import pickle
import torch
import torchvision
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from nokogiri.working_dir import working_dir

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

query = "flandre scarlet"
threshold=0.9
skip = 0
batch = 32
sample_num = 1 * 10**7
save_freq = 2500
assert sample_num % batch == 0
assert (sample_num//batch)%save_freq == 0
label = torch.zeros([1, G.c_dim], device=device)
root = Path(f"/data/natsuki/fact_{query.replace(' ', '_')}_{fm.parent.stem}")
root.mkdir(exist_ok=True)
pbar = tqdm(range(sample_num//batch), dynamic_ncols=True)
ok = 0
preds = list()
for j in pbar:
    synth_image = G.synthesis(
        G.mapping(
            torch.from_numpy(np.array([np.random.RandomState(batch*j+i+skip).randn(G.z_dim) for i in range(batch)])).to(device),
            label
        ),
        noise_mode='const'
    )
    synth_image = (synth_image + 1) * (255/2)
    pros_image = to224(synth_image)[:,[2,1,0],:,:] - mean
    pred = model_tag(pros_image)[:,:,0,0]
    preds.append(pred.cpu().numpy())
    for i in range(batch):
        seed = batch*j+i+skip
        score = pred[i][model_tag.index[query]]
        if threshold < score:
            txt = f"{float(score):.8f}_"[2:]+f"{seed}.png"
            Image.fromarray(
                synth_image[i].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy(),
                'RGB'
            ).save(root/txt)
            ok += 1
    pbar.set_description(f"{ok=} {seed=}")
    if (j+1)%save_freq == 0:
        print("saving...")
        preds = [np.vstack(preds)]
        with open(root/"preds.npy", "wb" ) as f:
            np.save(f, preds[0])