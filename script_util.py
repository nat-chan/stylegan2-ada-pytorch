from typing import *
from IPython.display import HTML, display
from pathlib import Path
from tqdm.notebook import tqdm
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from nokogiri.curry import curry
from collections.abc import Iterable

root = Path("/data/natsuki")

rcParams = {
    "font.size": 10,
    "font.family": "IPAPGothic",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.facecolor": "white",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
}

cmap = plt.get_cmap("tab20b")

def bucket(_id: str) -> str:
    return _id[-3:].zfill(4)

def prepare(prefix):
    (root/f"danbooru2020/{prefix}").mkdir(exist_ok=True)
    for i in range(1000):
        (root/f"danbooru2020/{prefix}/{str(i).zfill(4)}").mkdir(exist_ok=True)

def fname2id(fname: str) -> str:
    return str(fname).split("/")[-1].split(".")[0]

def fname2prefix(fname: str) -> str:
    return str(fname).split("/")[-3]

def id2fname(_id, prefix="512white", ext="png", bucket=bucket, front="", back="", root=root):
    _id = str(_id)
    if not type(ext) == str:
        ext = ext(_id)
    if not type(bucket) == str:
        bucket = bucket(_id)
    return str(Path(root)/f"danbooru2020/{prefix}/{bucket}/{front}{_id}{back}.{ext}")

id2 = curry(id2fname)

def budget(N, workers=8):
    def g(n, d):
        if not d <= n:
            return ""
        arr = [n//d]*d
        for i in range(n-(n//d)*d):
            arr[i] += 1
        assert sum(arr) == n
        cumsum = [0]
        for a in arr:
            cumsum.append(cumsum[-1]+a)
        return f"<tr><td>{d}</td>"+"".join(f"<td>{{{s}..{e-1}}}</td>" for s, e in zip(cumsum[:-1], cumsum[1:]))+"</tr>"
    display(HTML("<table>"+"\n".join(g(N, d) for d in range(2, workers+1))+"</table>"))

def split_dump(fnames, name, N=1000, workers=8):
    if N <= 0:
        N = len(fnames)
    (root/f"danbooru2020/dump_{name}").mkdir(exist_ok=True)
    for i in range((len(fnames)-1)//N+1):
        with (root/f"danbooru2020/dump_{name}/{i}").open("w") as f:
            f.write('\n'.join(fnames[i*N:(i+1)*N])+"\n")
    budget((len(fnames)-1)//N+1, workers=8)

def search_dump(txt, name):
    for fname in tqdm(list((root/f"danbooru2020/dump_{name}/").glob("*"))):
        if txt in fname.read_text():
            print(str(fname))
            return

class Filter(dict):
    def __getitem__(self, k):
        if type(k) == int:
            k = list(self.keys())[k]
        return super().__getitem__(k)
    def __setitem__(self, k, v):
        if type(k) == int:
            k = list(self.keys())[k]
        super().__setitem__(k, v)
        i = list(self.keys()).index(k)
        print("{:>3}  {:>9,}  {}".format(i, len(v), k))

def wrap_G(fm, device="cuda"):
    with open(fm, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    def map(seed=1, psi=1):
        label = torch.zeros([1, G.c_dim], device=device)
        if isinstance(seed, Iterable):
            z = torch.from_numpy(np.array([np.random.RandomState(s).randn(G.z_dim) for s in seed])).to(device)
        else:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=psi)
        return w
    def synth(w, synth=True, retarr=False):
        if synth:
            synth_image = G.synthesis(w, noise_mode='const')
        else:
            synth_image = w
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        img = Image.fromarray(synth_image, 'RGB')
        if retarr:
            return img, synth_image
        else:
            return img
    G.map = map
    G.synth = synth
    return G