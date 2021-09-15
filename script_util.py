from typing import *
from IPython.display import HTML, display
from pathlib import Path
from tqdm.notebook import tqdm
from pickle import Unpickler
import matplotlib.pyplot as plt
from nokogiri.curry import curry
from nokogiri.tqdm_load import tqdm_load

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

def id2fname(_id, prefix="512white", ext="png", int="", bucket=bucket, root=root):
    _id = str(_id)
    if not type(ext) == str:
        ext = ext(_id)
    if not type(bucket) == str:
        bucket = bucket(_id)
    return str(Path(root)/f"danbooru2020/{prefix}/{bucket}/{int}{_id}.{ext}")

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