from typing import *
from IPython.display import HTML, Javascript, display
import ipywidgets as wi
from pathlib import Path
from random import Random
import pickle
from tqdm.notebook import tqdm
from pickle import Unpickler
from functools import lru_cache
import matplotlib.pyplot as plt

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

def id2fname(_id, prefix="512white", ext="png", bucket=bucket, root=root):
    _id = str(_id)
    if not type(ext) == str:
        ext = ext(_id)
    if not type(bucket) == str:
        bucket = bucket(_id)
    return str(Path(root)/f"danbooru2020/{prefix}/{bucket}/{_id}.{ext}")

class CurriedFunc:
    def __init__(self, func, *args):
        self.func = func
        self.args = args
    def __getattr__(self, arg):
        return self.__getitem__(arg)
    def __getitem__(self, arg):
        return CurriedFunc(self.func, *self.args, arg)
    def __call__(self, *args, **kwargs):
        return self.func(*args, *self.args, **kwargs)
    def __str__(self):
        return f"f({', '.join(['・']+list(self.args))})"
    def __repr__(self):
        return self.__str__()
id2 = CurriedFunc(id2fname)

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

class TQDMBytesReader(object):
    def __init__(self, fd, tqdm, total, desc=''):
        self.fd = fd
        self.tqdm = tqdm(total=total)
        self.tqdm.set_description(desc)
    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.tqdm.update(len(bytes))
        return bytes
    def readline(self):
        bytes = self.fd.readline()
        self.tqdm.update(len(bytes))
        return bytes
    def __enter__(self):
        self.tqdm.__enter__()
        return self
    def __exit__(self, *args, **kwargs):
        return self.tqdm.__exit__(*args, **kwargs)

def tqdm_load(fname, tqdm=tqdm, desc=''):
    with open(fname, "rb") as fd:
         total = Path(fname).stat().st_size
         with TQDMBytesReader(fd, tqdm=tqdm, total=total, desc=desc) as pbfd:
             up = Unpickler(pbfd)
             obj = up.load()
    return obj