# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda"
import pickle
import json
import csv
from pathlib import Path
from typing import List, Tuple
from script_util import csv_read, csv_write
# %% 大局的な条件
MAX = 10**5
MIN = 1024
class model_tag: # script_i2v_factorと名前空間をあわせる
    tags = json.loads(open("/home/natsuki/illustration2vec/tag_list.json", 'r').read())
    index = {t: i for i, t in enumerate(tags)}
root = Path(f"/data/natsuki/factory")
master: List[List[Tuple[float, int]]] = [list() for _ in model_tag.tags]
ok: List[int] = list()
# %%
for k, tag in enumerate(model_tag.tags):
    with open(root/f"{k}.pkl", "rb") as f:
        master[k] = pickle.load(f)
        size = len(master[k])
        assert size == len({seed for _, seed in master[k]}), "seedの重複"
        if MIN < size:
            print(tag, size)
            ok.append(k)
        else:
            pass
# %%
if False:
    "heapqを使う方法から愚直にappendするやり方に切り替える。scoreの符号反転に注意"
    for k, tag in enumerate(model_tag.tags):
        master[k].sort(key=lambda x: x[1])
        master[k] = [(-mscore, seed) for mscore, seed in master[k]]
# %%
for k, tag in enumerate(model_tag.tags):
    csv_write(k, root, master)
assert csv_read(0, root) == master[0], "float比較してるけどTrue返る。すげぇ"
