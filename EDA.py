# %%
from typing import *
from  PIL import Image
from pathlib import Path
import json
import pickle
from collections import defaultdict
from vivid.utils import timer
root = Path("/data/natsuki/danbooru2020")
# %%
i = 0
"""
# include

# exclude 

is_deleted	    234,782	Files which are marked as 'deleted' on Danbooru
no_humans	    56,737	Images which do not show humans
not-image	    14,420	Files which are not images (rar, zip, swf, mpg, etc)
photo         	10,161	Files which are photographs, not bitmaps
text_only_page	1,865	Images consisting solely of text
duplicates	    20,187	Duplicated images (1)

"""
# %%
def write(i=0):
    with timer(prefix="read_txts"): # 160[s]
        txts = (root/f"{i}").read_text().rstrip().split("\n")
    with timer(prefix="txts_to_pkl"): # 686[s]
        master_id = dict()
        master_tag = defaultdict(set)
        for txt in txts:
            parsed = json.loads(txt)
            master_id[parsed["id"]] = parsed
            for tag in parsed["tags"]:
                master_tag[tag["name"]].add(parsed["id"])

        with open((root/f"{i}_id.pkl"), "wb") as f:
            pickle.dump(master_id, f, protocol=4)

        with open((root/f"{i}_tag.pkl"), "wb") as f:
            pickle.dump(master_tag, f, protocol=4)

# %%
with timer(prefix="read_id_pkl"): # 311[s]
    with open((root/f"{i}_id.pkl"), "rb") as f:
        master_id = pickle.load(f)
# %%
with timer(prefix="read_tag_pkl"): # 58[s]
    with open((root/f"{i}_tag.pkl"), "rb") as f:
        master_tag = pickle.load(f)
# %%
# https://github.com/fire-eggs/Danbooru2019
# これよりちょっと増えてる
len(master_tag["touhou"])
len(master_tag["hatsune_miku"])
# %%
len(master_tag) # 434947
# %% 
# https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/examples/Image%20Browser.ipynb
import matplotlib.pyplot as plt
from ipywidgets import interact
from sklearn import datasets
digits = datasets.load_digits()
def browse_images(digits):
    n = len(digits.images)
    def view_image(i):
        plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %s' % digits.target[i])
        plt.show()
    interact(view_image, i=(0,n-1))
browse_images(digits)
# %%
def bucket(_id):
    return _id[-3:].zfill(4)