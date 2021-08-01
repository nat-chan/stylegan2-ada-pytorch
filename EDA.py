# %%
from typing import *
import matplotlib.pyplot as plt
import random
from typing import *
from  PIL import Image
from pathlib import Path
import json
import pickle
from collections import defaultdict
from vivid.utils import timer
import subprocess
from subprocess import PIPE
from multiprocessing import Pool
from tqdm import tqdm
from time import time
root = Path("/data/natsuki/danbooru2020")
old = set(Path("/data/natsuki/danbooru2019/512px.dat").read_text().strip().split())
safe =  set((root/"512px.dat").read_text().strip().split())
i = 0
# %%
"""
https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/examples/Image%20Browser.ipynb
https://github.com/fire-eggs/Danbooru2019
https://twitter.com/ak92501/status/1193674395463622656

# include
standing
full_body
high_res

# exclude 
is_deleted	    234,782	Files which are marked as 'deleted' on Danbooru
no_humans	    56,737	Images which do not show humans
not-image	    14,420	Files which are not images (rar, zip, swf, mpg, etc)
photo         	10,161	Files which are photographs, not bitmaps
text_only_page	1,865	Images consisting solely of text
duplicates	    20,187	Duplicated images (1)


len(tags) # 434947

standing  339801
full_body 284069
積         78734
highres  1854115
積         45763
- 
no_humans  55626
text_only_page 1877
差         45046

len(v0)=76420
len(v1)=45046
monochrome 390979
差
len(v0)=73983
len(v1)=43794

len(old&v0) 53347
len(old&v1) 29197

is_deleted ?
not-image  ?
photo ? photoなんちゃらがいっぱい
duplicates ?

512pxにあるもの
34087 -> 30531

    | tags["character_name"]
38477 -> 35915

simple_background
white_background

armor
4277725
4261497

1680309 細長
1114782 パケ
1133081 もじ入り character_name
1536047 ふで文字入り
1625537 これもキャラ名？
1161107 めっちゃ遠くから
2315298 photo
2457006 二人写ってる
SDキャラ
完全に後ろを向いている状態
ルカリオ
2930175 ウマ人間
ふりかえる
鎧
車
デジタルネイティブじゃない

    | tags["weapon"] で
len(v0)=32634
len(v1)=19333
"""
# %%
def myplot(path :Path):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Image.open(path))
    plt.show()
# %%
def write(i=0):
    with timer(prefix="read_txts"): # 160[s]
        txts = (root/f"{i}").read_text().rstrip().split("\n")
    with timer(prefix="txts_to_pkl"): # 686[s]
        ids = dict()
        tags = defaultdict(set)
        for txt in txts:
            parsed = json.loads(txt)
            ids[parsed["id"]] = parsed
            for tag in parsed["tags"]:
                tags[tag["name"]].add(parsed["id"])

        with open((root/f"{i}_id.pkl"), "wb") as f:
            pickle.dump(ids, f, protocol=4)

        with open((root/f"{i}_tag.pkl"), "wb") as f:
            pickle.dump(tags, f, protocol=4)

# %%
with timer(prefix="read_id_pkl"): # 311[s]
    with open((root/f"{i}_id.pkl"), "rb") as f:
        ids = pickle.load(f)
# %%
with timer(prefix="read_tag_pkl"): # 58[s]
    with open((root/f"{i}_tag.pkl"), "rb") as f:
        tags = pickle.load(f)
# %%
# 
# これよりちょっと増えてる
len(tags["touhou"])
len(tags["hatsune_miku"])
# %%

# %% 
# %%
def bucket(_id):
    return _id[-3:].zfill(4)
# %%

"""
"""
v0 = (
    ( safe
    & tags["standing"]
    & tags["full_body"]
    &   ( tags["simple_background"]
        | tags["white_background"]
        | tags["black_background"]
        | tags["transparent_background"]
        | tags["grey_background"]
        | tags["red_background"]
        | tags["blue_background"]
        | tags["green_background"]
        | tags["yellow_background"]
        | tags["pink_background"]
        | tags["brown_background"]
        | tags["purple_background"]
        | tags["orange_background"]
        | tags["beige_background"]
        | tags["two-tone_background"]
        | tags["gradient_background"]
        | tags["checkered_background"]
        | tags["aqua_background"]
        )
    & ( tags["1girl"] | tags["solo"] )
    ) -
    ( tags["no_humans"]
    | tags["monochrome"]
    | tags["text_only_page"]
    | tags["photo-referenced"]
    | tags["octoling"]
    | tags["1boy"]
    | tags["2boys"]
    | tags["3boys"]
    | tags["4boys"]
    | tags["5boys"]
    | tags["6+boys"]
    | tags["multiple_boys"]
    | tags["car"]
    | tags["bicycle"]
    | tags["back"]
    | tags["full_armor"]
    | tags["arachne"]
    | tags["multiple_views"]
    | tags["robot"]
    | tags["pokemon_(creature)"]
    | tags["lineart"]
    | tags["variations"]
    | tags["zoom_layer"]
    | tags["ass"]
    | tags["from_behind"]
    )
)
v1 = v0 & tags["highres"]
print(f"{len(v0)=}")
print(f"{len(v1)=}")
# %%
ver = "v2"
subprocess.call(["rm", "-rf", root/ver, root/f"{ver}.txt"])
subprocess.call(["mkdir", "-p", root/ver])
with (root/f"{ver}.txt").open("w") as f:
    f.writelines(str(root/"512px"/bucket(_id)/f"{_id}.jpg")+"\n" for _id in v1)

# %%
for tag in ids["2090603"]["tags"]:
    print(tag["name"])
# %%
#for tag in ids["2315298"]["tags"]:
for tag in ids["324098"]["tags"]:
    print(tag["name"])
# %%
tag_list = [k for k in tags]
tag_list.sort(key=lambda x: len(tags[x]), reverse=True)
# %%
# %%
A = v1&tags["from_behind"]
#A = v1-v2
A = list(A)
random.shuffle(A)
_id = A[0]
image = Image.open(str(root/"512px"/bucket(_id)/f"{_id}.jpg"))
print(_id)
image

# %% displayfr(image)
for k in range(1):
    pass

# %%
for k in tags:
    if "boys" in k:
        print(k)


# %%  background

bg0 = (
    ( safe
    & tags["standing"]
    & tags["full_body"]
    & tags["highres"]
    & ( tags["1girl"] | tags["solo"] )
    ) -
    ( tags["no_humans"]
    | tags["monochrome"]
    | tags["text_only_page"]
    | tags["photo-referenced"]
    | tags["octoling"]
    | tags["1boy"]
    | tags["2boys"]
    | tags["3boys"]
    | tags["4boys"]
    | tags["5boys"]
    | tags["6+boys"]
    | tags["multiple_boys"]
    | tags["car"]
    | tags["bicycle"]
    | tags["back"]
    | tags["full_armor"]
    | tags["arachne"]
    | tags["multiple_views"]
    | tags["robot"]
    | tags["pokemon_(creature)"]
    | tags["lineart"]
    | tags["variations"]
    | tags["zoom_layer"]
    | tags["ass"]
    | tags["from_behind"]
    )
)
# %%
yes = set("""
simple_background
white_background
grey_background
gradient_background
transparent_background
blue_background
pink_background
black_background
yellow_background
two-tone_background
red_background
brown_background
purple_background
green_background
orange_background
beige_background
""".strip().split())
no = set("""
blurry_background
photo_background
""".strip().split())
# aqua_background
# polka_dot_background
# floral_background
# striped_background
# checkered_background
# starry_background
# multicolored_background
# %% 背景タグの一覧
background_tags = list()
for tag in tags:
    if "background" in tag:
        background_tags.append( (len(tags[tag]), tag) )
background_tags.sort(reverse=True)
for num, tag in background_tags:
    if tag in yes:
        continue
    if tag in no:
        continue
    print(tag, num)
    A = bg0&tags[tag]
    A = list(A)
    random.shuffle(A)
    _id = A[0]
    myplot(root/"512px"/bucket(_id)/f"{_id}.jpg")
# %%
