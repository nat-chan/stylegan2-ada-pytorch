from pathlib import Path
import numpy as np

items = ["white55", "white55_std0"]
for item in items:
    root = Path(f"/home/natsuki/pixel2style2pixel/examples/{item}")
    flist = sorted(root.glob("*.txt"))

    score = np.array([float(fname.read_text()) for fname in flist])
    μ = score.mean()
    σ = score.std()

    print(root.stem, f"{μ=:.2f}, {σ=:.2f}")