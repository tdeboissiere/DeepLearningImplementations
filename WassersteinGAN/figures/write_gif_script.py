import glob
from natsort import natsorted

list_toy = natsorted(glob.glob("*toy_dataset_iter*"))
list_toy_str = " ".join(list_toy)

with open("make_gif.sh", "w") as f:

    cmd = "convert -delay 15 -resize 300x300 -loop 0 %s MoG_dataset.gif" % list_toy_str
    f.write(cmd)
