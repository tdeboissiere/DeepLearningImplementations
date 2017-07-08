import glob
import shlex
import subprocess
from natsort import natsorted


list_files = glob.glob("figures/*")
list_files_20pts = natsorted([f for f in list_files if "20_npts" in f])
list_files_100pts = natsorted([f for f in list_files if "100_npts" in f])

str_20pts = " ".join(list_files_20pts)
str_100pts = " ".join(list_files_100pts)

cmd = "convert -delay 80 -resize 600x600 -loop 0 %s figures/tang_20pts.gif" % str_20pts
subprocess.call(shlex.split(cmd))

cmd = "convert -delay 80 -resize 600x600 -loop 0 %s figures/tang_100pts.gif" % str_100pts
subprocess.call(shlex.split(cmd))
