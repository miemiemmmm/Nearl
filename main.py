import subprocess, os, sys
from BetaPose import representations

msms = "/media/yzhang/MieT5/BetaPose/msms_i86_64Linux2_2.6.1/msms.x86_64Linux2.2.6.1"
inputxyzr = "/tmp/tmp_2f96bb2984_msms.xyzr"
outprefix = "/tmp/tmp_2f96bb2984_test"

d = 4;
r = 1.5;

for i in range(10000):
  subprocess.run([msms, "-if", inputxyzr, "-of", outprefix, "-density", str(d), "-probe_radius", str(r), "-all"]);
  mesh = representations.msms2mesh(f"{outprefix}.vert", f"{outprefix}.face", filename="");
  mesh.get_volume()

