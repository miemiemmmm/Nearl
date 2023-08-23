import time, os
import importlib.resources as resources

import numpy as np
import pytraj as pt

from nearl.features import fingerprint

print("Running vectorize.py")


pdb_folder = resources.files("nearl").joinpath("data/small_pdbbind")

all_items = os.listdir(pdb_folder)
all_paths = [os.path.join(pdb_folder, item) for item in all_items if os.path.isfile(os.path.join(pdb_folder, item)) and item.endswith(".pdb")]

# Filter out only the files from the list of items

# print(all_paths)

def VECTORIZE(pdbfile):
  st = time.perf_counter()
  traj = pt.load(pdbfile)
  repres = fingerprint.generator(traj)
  repres.center = np.mean(traj.xyz[0], axis=0)
  repres.length = [8 ,8 ,8]
  repres.frame = 0
  slices, segments = repres.slicebyframe()
  feature_vector, mesh_obj, fpfh = repres.vectorize(segments)
  print(f"Vectorization Success: used {time.perf_counter( ) -st:.3f} seconds")

for file in all_paths:
  print(f"Vectorizing {file}")
  VECTORIZE(file)

exit(1)

vectorize = VECTORIZE();

def BENCHMARK(rounds=100):
  from BetaPose import chem
  import timeit, functools
  st = time.perf_counter();
  traj = pt.load(testpdb);

  repres = fingerprint.generator(traj);
  repres.center = np.mean(traj.xyz[0], axis=0);
  repres.length = [8 ,8 ,8];
  repres.frame = 0;

  slices, segments = repres.slicebyframe();
  seg1 = utils.ordersegments(segments)[0]
  idxs = np.where(segments == seg1)[0];

  repres.resmask = utils.getresmask(repres.traj, utils.getmaskbyidx(repres.traj, idxs));
  repres.charges = chem.Chargebytraj(repres.traj, repres.frame, repres.resmask);

  tmp_comb = functools.partial(repres.atom_type_count, idxs);
  t1 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.partial_charge, idxs);
  t2 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.partial_charge, idxs);
  t3 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.pseudo_energy, idxs);
  t4 = timeit.timeit(tmp_comb, number = int(rounds));

  repres.mesh = repres.segment2mesh(idxs)
  tmp_comb = functools.partial(repres.segment2mesh, idxs);
  t5 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.volume, repres.mesh);
  t6 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.surface, repres.mesh);
  t7 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.mean_radius, repres.mesh);
  t8 = timeit.timeit(tmp_comb, number = int(rounds));

  tmp_comb = functools.partial(repres.convex_hull_ratio, repres.mesh);
  t9 = timeit.timeit(tmp_comb, number = int(rounds));
  t_sum = time.perf_counter() - st;
  print(f"{'Atom Types':15s}: {t1:6.3f} {t1 /t_sum *100:8.3f}%")
  print(f"{'Donor/Acceptor':15s}: {t2:6.3f} {t2 /t_sum *100:8.3f}%")
  print(f"{'Partial charge':15s}: {t3:6.3f} {t3 /t_sum *100:8.3f}%")
  print(f"{'Pseudo energy':15s}: {t4:6.3f} {t4 /t_sum *100:8.3f}%")
  print(f"{'Meshify':15s}: {t5:6.3f} {t5 /t_sum *100:8.3f}%")
  print(f"{'Volume':15s}: {t6:6.3f} {t6 /t_sum *100:8.3f}%")
  print(f"{'Surface':15s}: {t7:6.3f} {t7 /t_sum *100:8.3f}%")
  print(f"{'Mean Radius':15s}: {t8:6.3f} {t8 /t_sum *100:8.3f}%")
  print(f"{'Convex Hull':15s}: {t9:6.3f} {t9 /t_sum *100:8.3f}%")
  print(f"Totally used {t_sum:6.3f} seconds ({t_sum/rounds:6.4f} per frame)")



