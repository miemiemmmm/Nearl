import time, os
import numpy as np

from nearl.features import fingerprint
from nearl.data import MINI_SET
from nearl.io import TrajectoryLoader


MINI_SET_FILES = os.listdir(MINI_SET)
MINI_SET_PATHS = [os.path.join(MINI_SET, item) for item in MINI_SET_FILES if os.path.isfile(os.path.join(MINI_SET, item)) and item.endswith(".pdb")]

def do_vectorize():
  loader = TrajectoryLoader(MINI_SET_PATHS, MINI_SET_PATHS)
  c = 0
  for traj in loader:
    st = time.perf_counter()
    repres = fingerprint.generator(traj)
    repres.center = np.mean(traj.xyz[0], axis=0)
    repres.length = [8 ,8 ,8]
    repres.frame = 0
    segments = repres.query_segments()
    feature_vector, mesh = repres.vectorize(segments)
    c += 1
    print(f"Vectorization Success: Case {c:>3d} took {time.perf_counter( ) -st:.3f} seconds")

def doben(rounds=100):
  from nearl import utils, io
  import timeit, functools
  st = time.perf_counter()
  traj = io.Trajectory(MINI_SET_PATHS[0])

  repres = fingerprint.generator(traj)
  repres.center = np.mean(traj.xyz[0], axis=0)
  repres.length = [8 ,8 ,8]
  repres.frame = 0

  segments = repres.query_segments()
  seg1 = utils.order_segments(segments)[0]
  idxs = np.where(segments == seg1)[0]

  repres.resmask = utils.get_residue_mask(repres.traj, utils.get_mask_by_idx(repres.traj, idxs))
  repres.charges = utils.chemtools.Chargebytraj(repres.traj, repres.frame, repres.resmask)

  tmp_comb = functools.partial(repres.atom_type_count, idxs)
  t1 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.partial_charge, idxs)
  t2 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.partial_charge, idxs)
  t3 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.pseudo_energy, idxs)
  t4 = timeit.timeit(tmp_comb, number = int(rounds))

  repres.mesh = repres.segment_to_mesh(idxs)
  tmp_comb = functools.partial(repres.segment_to_mesh, idxs)
  t5 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.volume, repres.mesh)
  t6 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.surface, repres.mesh)
  t7 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.mean_radius, repres.mesh)
  t8 = timeit.timeit(tmp_comb, number = int(rounds))

  tmp_comb = functools.partial(repres.convex_hull_ratio, repres.mesh)
  t9 = timeit.timeit(tmp_comb, number = int(rounds))
  t_sum = time.perf_counter() - st
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



