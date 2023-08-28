import os

__nc1 = os.path.join(os.path.dirname(__file__), "example_1.nc")
__pdb1 = os.path.join(os.path.dirname(__file__), "example_1.pdb")
if os.path.isfile(__nc1) and os.path.isfile(__pdb1):
  traj_pair_1 = (__nc1, __pdb1)

