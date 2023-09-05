import os
import pandas as pd
_nc1 = os.path.join(os.path.dirname(__file__), "example_1.nc")
_pdb1 = os.path.join(os.path.dirname(__file__), "example_1.pdb")
if os.path.isfile(_nc1) and os.path.isfile(_pdb1):
  MINI_TRAJ = (_nc1, _pdb1)
  del _nc1, _pdb1

MINI_SET = os.path.join(os.path.dirname(__file__), "small_pdbbind")


REFINED_SET = pd.read_csv(os.path.join(os.path.dirname(__file__), "PDBBind_refined_v2020.csv"))
# REFINED_SET = os.path.join(os.path.dirname(__file__), "PDBBind_refined_v2020.csv")
GENERAL_SET = pd.read_csv(os.path.join(os.path.dirname(__file__), "PDBBind_general_v2020.csv"))
# GENERAL_SET = os.path.join(os.path.dirname(__file__), "PDBBind_general_v2020.csv")

MINI_PARMS = {
  # POCKET SETTINGS
  "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
  "CUBOID_LENGTH": [24, 24, 24],     # Unit: Angstorm (Need scaling)
}