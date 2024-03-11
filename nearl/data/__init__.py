import os


if os.path.isfile(os.path.join(os.path.dirname(__file__), "example_1.nc")) and os.path.isfile(os.path.join(os.path.dirname(__file__), "example_1.pdb")):
  MINI_TRAJ = (os.path.join(os.path.dirname(__file__), "example_1.nc"), os.path.join(os.path.dirname(__file__), "example_1.pdb"))

MINI_SET = os.path.join(os.path.dirname(__file__), "small_pdbbind")


REFINED_SET = os.path.join(os.path.dirname(__file__), "PDBBind_refined_v2020.csv")
# REFINED_SET = os.path.join(os.path.dirname(__file__), "PDBBind_refined_v2020.csv")
GENERAL_SET = os.path.join(os.path.dirname(__file__), "PDBBind_general_v2020.csv")
# GENERAL_SET = os.path.join(os.path.dirname(__file__), "PDBBind_general_v2020.csv")

