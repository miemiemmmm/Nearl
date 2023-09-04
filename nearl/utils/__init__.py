import importlib

UTILS_ACTIONS = [
  "get_hash",
  "dist_caps",
  "fetch",
  "get_mask",
  "get_residue_mask",
  "get_mask_by_idx",
  "get_protein_mask",
  "color_steps",
  "msd",
  "mscv",
  "filter_points_within_bounding_box",
  "_entropy",
  "get_pdb_title",
  "get_pdb_seq",
  "data_from_fbag",
  "data_from_tbag",
  "data_from_tbagresults",
  "data_from_fbagresults",
  "conflict_factor",
]

# Only import functions that are pre-defined in the UTILS_ACTIONS (important actions)
for func_name in UTILS_ACTIONS:
  imported_function = importlib.import_module(f'nearl.utils.utils').__dict__.get(func_name)
  if imported_function is not None:
    globals()[func_name] = imported_function
  else:
    print(f"Function {func_name} not found.")

# Import from nearl.utils.draw modules;
DRAW_ACTIONS = [
  "draw_grid_mols"
]

for func_name in DRAW_ACTIONS:
  imported_function = importlib.import_module(f'nearl.utils.draw').__dict__.get(func_name)
  if imported_function is not None:
    globals()[func_name] = imported_function
  else:
    print(f"Function {func_name} not found.")

CHEM_ACTIONS = [
  "combine_molpdb",
  "correct_mol_by_smiles",
  "sanitize_bond",
  "traj_to_rdkit",
  "write_pdb_block",

]

for func_name in CHEM_ACTIONS:
  imported_function = importlib.import_module(f'nearl.utils.chemtools').__dict__.get(func_name)
  if imported_function is not None:
    globals()[func_name] = imported_function
  else:
    print(f"Function {func_name} not found.")


TRANSFORM_ACTIONS = [
  "transform_pcd",
  "tm_euler",
  "tm_quaternion",
]

for func_name in TRANSFORM_ACTIONS:
  imported_function = importlib.import_module(f'nearl.utils.transform').__dict__.get(func_name)
  if imported_function is not None:
    globals()[func_name] = imported_function
  else:
    print(f"Function {func_name} not found.")

VIEW_ACTIONS = [
  "display_config",
  "random_color",
  "nglview_mask",
  "TrajectoryViewer",
]
for func_name in VIEW_ACTIONS:
  imported_function = importlib.import_module(f'nearl.utils.view').__dict__.get(func_name)
  if imported_function is not None:
    globals()[func_name] = imported_function
  else:
    print(f"Function {func_name} not found.")

####################################################################
# cluster, session_prep and other are not fully developed yet
####################################################################