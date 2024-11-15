import nearl
import nearl.data

# Make up the list of tuple with the PDB files
trajs = [(i,) for i in nearl.data.MINI_SET]
loader = nearl.TrajectoryLoader(trajs)

# The protein and ligands are combined and ligands are named as LIG
for traj in loader:
  print(traj.top.select(":LIG"))
                            
FEATURIZER_PARMS = {
  "dimensions": 32,       # Dimension of the 3D grid
  "lengths": 16,          # Length of the 3D grid in Angstrom, it yields 0.5 resolution
  "time_window": 1,       # !! Number of frames has to be 1 when using static structures
  "sigma": 1.5,
  "cutoff": 3.5,
  "outfile": "/tmp/features.h5",
}
featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)


# Register two simple features
featurizer.register_feature(nearl.features.Mass(selection=":LIG", outkey="lig_annotation"))
featurizer.register_feature(nearl.features.Mass(selection="!:LIG,T3P", outkey="prot_annotation"))  # Append another feature

# Register the trajectory loader, focus and run the featurization
featurizer.register_trajloader(loader)  # Register the trajectory loader in the first step
featurizer.register_focus([":LIG"], "mask")  # focus on the ligand
featurizer.main_loop()
