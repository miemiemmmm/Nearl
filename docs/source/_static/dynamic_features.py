# Dynamic features: property density flow and marching observers
################################################################

import h5py

import nearl
import nearl.features
import nearl.featurizer
import nearl.io

FEATURIZER_PARMS = {
    "dimensions": 32,  # Dimension of the 3D grid
    "lengths": 16,  # Length of the 3D grid in Angstrom, it yields 0.5 resolution
    "time_window": 10,  # Number of frames aggregated into one dynamic feature
    "sigma": 1.5,  # Width of the Gaussian used by the density flow
    "cutoff": 3.5,  # Density flow: Gaussian cutoff; observers: observation radius
    "outfile": "/tmp/dynamic_features.h5",
}

LIGAND = ":LIG"
POCKET = "!(:LIG,T3P)"

if __name__ == "__main__":
    EXAMPLE_DATA = nearl.get_example_data("/tmp/nearl_test")
    loader = nearl.io.TrajectoryLoader(EXAMPLE_DATA["MINI_TRAJSET"], mask="!:T3P")
    featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

    featurizer.register_features(
        [
            # Where the ligand mass sits on average over the 10-frame window
            nearl.features.DensityFlow(
                selection=LIGAND,
                weight_type="mass",
                agg="mean",
                outkey="df_mass_mean",
            ),
            # Where that mass density fluctuates the most
            nearl.features.DensityFlow(
                selection=LIGAND,
                weight_type="mass",
                agg="standard_deviation",
                outkey="df_mass_std",
            ),
            # The same question asked of the pocket's aromatic atoms
            nearl.features.DensityFlow(
                selection=POCKET,
                weight_type="aromaticity",
                agg="mean",
                outkey="df_arom_mean",
            ),
            # How crowded each observer is, averaged over the window
            nearl.features.MarchingObservers(
                selection=LIGAND,
                obs="density",
                weight_type="mass",
                agg="mean",
                outkey="obs_density_mean",
            ),
            # How many different atoms an observer ever saw
            nearl.features.MarchingObservers(
                selection=LIGAND,
                obs="distinct_count",
                weight_type="atomic_id",
                agg="max",
                outkey="obs_distinct_max",
            ),
            # Whether the pocket atoms an observer sees spread out or close in
            nearl.features.MarchingObservers(
                selection=POCKET,
                obs="radius_of_gyration",
                weight_type="mass",
                agg="drift",
                outkey="obs_rog_drift",
            ),
        ]
    )

    featurizer.register_trajloader(loader)
    featurizer.register_focus([LIGAND], "mask")
    featurizer.run()

    # Every feature lands in the same HDF5 file under its own outkey
    with h5py.File(FEATURIZER_PARMS["outfile"], "r") as hdf:
        keys = [k for k in hdf if k != "featurizer_parms"]
        print(f"\n{'outkey':<20}{'shape':>20}{'min':>10}{'max':>10}")
        for key in sorted(keys):
            arr = hdf[key][:]
            print(f"{key:<20}{arr.shape!s:>20}{arr.min():>10.3f}{arr.max():>10.3f}")
