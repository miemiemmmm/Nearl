import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np

import nearl.features
import nearl.featurizer
from nearl.io.traj import MisatoTraj, Trajectory

# Define the way to generate labels for the database-indexed trajectories


class _PreserveNewlinesHelpFormatter(argparse.HelpFormatter):
    """HelpFormatter that keeps explicit newlines and leading indentation."""

    def _split_lines(self, text, width):
        lines = []
        for paragraph in text.splitlines():
            # Keep the leading whitespace of each line so tabs/spaces used to
            # align columns are not collapsed by textwrap.
            stripped = paragraph.lstrip()
            indent = paragraph[: len(paragraph) - len(stripped)]
            wrapped = super()._split_lines(stripped, width)
            lines.extend(indent + line for line in wrapped)
        return lines


def parser():
    parser = argparse.ArgumentParser(
        description="Featurize a trajectory set for benchmarking",
        formatter_class=_PreserveNewlinesHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--trajlist",
        type=str,
        required=True,
        help="The file containing the list of trajectories",
    )
    parser.add_argument(
        "-m",
        "--database_dir",
        type=str,
        default="",
        help="The directory of the trajectory database (required for --trajlist_format indexed)",
    )
    parser.add_argument(
        "--trajlist_format",
        type=str,
        default="indexed",
        choices=["indexed", "paired"],
        help=(
            "The structure of the trajectory list file. One of:\n"
            "  indexed -> each line is an identifier; trajectories are resolved\n"
            "            relative to --database_dir via MisatoTraj\n"
            "  paired  -> each line is '<trajectory file> <topology file>'\n"
            "            loaded via the generic Trajectory class\n"
        ),
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="", help="The output directory"
    )
    parser.add_argument(
        "-t",
        "--feature_type",
        type=str,
        required=True,
        choices=FEATURE_TYPES,
        help=(
            "Which feature set to benchmark. One of:\n"
            + "\n".join(f"  {k:<10}-> {v}" for k, v in FEATURE_TYPES.items())
            + "\n"
        ),
    )

    # Featurization settings
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=32,
        help="The dimension of the feature vector",
    )
    parser.add_argument(
        "-l", "--length", type=int, default=24, help="The length of the bounding box"
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=5.0,
        help="The cutoff distance for the feature selection",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=1.5,
        help="The sigma value for the feature selection",
    )
    parser.add_argument(
        "-w",
        "--windowsize",
        type=int,
        default=20,
        help="The time window for the feature selection",
    )

    parser.add_argument(
        "--focus_mask",
        type=str,
        default="",
        help=(
            "Atom-selection mask defining the focal point (the ligand).\n"
            "Defaults to ':MOL' for indexed trajlists and ':LIG' for paired ones."
        ),
    )
    parser.add_argument(
        "--h5prefix",
        type=str,
        default="Output",
        help="The prefix of the output h5 file",
    )
    parser.add_argument(
        "--baseline_map",
        type=str,
        default="data/PDBBind_general_v2020.csv",
        help="Path to the PDBBind baseline CSV used for the LabelAffinity feature",
    )
    parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
    parser.add_argument(
        "--task_index", type=int, default=0, help="The task index to run"
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
    return args


def get_trajlist_indexed(trajlist_file, database_dir):
    """Read an identifier-per-line trajlist; entries are resolved against database_dir."""
    with open(trajlist_file) as f:
        identifiers = f.read().strip("\n").split("\n")
    trajlists = [(i, database_dir) for i in identifiers]
    return trajlists


def get_trajlist_paired(trajlist_file):
    """Read a pair-per-line trajlist (one '<traj file> <topology file>' pair per line)."""
    with open(trajlist_file) as f:
        trajlists = [line.split() for line in f.read().splitlines() if line.strip()]
    return trajlists


# ---------------------------------------------------------------------------
# Feature-set enumeration for the -t / --feature_type flag.
#
# Each entry maps a short, documented name to a callable that builds the
# OrderedDict of features to benchmark. This lets you isolate a single feature
# (or a specific selection/aggregation variant) so its cost can be measured on
# its own, instead of always running the full three-feature set together.
#
# The "prot" variants select the protein (all atoms except :MOL).
# ---------------------------------------------------------------------------
FEATURE_TYPES = {
    "mass": "Mass only (static, CPU-side)",
    "mo": "MarchingObservers only (dynamic, GPU)",
    "pdf": "DensityFlow only (dynamic, GPU)",
    "mass_prot": "Mass on the protein (selection='!:MOL')",
    "mo_prot": "MarchingObservers on the protein (selection='!:MOL')",
    "pdf_prot": "DensityFlow on the protein (selection='!:MOL')",
}


def _mass(selection=None, outkey="mass_feat", sigma=1.5):
    kwargs = {"outkey": outkey, "sigma": sigma}
    if selection is not None:
        kwargs["selection"] = selection
    return nearl.features.Mass(**kwargs)


def _mo(
    selection=None, outkey="mobs_feat", obs="mean_distance", agg="standard_deviation"
):
    kwargs = {
        "weight_type": "mass",
        "obs": obs,
        "agg": agg,
        "outkey": outkey,
    }
    if selection is not None:
        kwargs["selection"] = selection
    return nearl.features.MarchingObservers(**kwargs)


def _pdf(selection=None, outkey="pdf_feat", agg="standard_deviation", sigma=1.5):
    kwargs = {"weight_type": "mass", "agg": agg, "outkey": outkey, "sigma": sigma}
    if selection is not None:
        kwargs["selection"] = selection
    return nearl.features.DensityFlow(**kwargs)


def _build_features(sigma, feature_type):
    """Return the OrderedDict of features for the given -t enumeration."""
    features = OrderedDict()

    if feature_type == "mass":
        features["stat"] = _mass(sigma=sigma)
    elif feature_type == "mo":
        features["mo"] = _mo()
    elif feature_type == "pdf":
        features["pdf"] = _pdf(sigma=sigma)
    elif feature_type == "mass_prot":
        features["stat"] = _mass(selection="!:MOL", outkey="mass_prot", sigma=sigma)
    elif feature_type == "mo_prot":
        features["mo"] = _mo(selection="!:MOL", outkey="mobs_prot")
    elif feature_type == "pdf_prot":
        features["pdf"] = _pdf(selection="!:MOL", outkey="pdf_prot", sigma=sigma)
    else:  # pragma: no cover - argparse choices already restrict this
        raise ValueError(f"Unknown feature type: {feature_type}")

    return features


def get_features(sigma, feature_type):
    """Build the feature set for the requested -t enumeration."""
    return _build_features(sigma, feature_type)


if __name__ == "__main__":
    """
  Usage:
  python3 /MieT5/Nearl/scripts/benchmark_dataset.py -f /MieT5/Nearl/data/casf2016_test.txt -o /tmp/ -t pdf -d 32 -m /Matter/misato_database/ -c 2.5 -s 1.5
  """
    nearl.update_config(
        verbose=False,
        debug=False,
    )
    # nearl.update_config(verbose = True, debug = True)

    args = parser()
    args = vars(args)
    print(json.dumps(args, indent=2))
    task_nr = args.get("task_nr")
    task_index = args.get("task_index")
    h5_prefix = args.get("h5prefix")
    feature_type = args.get("feature_type")
    outputfile = os.path.join(
        os.path.abspath(args["output_dir"]), f"{h5_prefix}{task_index}.h5"
    )
    if os.path.exists(outputfile):
        raise ValueError(f"Output file {outputfile} exists. Please remove it first.")

    # Candidate trajectories
    database_dir = args.get("database_dir")
    training_set = args.get("trajlist")
    VOX_cutoff = args.get("cutoff")
    VOX_sigma = args.get("sigma")
    WINDOW_SIZE = args.get("windowsize")

    print(
        f"Input file: {training_set}, Output file: {outputfile}; Task {task_index} of {task_nr}"
    )

    # Initialize featurizer object and register necessary components
    FEATURIZER_PARMS = {
        "dimensions": [args.get("dimension")] * 3,
        "lengths": args.get("length"),
        "time_window": WINDOW_SIZE,
        "outfile": outputfile,
        "cutoff": VOX_cutoff,
        "padding": VOX_cutoff,
        "frame_offset": 9,
    }

    if args.get("trajlist_format") == "paired":
        trajlists = get_trajlist_paired(training_set)
        trajtype = Trajectory
        default_focus = ":LIG"
    else:
        if not database_dir:
            raise ValueError("--database_dir is required for --trajlist_format indexed")
        trajlists = get_trajlist_indexed(training_set, database_dir)
        trajtype = MisatoTraj
        default_focus = ":MOL"
    trajlists = np.array_split(trajlists, task_nr)[task_index]
    trajids = [i[0] for i in trajlists]
    print(f"Total number of trajectories {trajlists.__len__()}")

    ##############################################################
    np.random.seed(0)
    np.random.shuffle(trajlists)
    #  ['6p85' '/Matter/misato_database/']
    #  ['3u6h' '/Matter/misato_database/']
    #  ['2wcx' '/Matter/misato_database/']
    #  ['2qwe' '/Matter/misato_database/']
    #  ['6rih' '/Matter/misato_database/']
    # trajlists, trajids = trajlists[:100], trajids[:100]   # TODO: Remove this line for production run
    ##############################################################
    loader = nearl.io.TrajectoryLoader(
        trajlists, trajtype=trajtype, superpose=True, trajid=trajids
    )
    print(f"Performing the featurization on {len(loader)} trajectories")

    feat = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
    feat.register_trajloader(loader)
    focus_mask = args.get("focus_mask") or default_focus
    feat.register_focus([focus_mask], "mask")

    features = get_features(VOX_sigma, feature_type)

    # Labels
    if trajtype is MisatoTraj:
        features["pk_original"] = nearl.features.LabelAffinity(
            baseline_map=args.get("baseline_map"), outkey="pk_original"
        )
    # features["label_pcdt"] = nearl.features.LabelPCDT(selection=":MOL", baseline_map="/MieT5/Nearl/data/PDBBind_general_v2020.csv", outkey="label_pcdt")
    print(f"There are {len(features)} features registered: {features.keys()}")

    feat.register_features(features)

    # Time only the featurization itself (feat.run()). The CUDA kernels
    # synchronize internally (blocking cudaMemcpy back to host after each
    # kernel), so the elapsed time includes all GPU work. Interpreter
    # startup, argument parsing, trajectory-list setup and HDF5 output-file
    # checks are excluded.
    _t0 = time.perf_counter()
    feat.run()
    _t1 = time.perf_counter()
    print(f"BENCHMARK_RUN_SECONDS={(_t1 - _t0):.6f}")
