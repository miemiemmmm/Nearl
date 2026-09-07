"""Benchmark numpy-return commands vs the *_dlpack GPU-output commands.

Measures two things per feature family:
1. produce: wall time of the command itself
2. to_gpu:  wall time to get the result onto the GPU as a float32 torch tensor
           (torch.from_numpy(...).cuda() for the numpy path, no-op for dlpack)

Run with:
  CUDA_PREFIX=$(spack location -i cuda)
  export LD_LIBRARY_PATH="$CUDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
  python benchmarks/benchmark_gpu_output.py
"""
import time

import numpy as np
import torch

from nearl import commands

np.random.seed(0)

ATOM_NR = 300
FRAME_NR = 50
DIMS = np.array([32, 32, 32], dtype=np.int32)
SPACING = 1.0
CUTOFF = 2.5
SIGMA = 1.0
REPEATS = 200
WARMUP = 20


def timeit(fn, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0


def main():
    commands.init_context()
    assert commands.context_valid()

    coords = np.random.normal(size=(ATOM_NR, 3), loc=5, scale=1).astype(np.float32)
    weights1 = np.full((ATOM_NR,), 1.5, dtype=np.float32)
    traj = np.random.normal(size=(FRAME_NR, ATOM_NR, 3), loc=5, scale=2).astype(np.float32)
    weights_t = np.full((FRAME_NR * ATOM_NR,), 16.0, dtype=np.float32)

    rows = []

    def add_row(name, np_fn, dl_fn):
        t_np = timeit(np_fn)
        t_np_gpu = timeit(lambda: torch.from_numpy(np_fn()).cuda())
        t_dl = timeit(dl_fn)
        rows.append((name, t_np, t_np_gpu, t_dl))
        print(f"{name:20s} numpy={t_np:8.3f} ms  numpy+to_gpu={t_np_gpu:8.3f} ms  "
              f"dlpack={t_dl:8.3f} ms  speedup(vs to_gpu)={t_np_gpu / t_dl:5.2f}x")

    add_row(
        "frame_voxelize",
        lambda: commands.frame_voxelize(coords, weights1, DIMS, SPACING, CUTOFF, SIGMA),
        lambda: commands.frame_voxelize_dlpack(coords, weights1, DIMS, SPACING, CUTOFF, SIGMA),
    )
    add_row(
        "marching_observer",
        lambda: commands.marching_observer(traj, weights_t, DIMS, SPACING, CUTOFF, 1, 1),
        lambda: commands.marching_observer_dlpack(traj, weights_t, DIMS, SPACING, CUTOFF, 1, 1),
    )
    add_row(
        "density_flow",
        lambda: commands.density_flow(traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1),
        lambda: commands.density_flow_dlpack(traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1),
    )

    # Correctness cross-check on the benchmark data
    ref = commands.density_flow(traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1)
    out = commands.density_flow_dlpack(traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1)
    assert np.allclose(ref, out.cpu().numpy(), rtol=1e-4, atol=1e-4), "density_flow mismatch"
    print("correctness: density_flow_dlpack matches density_flow")

    with open(".benchmarks/05_goal_b_gpu_output.txt", "w") as f:
        f.write("# Goal (b) benchmark: numpy output vs *_dlpack GPU output\n")
        f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}, {REPEATS} repeats after {WARMUP} warmup\n")
        f.write(f"# data: {ATOM_NR} atoms, {FRAME_NR} frames, {tuple(DIMS)} grid, "
                f"spacing {SPACING}, cutoff {CUTOFF}, sigma {SIGMA}, context active\n")
        f.write("#\n")
        f.write(f"{'feature':20s} {'numpy ms':>10s} {'np+to_gpu ms':>13s} {'dlpack ms':>10s} "
                f"{'speedup':>8s}\n")
        for name, t_np, t_np_gpu, t_dl in rows:
            f.write(f"{name:20s} {t_np:10.3f} {t_np_gpu:13.3f} {t_dl:10.3f} "
                    f"{t_np_gpu / t_dl:7.2f}x\n")
    print("saved .benchmarks/05_goal_b_gpu_output.txt")


if __name__ == "__main__":
    main()
