"""Benchmark numpy-return commands vs the *_dlpack GPU-output commands.

Three destinations per feature family:
1. numpy:        the command's own wall time, result on the host
2. numpy+to_gpu: plus torch.from_numpy(...).cuda(), i.e. what a model actually needs
3. dlpack:       written straight into CUDA memory, allocated per call
4. dlpack out=:  written into a caller-owned tensor, no allocation at all

The out= column is the one to read for a training loop: the grid lands in a slice
of a batch tensor that torch's caching allocator already owns.

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
SWEEP = [16, 32, 48, 64, 96, 128]


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
    traj = np.random.normal(size=(FRAME_NR, ATOM_NR, 3), loc=5, scale=2).astype(
        np.float32
    )
    weights_t = np.full((FRAME_NR * ATOM_NR,), 16.0, dtype=np.float32)

    rows = []

    buffer = torch.empty(tuple(DIMS), dtype=torch.float32, device="cuda")

    def add_row(name, np_fn, dl_fn):
        t_np = timeit(np_fn)
        t_np_gpu = timeit(lambda: torch.from_numpy(np_fn()).cuda())
        t_dl = timeit(dl_fn)
        t_out = timeit(lambda: dl_fn(out=buffer))
        rows.append((name, t_np, t_np_gpu, t_dl, t_out))
        print(
            f"{name:20s} numpy={t_np:8.3f} ms  numpy+to_gpu={t_np_gpu:8.3f} ms  "
            f"dlpack={t_dl:8.3f} ms  dlpack_out={t_out:8.3f} ms  "
            f"speedup(vs to_gpu)={t_np_gpu / t_out:5.2f}x"
        )

    add_row(
        "frame_voxelize",
        lambda: commands.frame_voxelize(coords, weights1, DIMS, SPACING, CUTOFF, SIGMA),
        lambda **kw: commands.frame_voxelize_dlpack(
            coords, weights1, DIMS, SPACING, CUTOFF, SIGMA, **kw
        ),
    )
    add_row(
        "marching_observer",
        lambda: commands.marching_observer(
            traj, weights_t, DIMS, SPACING, CUTOFF, 1, 1
        ),
        lambda **kw: commands.marching_observer_dlpack(
            traj, weights_t, DIMS, SPACING, CUTOFF, 1, 1, **kw
        ),
    )
    add_row(
        "density_flow",
        lambda: commands.density_flow(traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1),
        lambda **kw: commands.density_flow_dlpack(
            traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1, **kw
        ),
    )

    # Correctness cross-check on the benchmark data
    ref = commands.density_flow(traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1)
    grid = commands.density_flow_dlpack(
        traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1
    )
    assert np.allclose(ref, torch.from_dlpack(grid).cpu().numpy(), rtol=1e-4, atol=1e-4)
    commands.density_flow_dlpack(
        traj, weights_t, DIMS, SPACING, CUTOFF, SIGMA, 1, out=buffer
    )
    assert np.allclose(ref, buffer.cpu().numpy(), rtol=1e-4, atol=1e-4)
    print("correctness: both dlpack destinations match density_flow")

    # The two destinations trade off with grid size: the copies the numpy path
    # makes scale with the grid, while the DLPack import is a fixed ~17 us of
    # protocol per call. Sweep to find where each one wins.
    print(f"\n{'density_flow across grid sizes':50s}")
    print(
        f"{'dim':>5} {'np+to_gpu ms':>13s} {'dlpack ms':>10s} {'dlpack out= ms':>15s} "
        f"{'best':>12s}"
    )
    sweep_rows = []
    for dim in SWEEP:
        grid = np.array([dim] * 3, dtype=np.int32)
        out = torch.empty((dim, dim, dim), dtype=torch.float32, device="cuda")

        def np_fn(g=grid):
            return commands.density_flow(traj, weights_t, g, SPACING, CUTOFF, SIGMA, 1)

        t_np_gpu = timeit(lambda f=np_fn: torch.from_numpy(f()).cuda())
        t_dl = timeit(
            lambda g=grid: commands.density_flow_dlpack(
                traj, weights_t, g, SPACING, CUTOFF, SIGMA, 1
            )
        )
        t_out = timeit(
            lambda g=grid, o=out: commands.density_flow_dlpack(
                traj, weights_t, g, SPACING, CUTOFF, SIGMA, 1, out=o
            )
        )
        best = min(
            ("np+to_gpu", t_np_gpu),
            ("dlpack", t_dl),
            ("out=", t_out),
            key=lambda pair: pair[1],
        )
        sweep_rows.append((dim, t_np_gpu, t_dl, t_out, best[0]))
        print(f"{dim:5d} {t_np_gpu:13.3f} {t_dl:10.3f} {t_out:15.3f} {best[0]:>12s}")

    with open(".benchmarks/05_goal_b_gpu_output.txt", "w") as f:
        f.write("# Goal (b) benchmark: numpy output vs *_dlpack GPU output\n")
        f.write(
            f"# {time.strftime('%Y-%m-%d %H:%M:%S')}, {REPEATS} repeats after {WARMUP} warmup\n"
        )
        f.write(
            f"# data: {ATOM_NR} atoms, {FRAME_NR} frames, {tuple(DIMS)} grid, "
            f"spacing {SPACING}, cutoff {CUTOFF}, sigma {SIGMA}, context active\n"
        )
        f.write("#\n")
        f.write(
            f"{'feature':20s} {'numpy ms':>10s} {'np+to_gpu ms':>13s} {'dlpack ms':>10s} "
            f"{'dlpack out= ms':>15s} {'speedup':>8s}\n"
        )
        for name, t_np, t_np_gpu, t_dl, t_out in rows:
            f.write(
                f"{name:20s} {t_np:10.3f} {t_np_gpu:13.3f} {t_dl:10.3f} "
                f"{t_out:15.3f} {t_np_gpu / t_out:7.2f}x\n"
            )
        f.write(f"\n# density_flow across grid sizes, {tuple(DIMS)} data above\n")
        f.write(
            f"{'dim':>5} {'np+to_gpu ms':>13s} {'dlpack ms':>10s} "
            f"{'dlpack out= ms':>15s} {'best':>12s}\n"
        )
        for dim, t_np_gpu, t_dl, t_out, best in sweep_rows:
            f.write(
                f"{dim:5d} {t_np_gpu:13.3f} {t_dl:10.3f} {t_out:15.3f} {best:>12s}\n"
            )
    print("saved .benchmarks/05_goal_b_gpu_output.txt")


if __name__ == "__main__":
    main()
