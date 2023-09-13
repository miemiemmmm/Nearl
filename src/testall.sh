#!/bin/bash -l

python3 -c "import interpolate; import numpy as np; ret = interpolate.sum_array(np.arange(100, dtype=np.int32)); print(f'sum is {ret}')"

python3 -c "import interpolate; import numpy as np; [interpolate.sum_array(np.arange(100, dtype=np.int32)) for i in range(1000)] "

python3 -c """import time;
import interpolate;
import numpy as np;
np.random.seed(1);
coord = np.random.normal(5,1, size=(1000,3));
grid = np.meshgrid(
  np.linspace(0,21,85),
  np.linspace(0,21,85),
  np.linspace(0,21,85),
  indexing='ij'
);
grid_coord = np.column_stack([g.ravel() for g in grid]);
weights = np.ones(1000).astype(np.float64);
print(coord.shape, grid_coord.shape, weights.shape);
st_cu = time.perf_counter();
ret = interpolate.interpolate(grid_coord, coord , weights).round(3);
print('CUDA: ', ret.shape, ret.sum(), time.perf_counter() - st_cu);
st_acc = time.perf_counter();
_ret = interpolate.interpolate_acc(grid_coord, coord , weights).round(3);
print('OpenACC: ', _ret.shape, _ret.sum(), time.perf_counter() - st_acc);
print('CUDA == OpenACC: ', False not in np.isclose(ret.round(3), _ret.round(3)));
mask = np.where(np.isclose(ret.round(3), _ret.round(3)) == False);
if (np.count_nonzero(mask[0])): print(ret[mask], _ret[mask]);
"""

python3 -c """
import time;
import numpy as np;
import nearl.utils;
import interpolate;
testarr = np.array([1,1,1,1,1,1,1,1.2]);
st0 = time.perf_counter();
ret = interpolate.entropy(testarr);
diff0 = time.perf_counter() - st0;
st1 = time.perf_counter();
ret1 = nearl.utils.entropy(testarr);
diff1 = time.perf_counter() - st1;
print(f'The entropy: From C++ {ret:.3f} ({diff0*1e6:.1f} us); From Python: {ret:.3f} ({diff1*1e6:.1f} us)');

ret = interpolate.gaussian(np.array(0));
print(f'Gaussian of 0: {ret}');
"""

python3 -c """
import time;
import interpolate;
import numpy as np;
np.random.seed(1);
coord = np.random.normal(5,1, size=(1000,3));
grid = np.meshgrid(
  np.linspace(0,21,85),
  np.linspace(0,21,85),
  np.linspace(0,21,85),
  indexing='ij'
);
grid_coord = np.column_stack([g.ravel() for g in grid]);
weights = np.ones(1000).astype(np.float64);
st_cu = time.perf_counter();
ret = interpolate.query_grid_entropy(grid_coord, coord , weights).round(3);
print(f'CUDA query grid: Shape: {ret.shape}; Sum: {ret.sum()}; Time elapse: {time.perf_counter() - st_cu:.3f} seconds ');
"""
