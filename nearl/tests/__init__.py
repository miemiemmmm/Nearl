import time

def vectorize():
  from .vectorize import do_vectorize
  st = time.perf_counter()
  do_vectorize()
  print(f"Vectorization done, The test took {time.perf_counter() - st:.3f} seconds")

def open3D_functions():
  from . import open3d_func
  open3d_func.convex_ratio()

def jax_2dcnn():
  import nearl.tests.training_jax as training_jax
  training_jax.cnn_2d()

def jax_3dcnn():
  import nearl.tests.training_jax as training_jax
  training_jax.cnn_3d()
