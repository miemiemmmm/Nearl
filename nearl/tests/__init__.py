def vectorize():
  print("running vectorize")
  from nearl.tests import vectorize
  print(dir(vectorize))

  print(vectorize.file_list)
  print("vectorize done")

def open3D():
  from . import open3d_func
  open3d_func.convex_ratio()

def jax_2dcnn():
  import nearl.tests.training_jax as training_jax
  training_jax.cnn_2d()
