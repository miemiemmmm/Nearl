6. Batch PDB structure loading
==============================

Single snapshot from MD or static structure (like PDB) are dealt as a trajectory with only one frame. If this is the case, you could only needs to load the structure as  

.. code-block:: python

  from nearl.io import Trajectory, TrajectoryLoader
  traj_list = [pdbfile1, pdbfile2, pdbfile3, ..., pdbfileN]
  traj_loader = TrajectoryLoader([(i,) for i in traj_list], trajtype=Trajectory)






