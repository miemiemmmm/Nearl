Manual focus parser
===================

.. code-block:: python

  TO BE ADDED

.. If the built-in focus parser does not meet your needs, you could define your own focus parser. 
.. The following is an example to read a standalone json file recording the atom indices of the structure of interest. 

.. # Initialize the proper shaped array to store the focal points

.. # There has to be three dimensions: number of frame slices, number of focal points, and 3

.. .. code-block:: python

..   def manual_focal_parser(traj): 
..     # The reference json file to map the indices to track
..     ligandmap = "/MieT5/Nearl/data/misato_ligand_indices.json"
..     # The sliding time window has to match the one put in the featurizer
..     timewindow = 10
..     with open(ligandmap, "r") as f:
..       LIGAND_INDICE_MAP = json.load(f)
..       ligand_indices = np.array(LIGAND_INDICE_MAP[traj.identity.upper()])
..     FOCALPOINTS = np.full((traj.n_frames // timewindow, 1, 3), 99999, dtype=np.float32)
..     for i in range(traj.n_frames // timewindow):
..       FOCALPOINTS[i] = np.mean(traj.xyz[i*timewindow][ligand_indices], axis=0)
..     return FOCALPOINTS

.. To use the manually defined focal point parser, you need to register that function to the featurizer

.. .. code-block:: python

..   feat.register_focus(manual_focal_parser, "function")



.. TODO
.. Add the tutorial index when appropriate
.. Add script download link when appropriate
