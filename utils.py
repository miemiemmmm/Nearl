import pytraj as pt 
from scipy.spatial import distance_matrix

def conflictfactor(pdbfile, ligname, cutoff=5):
  VDWRADII = {'1': 1.1, '2': 1.4, '3': 1.82, '4': 1.53, '5': 1.92, '6': 1.7, '7': 1.55, '8': 1.52,
    '9': 1.47, '10': 1.54, '11': 2.27, '12': 1.73, '13': 1.84, '14': 2.1, '15': 1.8,
    '16': 1.8, '17': 1.75, '18': 1.88, '19': 2.75, '20': 2.31, '28': 1.63, '29': 1.4,
    '30': 1.39, '31': 1.87, '32': 2.11, '34': 1.9, '35': 1.85, '46': 1.63, '47': 1.72,
    '48': 1.58, '50': 2.17, '51': 2.06, '53': 1.98, '54': 2.16, '55': 3.43, '56': 2.68,
    '78': 1.75, '79': 1.66, '82': 2.02, '83': 2.07
  }
  traj = pt.load(pdbfile, top=pdbfile);
  traj.top.set_reference(traj[0]);
  pocket_atoms = traj.top.select(f":{ligname}<:{cutoff}");
  atoms = np.array([*traj.top.atoms])[pocket_atoms];
  coords = traj.xyz[0][pocket_atoms];
  atomnr = len(pocket_atoms);
  cclash=0;
  ccontact = 0;
  for i, coord in enumerate(coords):
    partners = [atoms[i].index]
    for j in list(atoms[i].bonded_indices()):
      if j in pocket_atoms:
        partners.append(j)
    partners.sort()
    otheratoms = np.setdiff1d(pocket_atoms, partners)
    ret = distance_matrix([coord], traj.xyz[0][otheratoms])
    thisatom = atoms[i].atomic_number
    vdw_pairs = np.array([VDWRADII[str(i.atomic_number)] for i in np.array([*traj.top.atoms])[otheratoms]]) + VDWRADII[str(thisatom)]
    cclash += np.count_nonzero(ret < vdw_pairs - 1.25)
    ccontact += np.count_nonzero(ret < vdw_pairs + 0.4)

    st = (ret < vdw_pairs - 1.25)[0];
    if np.count_nonzero(st) > 0:
      partatoms = np.array([*traj.top.atoms])[otheratoms][st];
      thisatom = np.array([*traj.top.atoms])[atoms[i].index];
      for part in partatoms:
        dist = distance_matrix([traj.xyz[0][part.index]], [traj.xyz[0][thisatom.index]]);
        print(f"Found clash between: {thisatom.name}({thisatom.index}) and {part.name}({part.index}); Distance: {dist.squeeze().round(3)}")

  factor = 1 - ((cclash/2)/((ccontact/2)/atomnr))
  print(f"Clashing factor: {round(factor,3)}; Atom selected: {atomnr}; Contact number: {ccontact}; Clash number: {cclash}");
  return factor
