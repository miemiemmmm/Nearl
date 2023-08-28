import numpy as np
from rdkit.Chem import Draw


def get_axis_index(idx, colnr):
  x = np.floor(idx/colnr).astype(int)
  y = idx%colnr
  return (x,y)


def draw_grid_mols(axes, mols, colnr):
  for axis in axes.reshape((-1,1)):
    axis[0].axis("off")
  for idx, mol in enumerate(mols):
    figi = Draw.MolToImage(mol)
    figi.thumbnail((100, 100))
    index = get_axis_index(idx, colnr)
    axes[index].imshow(figi)
    axes[index].set_title(f"SubStruct {idx+1}")


def draw_2d(mols, **kwarg):
  """
  Draw molecule grid image from a list of Chem.Mol molecules
  """
  from IPython.display import display
  from rdkit import Chem
  from rdkit.Chem import Draw, AllChem
  mpr = kwarg.get("mpr", 5)
  legends = kwarg.get("legends",[])
  label = kwarg.get("label", False)
  rm_h = kwarg.get("removeHs", False)
  san = kwarg.get("sanitize", False)
  _mols = []
  for mol in mols:
    if mol is not None:
      try:
        _mol = Chem.Mol(mol)
        AllChem.Compute2DCoords(_mol)
        if rm_h:
          _mol = Chem.RemoveHs(_mol)
        if label:
          for idx, atom in enumerate(_mol.GetAtoms()):
            idx = atom.GetIdx()
            atom.SetProp("atomLabel", atom.GetSymbol()+str(idx))
            atom.SetAtomMapNum(atom.GetIdx())
        Chem.SanitizeMol(_mol)
        _mols.append(_mol)
      except:
        _mols.append(None)
    else:
      _mols.append(None)
  if len(legends) == 0:
    img = Draw.MolsToGridImage(_mols, molsPerRow=mpr, subImgSize=(400,400), maxMols=200,
                               returnPNG=False)
  else:
    img = Draw.MolsToGridImage(_mols, molsPerRow=mpr, subImgSize=(400,400), maxMols=200,
                               legends=legends, returnPNG=False)
  display(img)
  return img


def draw_2df(file_path, **kwarg):
  """
  Draw molecule grid image from a physical file
  """
  from nearl.utils import chemtools
  suppl = chemtools.molfile_to_rdkit(file_path, **kwarg)
  img = draw_2d(suppl, **kwarg)
  return img




