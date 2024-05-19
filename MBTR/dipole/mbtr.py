from ase.build import molecule
from dscribe.descriptors import MBTR
import numpy as np

species = ["H", "C", "O", "N","F"]

#mbtr1 = MBTR(
#    species=species,
#    geometry={"function": "inverse_distance"},
#    grid={"min": 0, "max": 1.0, "n": 100, "sigma": 0.1},
#    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
#    periodic=False,normalization="l2")

#mbtr2 = MBTR(
#    species=species,
#    geometry={"function": "atomic_number"},
#    grid={"min": 0, "max": 1.0, "n": 100, "sigma": 0.1},
#    weighting={"function": "unity", "scale": 0.5, "threshold": 1e-3},
#    periodic=False,normalization="l2")

#mbtr3 = MBTR(
#    species=species,
#    geometry={"function": "cosine"},
#    grid={"min": 0, "max": 1.0, "n": 25, "sigma": 0.2},
#    weighting={"function": "smooth_cutoff","r_cut":3.5, "scale": 0.5, "threshold": 1e-3},
#    periodic=False,normalization="l2")


mbtr1 = MBTR(
      species=species,
      geometry={"function": "inverse_distance"},
      grid={"min": 0, "max": 1.0, "n": 100, "sigma": 0.1},
      weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
      periodic=False,normalization="l2")

mbtr2 = MBTR(
      species=species,
      geometry={"function": "atomic_number"},
      grid={"min": 0, "max": 1.0, "n": 20, "sigma": 0.1},
      weighting={"function": "unity"},
      periodic=False,normalization="n_atoms")

mbtr3 = MBTR(
      species=species,
      geometry={"function": "cosine"},
      grid={"min": 0, "max": 1.0, "n": 25, "sigma": 0.35},
      weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
      periodic=False,normalization="l2")



import ase 
from ase.io import read

#mols = read('77153.xyz',index=':')
mols = read('test2.xyz',index=':')
print(mols)

mols_mbtr1 = mbtr1.create(mols)
mols_mbtr2 = mbtr2.create(mols)
mols_mbtr3 = mbtr3.create(mols)
print(mols_mbtr1.shape,mols_mbtr2.shape,mols_mbtr3.shape)
mols_mbtr =np.concatenate((mols_mbtr1, mols_mbtr2, mols_mbtr3), axis=1)


print(mols_mbtr.shape)
p_mbtr = mols_mbtr[mols_mbtr > 0]
#i_soap = np.where(mols_soap > 0)

print(len(p_mbtr), p_mbtr)
#print(i_soap)
#print(len(mols_soap))
for mbtr in mols_mbtr:
    print(type(mbtr))
    print(mbtr.shape)

