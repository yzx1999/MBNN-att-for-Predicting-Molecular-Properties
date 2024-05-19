from ase.build import molecule
from dscribe.descriptors import SOAP
import numpy as np

species = ["H", "C", "O", "N","F"]
r_cut = 6.0
n_max = 6
l_max = 6
sigma = 0.1

soap = SOAP(
    species=species,
    periodic=False,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma,
    rbf='gto'
)


import ase 
from ase.io import read

#mols = read('77153.xyz',index=':')
mols = read('77153.xyz')
print(mols)

mols_soap = soap.create(mols)

print(mols_soap.shape)
print(mols_soap)
#p_soap = mols_soap[mols_soap > 0]
#i_soap = np.where(mols_soap > 0)

#print(len(p_soap), p_soap)
#print(i_soap)
#print(len(mols_soap))
#for msoap in mols_soap:
#    print(msoap.shape)

