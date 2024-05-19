import scipy.sparse
from torch_geometric.datasets.qm9 import *
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from ase import Atoms
#from dscribe.descriptors import SOAP



Eletable = [ 'H'  ,   'He' ,  'Li' ,  'Be' ,  'B'  ,  'C'  ,  'N' ,  'O' ,
             'F'  ,   'Ne' ,  'Na' ,  'Mg' ,  'Al' ,  'Si' ,  'P' ,  'S' ,
             'Cl' ,   'Ar' ,  'K'  ,  'Ca' ,  'Sc' ,  'Ti' ,  'V' ,  'Cr',
             'Mn' ,   'Fe' ,  'Co' ,  'Ni' ,  'Cu' ,  'Zn' ,  'Ga',  'Ge',
             'As' ,   'Se' ,  'Br' ,  'Kr' ,  'Rb' ,  'Sr' ,  'Y' ,  'Zr',
             'Nb' ,   'Mo' ,  'Tc' ,  'Ru' ,  'Rh' ,  'Pd' ,  'Ag',  'Cd',
             'In' ,   'Sn' ,  'Sb' ,  'Te' ,  'I'  ,  'Xe' ,  'Cs',  'Ba',
             'La' ,   'Ce' ,  'Pr' ,  'Nd' ,  'Pm' ,  'Sm' ,  'Eu',  'Gd',
             'Tb' ,   'Dy' ,  'Ho' ,  'Er' ,  'Tm' ,  'Yb' ,  'Lu',  'Hf',
             'Ta' ,   'W'  ,  'Re' ,  'Os' ,  'Ir' ,  'Pt' ,  'Au',  'Hg',
             'Tl' ,   'Pb' ,  'Bi' ,  'Po' ,  'At' ,  'Rn' ,  'Fr',  'Ra',
             'Ac' ,   'Th' ,  'Pa' ,  'U'  ,  'Np' ,  'Pu' ,  'Am',  'Cm',
             'Bk' ,   'Cf' ,  'Es' ,  'Fm' ,  'Md' ,  'No' ,  'Lr',  'Rf',
             'Db' ,   'Sg' ,  'Bh' ,  'Hs' ,  'Mt' ,  'Ds' ,  'Rg',  'Cn',
             'Uut',   'Fl' ,  'Uup',  'Lv' ,  'Uus',  'UUo', ]


def calsoap(z,pos):
    str_zlist = [Eletable[ele-1] for ele in z]
    strz = ''.join(str_zlist)
    mol = Atoms(strz, pos.tolist())
    species = ["H", "C", "O", "N","F"]
    r_cut = 6.0
    n_max = 6
    l_max = 6
    sigma = 0.1
    
#    print("print SOAP")
#    soap = SOAP(
#     species=species,
#     periodic=False,
#     r_cut=r_cut,
#     n_max=n_max,
#     l_max=l_max,
#     sigma=sigma,
#     rbf='gto'
 #   )
 #   mol_soap = soap.create(mol)
    mol_soap = None
    return mol_soap


class ImproveQM9(QM9):

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')
            #posc = pos - pos*z.mean()
            descrip = calsoap(z,pos)
            descrip = torch.tensor(descrip, dtype=torch.float32)
            #print(descrip.shape)
            data = Data(x=x, z=z, pos=pos, descrip=descrip, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            #print(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


def QM9_dataloader(batch_size):
    dataset = ImproveQM9(root='./data')
    train_dataset, valid_dataset = random_split(dataset, [0.95, 0.05])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    info = {
        "train_count": len(train_dataset),
        "valid_count": len(valid_dataset),
    }
    return train_loader, valid_loader, info


if __name__ == '__main__':
    QM9_dataloader(512)
