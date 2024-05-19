import scipy.sparse
import os
import numpy as np
import torch
import pickle

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices

class PTSDDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'data', "clean_data", "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "data", "clean_data", "processed")

    @property
    def raw_file_names(self):
        names = ['node_attributes', 'node_pos', 'ptsd_norm_x', 'edge_index', 'distince', 'graph_labels', 'idx', 'name',
                 'z']
        return [f'{self.name}.csv' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        node_attributes = np.loadtxt(os.path.join(self.raw_dir, "node_attributes.csv"), delimiter=",")
        node_pos = np.loadtxt(os.path.join(self.raw_dir, "node_pos.csv"), delimiter=",")
        ptsd_norm_x = np.loadtxt(os.path.join(self.raw_dir, "ptsd_norm_x.csv"), delimiter=",")
        print("dealing x...")
        x = torch.tensor(np.concatenate((node_attributes, node_pos, ptsd_norm_x), axis=1),dtype=torch.float32)
        print("dealing x...:", x.shape)
        print("dealing edge_index...")
        edge_index = torch.tensor(
            np.loadtxt(os.path.join(self.raw_dir, "edge_index.csv"), delimiter=","),dtype=torch.int64
        ).t().contiguous()
        print("dealing edge_index...:", edge_index.shape)
        print("dealing distance...")
        distance = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "distance.csv"), delimiter=","),dtype=torch.float32)
        print("dealing distance...:", distance.shape)
        print("dealing pos...")
        pos = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "node_pos.csv"), delimiter=","),dtype=torch.float32)
        print("dealing pos...:", pos.shape)
        print("dealing z...")
        z = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "z.csv"), delimiter=","),dtype=torch.int64)
        print("dealing z...:", z.shape)
        print("dealing edge_attr...")
        edge_attr = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "edge_attr.csv"), delimiter=","),dtype=torch.float32)
        print("dealing edge_attr...:", edge_attr.shape)
        print("dealing y...")
        y = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "graph_labels.csv"), delimiter=","),dtype=torch.float32)
        print("dealing y...:", y.shape)
        print("dealing idx...")
        idx = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "idx.csv"), delimiter=","),dtype=torch.int64)
        print("dealing idx...:", idx.shape)
        # name list
        with open(os.path.join(self.raw_dir, "name.csv")) as f:
            name = f.read().split("\n")
        # Data(x=[5, 11], edge_index=[2, 8], edge_attr=[8, 4], y=[1, 19], pos=[5, 3], z=[5], name='gdb_1', idx=[1])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, z=z, name=name, idx=idx,distance=distance)

        with open(os.path.join(self.raw_dir, "slices.pkl"), 'rb') as file:
            slices = pickle.load(file)
        #处理batch
        batch = torch.tensor(np.loadtxt(os.path.join(self.raw_dir, "indicator.csv"), delimiter=","),dtype=torch.int64)
        self.data, self.slices = split(data, batch)
        torch.save((self._data, self.slices), self.processed_paths[0])


def QM9_dataloader():
    file_path = os.path.join(FILE_PATH)
    dataset = PTSDDataset(file_path)
    train_dataset, valid_dataset = random_split(dataset, [0.9, 0.1])
    print(train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)


if __name__ == '__main__':
    train_loader, valid_loader, train_count, valid_count = QM9_dataloader()
    for idx, data in enumerate(train_loader):
        print(data.x[0:11])
        print(data)
        break

