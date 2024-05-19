import e3nn
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

class OneeleNet(torch.nn.Module):
    def __init__(self,input_channel=420,hidden_channel=128):
        super().__init__()
        self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(input_channel, hidden_channel),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_channel, hidden_channel),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_channel, hidden_channel),
                torch.nn.LayerNorm(hidden_channel),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_channel, 1), )        
        self.z_emb = torch.nn.Embedding(105, input_channel)
        self.E_emb_w = torch.nn.Embedding(105, 1)
        self.E_emb_b = torch.nn.Embedding(105, 1)
        
    def forward(self, data,mean,std) -> Tensor:
        batch, x = data.batch, data.x
        z = x[:,5].int()
        x = torch.cat((x[:, :11],x[:, 14:]), dim=1)
        elez = self.z_emb(z - 1)
        x = x * elez
        out = self.out_layer(x)
        shiftele1 = self.E_emb_w(z - 1)
        shiftele2 = self.E_emb_b(z - 1)
        out = shiftele1*out + shiftele2
        energy = scatter(out, batch, dim=0)
    #    energy = energy*std+mean
        return energy
        

if __name__ == '__main__':
    net = OneeleNet()
    print(net)
