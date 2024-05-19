import e3nn
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

class OneeleNet(torch.nn.Module):
    def __init__(self,input_channel=420,hidden_channel=128):
        super().__init__()
        self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(input_channel, hidden_channel*2),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_channel*2, hidden_channel),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_channel, hidden_channel),
                torch.nn.LayerNorm(hidden_channel),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_channel, 1), )        
#        self.z_emb = torch.nn.Embedding(105, input_channel)
#        self.E_emb_w = torch.nn.Embedding(105, 1)
#        self.E_emb_b = torch.nn.Embedding(105, 1)
        
    def forward(self, data,mean,std) -> Tensor:
        z, batch, x = data.z, data.batch, data.descrip
        #x = x[:,:500]
        #print(x.shape)
 #       elez = self.z_emb(z - 1)
        out = self.out_layer(x)
        energy =out
 #       shiftele1 = self.E_emb_w(z - 1)
 #       shiftele2 = self.E_emb_b(z - 1)
 #       out = shiftele1*out + shiftele2
 #       energy = scatter(out, batch, dim=0)
        energy = energy*std+mean
        return energy
        

if __name__ == '__main__':
    net = OneeleNet()
    print(net)
