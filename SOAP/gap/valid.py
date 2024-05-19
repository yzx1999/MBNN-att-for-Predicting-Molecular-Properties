import scipy.signal
import sys, os
import datetime
import random
import numpy as np
import math

import torch
import torch.optim as optim

from model import OneeleNet
from load_data import QM9_dataloader
from schedules import LinearWarmupExponentialDecay


mean_proper = [2.706, 75.19, -0.240, 0.011, 0.251, 1189.53, 0.149,-411.544,-411.536,-411.535,-411.577,31.601,-1750.813,-1761.481,-1771.547,-1629.388]

std_proper =[1.53, 8.19,0.02,0.05,0.05,279.76,0.03,40.06,40.06,40.06,40.06,4.06,239.31,241.44,243.15,220.21]

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(99)
    device = torch.device(f"cuda:5" if torch.cuda.is_available() else "cpu")

    print("device", device)
    print("pid", os.getpid())
    #######################################################################################################
    batch_size = 256
    train_loader, valid_loader, info = QM9_dataloader(batch_size)
    print("batch_size", batch_size)
    #######################################################################################################
    cutoff = 7.0
    input_channel=3255
    hidden_channel = 512

    num_layers = 5
    pair_hidden = 256
    save_path = "./512_4_SmoothL1Loss_23.pth"
    model_homo = OneeleNet(input_channel = input_channel,hidden_channel=hidden_channel)
    model_homo.load_state_dict(torch.load(save_path))
    model_homo = model_homo.to(device)
    save_path = "./512_4_SmoothL1Loss_33.pth"
    model_lumo = OneeleNet(input_channel = input_channel,hidden_channel=hidden_channel)
    model_lumo.load_state_dict(torch.load(save_path))
    model_lumo = model_lumo.to(device)
    print("cutoff", cutoff)
    print("hidden_channel", hidden_channel)
    print("num_layers", num_layers)



    property = 4

    print("property", property)

    for epoch in range(1):
        model_homo.eval()
        model_lumo.eval()
        _energy_out, _energy_y = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        ###############################################不变确认
        _rotate_out = torch.Tensor([]).to(device)
        theta = torch.tensor([3.1415926 / 2])  # 45度
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ]).to(device)
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                data.to(device)
                homo = model_homo(data,mean_proper[property-2],std_proper[property-2])
                lumo = model_lumo(data,mean_proper[property-1],std_proper[property-1])
                out = lumo-homo
                print(out.shape)
                _energy_out = torch.cat([_energy_out, out.detach_()], dim=0)
                _energy_y = torch.cat([_energy_y, data.y[:, property].unsqueeze(1)], dim=0)
                ###################################旋转

        loss = torch.mean(torch.abs(_energy_out - _energy_y)).cpu().item()
        print(f"[VALID] [loss]:{loss:.4f}")
