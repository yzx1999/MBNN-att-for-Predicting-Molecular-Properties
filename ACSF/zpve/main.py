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
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    print("device", device)
    print("pid", os.getpid())
    #######################################################################################################
    batch_size = 256
    train_loader, valid_loader, info = QM9_dataloader(batch_size)
    print("batch_size", batch_size)
    #######################################################################################################
    cutoff = 7.0
    input_channel=3275
    hidden_channel = 512
    num_layers = 4
    model = OneeleNet(input_channel = input_channel,hidden_channel=hidden_channel).to(device)
#    model = PTSDGraphNet(cutoff=cutoff, hidden_channel=hidden_channel, num_layers=num_layers).to(device)
    print("cutoff", cutoff)
    print("hidden_channel", hidden_channel)
    print("num_layers", num_layers)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    batch_count = math.ceil(info["train_count"] / batch_size)
    scheduler_lr = LinearWarmupExponentialDecay(
        optimizer, warmup_steps=batch_count * 5, decay_rate=0.2, decay_steps=batch_count * 300
    )
    criterion = torch.nn.SmoothL1Loss()

    model = model.to(device)
    print('参数总量:', sum(p.numel() for p in model.parameters()))

    property = 6
    save_path = f'./{hidden_channel}_{num_layers}_{type(criterion).__name__}_{property}3.pth'

    print("property", property)
    print("save_path", save_path)

    best_loss = float("inf")
    for epoch in range(2000):
        model.train()
        _energy_out, _energy_y = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        start_time = datetime.datetime.now()
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data,mean_proper[property],std_proper[property])
            loss = criterion(out, data.y[:, property].view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            scheduler_lr.step()
            _energy_out = torch.cat([_energy_out, out.detach_().view(-1, 1)], dim=0)
            _energy_y = torch.cat([_energy_y, data.y[:, property].view(-1, 1)], dim=0)
            if idx % 50 == 0:
                info = f"[IDX]:{idx:0>3} [loss]:{loss.item():.6f}  [lr]:{optimizer.param_groups[0]['lr']:.7f}"
                print(info)
        time = datetime.datetime.now() - start_time
        loss = torch.mean(torch.abs(_energy_out - _energy_y)).cpu().item()
        print(f"[EPOCH]:{epoch:0>4d} [loss]:{loss:.4f} [lr]:{optimizer.param_groups[0]['lr']:.7f} [time]:{time}")
        # VALID
        model.eval()
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
                out = model(data,mean_proper[property],std_proper[property])
                _energy_out = torch.cat([_energy_out, out.detach_()], dim=0)
                _energy_y = torch.cat([_energy_y, data.y[:, property].unsqueeze(1)], dim=0)
                ###################################旋转
                data.pos = torch.matmul(data.pos, rotation_matrix.t())
                rotation_out = model(data,mean_proper[property],std_proper[property])  # [bs,1]
                _rotate_out = torch.cat([_rotate_out, rotation_out.detach_()], dim=0)

        loss = torch.mean(torch.abs(_energy_out - _energy_y)).cpu().item()
        print(f"[VALID] [loss]:{loss:.4f}] [rotate]:[{(_rotate_out - _energy_out).abs().mean():.6f}]")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)
            print(f"[BEST LOSS] {epoch} {best_loss:.5f}")
