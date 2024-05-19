import os, sys
import math
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def _min_zero(coor):  # 将cell过小的值置为0
    if abs(coor) < 1e-8: return 0
    return coor


def Lat(line):  # 将 a b c 三个角度转成 3*3的矩阵
    pbc = [float(l) for l in line.split()[1:7]]
    a, b, c = pbc[0:3]
    alpha, beta, gamma = [x * np.pi / 180.0 for x in pbc[3:]]

    bc2 = b ** 2 + c ** 2 - 2 * b * c * math.cos(alpha)
    h1 = _min_zero(a)
    h2 = _min_zero(b * math.cos(gamma))
    h3 = _min_zero(b * math.sin(gamma))
    h4 = _min_zero(c * math.cos(beta))
    h5 = _min_zero(((h2 - h4) ** 2 + h3 ** 2 + c ** 2 - h4 ** 2 - bc2) / (2 * h3))
    h6 = _min_zero(math.sqrt(c ** 2 - h4 ** 2 - h5 ** 2))
    lat = [h1, 0., 0., h2, h3, 0., h4, h5, h6]
    lat = [f"{i:.9f}" for i in lat]
    return lat


def check_positions_in_cell(positions, cell):  # 检查原子是否在晶胞内
    # Convert positions to fractional coordinates
    fractional_coords = torch.linalg.solve(cell.T, positions.T).T
    # Check if fractional coordinates are within [0, 1] for each position
    is_inside = torch.all((fractional_coords >= 0) & (fractional_coords <= 1), dim=1)

    return is_inside


def move_center_torch(pos, cell):  # 将原子移动到晶胞内
    fractional_coords = torch.linalg.solve(cell.T, pos.T).T
    adjusted_fractional_coords = fractional_coords % 1
    adjusted_positions = (cell.T @ adjusted_fractional_coords.T).T
    return adjusted_positions


def check_pos(pos, cell):
    pos = [[float(j) for j in p] for p in pos]
    cell = [float(j) for j in cell]
    pos = torch.tensor(pos, dtype=torch.float64)
    cell = torch.tensor(cell, dtype=torch.float64).view(3, 3)
    inside = check_positions_in_cell(pos, cell).all().item()
    if not inside:
        pos = move_center_torch(pos, cell)
        if not check_positions_in_cell(pos, cell).all().item(): raise Exception("逻辑异常")
    return pos


def deal_force_arc(raw_dir):  # 处理F文件
    energy_list, force_list, stress_list = [], [], []
    _force = []
    force_file = os.path.join(raw_dir, "force.arc")
    with open(force_file, "r") as f:
        for line in tqdm(f.readlines(), desc="force"):
            line = line.split()
            if line and line[0] == 'For':
                energy = line[-1]
                energy_list.append(energy)
            elif len(line) == 3:
                fx, fy, fz = line[0], line[1], line[2]
                _force.append((fx, fy, fz))
            elif len(line) == 6:
                xx, xy, xz, yy, yz, zz = line
                stress = [xx, xy, xz, xy, yy, yz, xz, yz, zz]
                stress_list.append(stress)
            elif len(line) == 0:
                if _force: force_list.append((_force))
                _force = []
    return energy_list, force_list, stress_list


def deal_structure_arc(raw_dir, energy_list, force_list, stress_list):
    structure_file = os.path.join(raw_dir, "structure.arc")
    idx, data_list = 0, []
    with open(structure_file, "r") as f:
        structure_end = False
        for line in tqdm(f.readlines(), desc="structure"):
            if "CORE" in line:
                line = line.split()
                _z, _pos = line[0], (line[1], line[2], line[3])
                z.append(_z)
                pos.append(_pos)
            elif "Energy" in line:
                line = line.split()
                energy, structure_id = line[-1], line[1]
            elif "PBC" in line and "ON" not in line:  # latice cell 为了计算精度需要以float64保存
                cell = Lat(line)
            elif "!DATE" in line:
                z, pos = [], []
            elif "end" in line:
                structure_end = not structure_end
            ###################################################################################结构结束
            if structure_end:
                # =============================================================
                if energy != energy_list[idx]: raise Exception(f"[{idx}]能量不相等")
                if len(z) != len(force_list[idx]): raise Exception(f"[{idx}]数量不相等")
                check_pos(pos, cell)
                data = {
                    "num": len(z),
                    "lattice": cell,
                    "properties": "species:S:1:pos:R:3:forces:R:3",
                    "stress": stress_list[idx],
                    "energy": energy,
                    "z": z,
                    "pos": pos,
                    "force": force_list[idx],
                    "structure_id": structure_id,
                }
                data_list.append(data)
                idx += 1
    return data_list


def write_extxyz(name, data_list, raw_dir):
    file = os.path.join(raw_dir, f"{name}.extxyz")
    with open(file, "w") as f:
        for data in data_list:
            num = data["num"]
            lattice = " ".join([str(i) for i in data["lattice"]])
            properties = data["properties"]
            energy = data["energy"]
            structure_id = data["structure_id"]
            stress = " ".join(data["stress"])
            z = data["z"]
            pos = data["pos"]
            force = data["force"]
            f.write(f"{num}\n")
            info = f'Lattice="{lattice}" Properties={properties} energy={energy} structure_id={structure_id} pbc="T T T"\n'
            f.write(info)
            for i in range(num):
                _z, _pos, _force = z[i], pos[i], force[i]
                _format = lambda x: " " * (20 - len(x)) + x
                _z = _z + " " * (4 - len(_z))
                _pos = "".join([_format(j) for j in pos[i]])
                _force = "".join([_format(j) for j in force[i]])
                line = f"{_z}{_pos}{_force}\n"
                f.write(line)
            # f.write("\n")


def load_structure(name, raw_dir):
    # 处理force文件
    energy_list, force_list, stress_list = deal_force_arc(raw_dir)
    # 处理structure文件
    data_list = deal_structure_arc(raw_dir, energy_list, force_list, stress_list)
    # 写extxyz
    write_extxyz(name, data_list, raw_dir)
    return data_list


if __name__ == '__main__':
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(FILE_PATH, "data")
    name = "TIOtest"
    load_structure(name, raw_dir)
