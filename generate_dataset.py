import os
import numpy as np
import math
import torch
from sklearn.metrics.pairwise import pairwise_distances
import glob
import shutil
import numpy as np
import re


def cartesian_to_fractional(cartesian_coords, box_size):
    """
    Convert Cartesian coordinates to fractional coordinates.

    Parameters:
    - cartesian_coords: Numpy array of shape (N, 3) representing N points in Cartesian space.
    - box_size: Tuple of size (3, ) representing the simulation box dimensions.

    Returns:
    - fractional_coords: Numpy array of shape (N, 3) representing N points in fractional space.
    """
    fractional_coords = cartesian_coords / box_size
    return fractional_coords


def dist_fractional(A, B):
    """
    This function calculates the distance between two atoms.
    Their coordinates are in the fractional coordinate format.
    A: coordinates of atom A. eg [0.105692 0.0273734 0.193402]
    B: coordinates of atom B. eg [0.094583 0.0534993 0.577509]
    """
    XA = A[0]
    YA = A[1]
    ZA = A[2]
    XB = B[0]
    YB = B[1]
    ZB = B[2]

    XAB = XA - XB
    XAB = XAB - math.floor(XAB)
    if XAB > 0.5:
        XAB = XAB - 1.0
    elif XAB < -0.5:
        XAB = XAB + 1.0
    YAB = YA - YB
    if YAB > 0.5:
        YAB = YAB - 1.0
    elif YAB < -0.5:
        YAB = YAB + 1.0
    ZAB = ZA - ZB
    if ZAB > 0.5:
        ZAB = ZAB - 1.0
    elif ZAB < -0.5:
        ZAB = ZAB + 1.0
    DAB = math.sqrt(XAB * XAB + YAB * YAB + ZAB * ZAB)
    return DAB

# Calculate pair-wise distance matrix of atoms in a molecule
def get_dmatrix(coordinates, scale):
    """
    This function calculates the pair-wise Euclidean distance of atoms in a molecule
    coordinates: coordinates of atoms
    scale: the cubic box size, in angstrom
    """
    # Add 1 to negative coordinates
    mask = coordinates < 0
    y = torch.zeros_like(coordinates)
    y[mask] = 1.0
    coordinates = coordinates + y
    dmatrix = torch.tensor(pairwise_distances(coordinates, metric=dist_fractional)) * scale
    return dmatrix


# Get the adjacency matrix of a molecule using te dmatrix
def get_adj_matrix(dmatrix, atom_types):
    """
    Using the cutoff distance to determine if there is a chemical bond between a pair of atoms
    dmatrix: pairwise distances of atoms
    cutoff: cutoff distance, in angstron
    """
    adjmatrix = torch.zeros_like(dmatrix, dtype=int)
    cutoffs = {('H', 'H'): 0.8, ('H', 'O'): 1.2, ('O', 'H'): 1.2, ('O', 'O'): 1.4}

    # 根据原子类型和对应的cutoff更新邻接矩阵
    for i, atom_i in enumerate(atom_types):
        for j, atom_j in enumerate(atom_types):
            cutoff = cutoffs[(atom_i, atom_j)]
            if dmatrix[i, j] <= cutoff:
                adjmatrix[i, j] = 1
    # mask = dmatrix <= cutoff
    # adjmatrix[mask] = 1
    adjmatrix.fill_diagonal_(0)
    return adjmatrix

def calculate_distance(coord1, coord2):
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    dz = coord2[2] - coord1[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)

def generate_btmatrix(d_path, c_path, output_dir):
    with open(d_path, "r") as f:
        dmatrix = f.readlines()
    dmatrix = [line.split() for line in dmatrix]
    dmatrix = np.asarray(dmatrix, dtype=float)
    with open(c_path, "r") as f:
        lines = f.readlines()[2:]  # Read from the third line to the end
        atom_types = [line.split()[0] for line in lines]
    # atom_types = ['O', 'H', 'H']
        # data = f.readlines()
    # data = [line.split() for line in data]
    # data = np.asarray(data, dtype=float)
    dmatrix = torch.tensor(dmatrix, dtype=torch.float32)

    adj = get_adj_matrix(dmatrix, atom_types)
    adj = adj.numpy()
    # base_name_without_ext = os.path.splitext(os.path.basename(input_filename))[0]
    # new_basename = input_filename.replace("DMATRIX", "BTMATRIX")
    match = re.search(r'DMATRIX_(\d+)', d_path)
    if match:
        i = match.group(1)  # 提取的数字部分，即'i'的值
        new_filename = f"BTMATRIX_{i}"
        output_filename = os.path.join(output_dir, new_filename)
    # output_filename = os.path.join(output_dir, os.path.basename(d_path))
    np.savetxt(output_filename, adj, fmt='%12d')

def generate_dmatrix(c_path, output_dir):
    with open(c_path, "r") as fptr:
        natoms = int(fptr.readline())
        fptr.readline()  # Skip the next line

        coordinates = []
        for _ in range(natoms):
            line = fptr.readline().split()
            atomname = line[0]
            coord = [float(x) for x in line[1:]]
            coordinates.append(coord)

    distance = np.zeros((natoms, natoms))

    for i in range(natoms):
        for j in range(natoms):
            distance[i][j] = calculate_distance(coordinates[i], coordinates[j])
            if distance[i][j] > 40.0:
                exit(1)

    # 获取文件的基本名称并移除".txt"扩展名
    # base_name_without_ext = os.path.splitext(os.path.basename(input_filename))[0]

    # 替换"CONFIG"为"DMATRIX"
    # new_basename = base_name_without_ext.replace("CONFIG", "DMATRIX")
    match = re.search(r'water(\d+)', c_path)
    if match:
        i = match.group(1)  # 提取的数字部分，即'i'的值
        new_filename = f"DMATRIX_{i}"
    output_filename = os.path.join(output_dir, new_filename)
    # new_filename = os.path.join(output_filename, f"DMATRIX_{i}")  # 根据新的文件名格式构建新文件名

    # 使用os.rename()函数重命名文件
    # os.rename(output_filename, new_filename)
    # single_output = os.path.join(output_dir, os.path.basename(input_filename))

    np.savetxt(output_filename, distance, fmt="%12.6f")

def generate_d_bt():
    # base_dir = '/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata'
    # pattern = 'WATER_*'  # 使用通配符匹配多个文件夹
    #
    # # 使用glob找到所有匹配的文件夹
    # directories = glob.glob(os.path.join(base_dir, pattern))
    #
    # for directory in directories:
    #     d_path = os.path.join(directory, 'DMATRIXES')
    #     c_path = os.path.join(directory, 'CONFIG')
    #     output_dir = os.path.join(directory, 'BTMATRIXES')
        # input_dir = os.path.join(directory, 'CONFIG')
        # output_dir = os.path.join(directory, 'DMATRIXES')
    #
    #     os.makedirs(output_dir, exist_ok=True)

    d_path = '/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata/21WATER/DMATRIXES'
    c_path = '/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata/21WATER/CONFIG'
    output_dir = '/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata/21WATER/BTMATRIXES'
    #
    # os.makedirs(output_dir, exist_ok=True)

    for i in range(0, 5600):
        dmatrix_filename = f"DMATRIX_{i}"
        config_filename = f"water{i}"
        d_filepath = os.path.join(d_path, dmatrix_filename)
        c_filepath = os.path.join(c_path, config_filename)
        generate_btmatrix(d_filepath, c_filepath, output_dir)

        # config_filename = f"water{i}"
        # c_filepath = os.path.join(c_path, config_filename)
        # generate_dmatrix(c_filepath, output_dir)

if __name__ == '__main__':
    # path = "/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata"
    # os.makedirs(path, exist_ok=True)
    # for i in range(19, 21):
    #     subdirectory_path = os.path.join(path, f"{i}WATER")
    #     os.makedirs(subdirectory_path, exist_ok=True)
    #     config = os.path.join(subdirectory_path, "CONFIG")
    #     os.makedirs(config, exist_ok=True)
    #     force = os.path.join(subdirectory_path, "FORCE")
    #     os.makedirs(force, exist_ok=True)
    #     energy = os.path.join(subdirectory_path, "ENERGY")
    #     os.makedirs(energy, exist_ok=True)
    #     bt = os.path.join(subdirectory_path, "BTMATRIXES")
    #     os.makedirs(bt, exist_ok=True)
    #     d = os.path.join(subdirectory_path, "DMATRIXES")
    #     os.makedirs(d, exist_ok=True)
    #
    #     for j in range(9, 10):
    #         path = "/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata"
    #         CONFIG1 = os.path.join(path, "CONFIG")
    #         ENERGY1 = os.path.join(path, "ENERGY")
    #         FORCE1 = os.path.join(path, "FORCE")
    #         DMATRIXES1 = os.path.join(path, "DMATRIXES")
    #         water = os.path.join(path, f"{j}WATER")
    #         CONFIG = os.path.join(water, "CONFIG")
    #         ENERGY = os.path.join(water, "ENERGY")
    #         FORCE = os.path.join(water, "FORCE")
    #         DMATRIXES = os.path.join(water, "DMATRIXES")
    #         for i in range(1001):
    #             k = i + 1001 * 19
    #             source_config_path = os.path.join(CONFIG1, f"water{k}")
    #             source_energy_path = os.path.join(ENERGY1, f"MLENERGY_{k}")
    #             source_force_path = os.path.join(FORCE1, f"MLFORCE_{k}")
    #             source_dma_path = os.path.join(DMATRIXES1, f"DMATRIX_{k}")
    #             destination_config_path = os.path.join(CONFIG, f"water{i}")
    #             destination_energy_path = os.path.join(ENERGY, f"MLENERGY_{i}")
    #             destination_force_path = os.path.join(FORCE, f"MLFORCE_{i}")
    #             destination_dma_path = os.path.join(DMATRIXES, f"DMATRIX_{i}")
    #
    #             shutil.move(source_config_path, destination_config_path)
    #             shutil.move(source_energy_path, destination_energy_path)
    #             shutil.move(source_force_path, destination_force_path)
    #             shutil.move(source_dma_path, destination_dma_path)

    # base_dir = '/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/data/1.8newdata'
    # pattern = 'WATER_*'  # 使用通配符匹配多个文件夹
    #
    # # 使用glob找到所有匹配的文件夹
    # directories = glob.glob(os.path.join(base_dir, pattern))
    #
    # for directory in directories:
    #     config = os.path.join(directory, "CONFIG")
    #     os.makedirs(config, exist_ok=True)
    #     energy = os.path.join(directory, "ENERGY")
    #     os.makedirs(energy, exist_ok=True)
    #     force = os.path.join(directory, "FORCE")
    #     os.makedirs(force, exist_ok=True)
    #     dmatrix = os.path.join(directory, "DMATRIXES")
    #     os.makedirs(dmatrix, exist_ok=True)
    #     btmatrix = os.path.join(directory, "BTMATRIXES")
    #     os.makedirs(btmatrix, exist_ok=True)
    #
    #     for i in range(5600):
    #         old_config = os.path.join(config, f"CONFIG_{i}.xyz")
    #         new_config = os.path.join(config, f"water{i}")
    #         os.rename(old_config, new_config)

        # old_config_path = os.path.join(config, "CONFIG_5600.xyz")
        # old_energy_path = os.path.join(energy, "MLENERGY_5600")
        # old_force_path = os.path.join(force, "MLFORCE_5600")
        # new_config_path = os.path.join(config, "CONFIG_0.xyz")
        # new_energy_path = os.path.join(energy, "MLENERGY_0")
        # new_force_path = os.path.join(force, "MLFORCE_0")
        # os.rename(old_config_path, new_config_path)
        # os.rename(old_energy_path, new_energy_path)
        # os.rename(old_force_path, new_force_path)

        # for i in range(1, 5601):
        #     source_config = os.path.join(directory, f"CONFIG_{i}.xyz")
        #     source_energy = os.path.join(directory, f"MLENERGY_{i}")
        #     source_force = os.path.join(directory, f"MLFORCE_{i}")
        #     destination_config = os.path.join(config, f"CONFIG_{i}.xyz")
        #     destination_energy = os.path.join(energy, f"MLENERGY_{i}")
        #     destination_force = os.path.join(force, f"MLFORCE_{i}")
        #     shutil.move(source_config, destination_config)
        #     shutil.move(source_energy, destination_energy)
        #     shutil.move(source_force, destination_force)
    generate_d_bt()


