import torch
import pickle
import pandas as pd
from typing import Optional, Tuple

import numpy as np
import os.path as osp
import os
from tqdm import tqdm
from torch_geometric.utils import degree
import random
import math
import matplotlib.pyplot as plt
from config import *
# from config import config_216
# from config import config_1, config_2, config_3, config_4, config_5, config_6
from dscribe.descriptors import SOAP
from ase import Atoms
from ase.io import read


# def numpy_Gaussian_scaler_1d(data, mean=-592.4566, var=989642.0194):
#     data_new = (data - mean) / var
#     return data_new

def numpy_min_max_scaler_1d(data, new_min=0.0, new_max=1.0):
    # assert isinstance(data, np.ndarray)
    # data_min = np.min(data)
    # data_max = np.max(data)
    # assert data_max - data_min > 0
    data_min = -361.77515914
    data_max = -17.19880547
    # data_max = -3721.3178
    core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new


def feature_min_max_scaler_1d(data, data_max, data_min, new_min=0.0, new_max=1.0):
    # assert isinstance(data, np.ndarray)
    # data_min = np.min(data)
    # data_max = np.max(data)
    # assert data_max - data_min > 0
    core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new


def reverse_min_max_scaler_1d(data_normalized, data_min=-3720.6905, data_max=-3721.3178, new_min=0.0, new_max=1.0):
    # 将归一化后的值映射回原始值的范围
    core = (data_normalized - new_min) / (new_max - new_min)
    data_original = core * (data_max - data_min) + data_min
    return data_original


def load_one_map(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    assert data.shape[0] == data.shape[1], "load data error in file {}".format(file_path)
    return data


# def load_one_coordinate(file_path):
#     with open(file_path, "r") as f:
#         data = f.readlines()[2:]
#     data = [line.split()[1:] for line in data]
#     data = np.asarray(data, dtype=float)
#     return data
def load_one_coordinate(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    return data


def load_one_coordinate_otherfild(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()[2:]  # 从第三行开始读取
    data = [line.split()[1:] for line in data]  # 忽略每行的第一列
    data = np.asarray(data, dtype=float)
    return data


def load_H2O_coordinate(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    # print(data.shape, data)
    return data


def load_one_structure(file_path):
    structure = read(file_path)
    # structure.set_cell([18.6206,18.6206,18.6206])
    species = set()
    species.update(structure.get_chemical_symbols())
    numbers = structure.get_global_number_of_atoms()
    return structure, species, numbers


def load_one_force(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    return data


def generate_atom_edges(edge_index_0, atom_num):
    item_list, item_count = edge_index_0.unique(return_counts=True)
    dic = {int(item_list[i]): int(item_count[i]) for i in range(item_list.shape[0])}
    atom_edges_array = torch.tensor([[dic[i]] if i in dic else [0] for i in range(atom_num)], dtype=torch.float32)
    return atom_edges_array


def generate_gap_file(folder, save_path, length, file_format, L, overwrite_flag=False):
    if osp.exists(save_path) and not overwrite_flag:
        print("Gap file {} exists. Skip generating ...".format(save_path))
        gaps = np.load(save_path)
        return gaps
    else:
        print("{}: {} files".format(folder, length))
        gaps = []
        for i in range(length):
            j = L[i]
            filename = osp.join(folder, file_format.format(j))
            with open(filename, "r") as f:
                lines = f.readlines()
                one_gap = float(lines[0])
            gaps.append(one_gap)
        gaps = np.asarray(gaps)
        gaps = gaps.astype(np.float64)
    min_gaps = np.min(gaps)
    max_gaps = np.max(gaps)
    return gaps, max_gaps, min_gaps


def generate_feature1(folder, length, file_format, L):
    features = []
    for i in tqdm(range(length)):
        j = L[i]
        c_path = osp.join(folder, file_format.format(j))
        c_data = load_one_coordinate(c_path)
        one_feature = generate_soap1(c_data)
        features.append(one_feature)
    features = np.asarray(features)
    features = features.reshape(-1, 722)
    max_value = []
    min_value = []
    with open("max_values_force_onehot_expand.txt", "r") as max_file:
        for line in max_file:
            max_value.append(float(line.strip()))
    with open("min_values_force_onehot_expand.txt", "r") as min_file:
        for line in min_file:
            min_value.append(float(line.strip()))
    for i in range(722):
        features[:, i] = feature_min_max_scaler_1d(features[:, i], max_value[i], min_value[i])
    return features


def generate_feature(folder, length, file_format, L):
    features = []
    for i in tqdm(range(length)):
        j = L[i]
        c_path = osp.join(folder, file_format.format(j))
        structure, species, numbers = load_one_structure(c_path)
        one_feature = generate_soap(structure, species, numbers)
        features.append(one_feature)
    # features = np.asarray(features)
    features = np.vstack(features)
    # features = features.reshape(-1, 1260)
    max_value = []
    min_value = []
    with open("max_values_force_onehot_expand.txt", "r") as max_file:
        for line in max_file:
            max_value.append(float(line.strip()))
    with open("min_values_force_onehot_expand.txt", "r") as min_file:
        for line in min_file:
            min_value.append(float(line.strip()))
    for i in range(722):
        features[:, i] = feature_min_max_scaler_1d(features[:, i], max_value[i], min_value[i])
    return features


def generate_dataset(input_path, output_path, config, DATA1=False):
    assert osp.exists(input_path)
    if not osp.exists(osp.join(output_path, "raw")):
        print("Created new folder: {}".format(osp.join(output_path, "raw")))
        os.makedirs(osp.join(output_path, "raw"))
    random_list = list(range(config.length))
    random.shuffle(random_list)

    gaps, max, min = generate_gap_file(osp.join(input_path, "ENERGY"),
                                       osp.join(input_path, "{}_gaps.npy".format(config.dataset)), config.length,
                                       config.format_eigen, random_list, overwrite_flag=True)
    gaps = numpy_min_max_scaler_1d(gaps)  # Normlize the energy
    if DATA1:
        features = generate_feature1(osp.join(input_path, "CONFIG"), config.length, config.format_conf, random_list)
    else:
        features = generate_feature(osp.join(input_path, "CONFIG"), config.length, config.format_conf, random_list)

    datasets = []
    num = 0
    for i in tqdm(range(config.length)):
        j = random_list[i]
        bt_path = osp.join(input_path, "BTMATRIXES", config.format_bmat.format(j))
        bt_data = load_one_map(bt_path)
        bt_data = torch.tensor(bt_data)
        a_path = osp.join(input_path, "FORCE", config.format_force.format(j))
        a_data = load_one_force(a_path)
        a_data = torch.tensor(a_data, dtype=torch.float32)
        d_path = osp.join(input_path, "DMATRIXES", config.format_dmat.format(j))
        d_data = load_one_map(d_path)
        d_data = torch.tensor(d_data, dtype=torch.float32)
        max_atoms = d_data.shape[0]
        c_path = osp.join(input_path, "CONFIG", config.format_conf.format(j))
        if DATA1:
            c_data = load_one_coordinate(c_path)
            c_data = torch.tensor(c_data, dtype=torch.float32)
        else:
            c_data = load_one_coordinate_otherfild(c_path)
            c_data = torch.tensor(c_data, dtype=torch.float32)

        matrix_data = bt_data * d_data
        edge_index = bt_data.nonzero().t().contiguous()
        node_degree = generate_atom_edges(edge_index[0], max_atoms)
        assert node_degree.shape[0] == c_data.shape[0], "Need to match max_natoms!"
        x_full = features[num: num + max_atoms, :]
        num += max_atoms
        # x_full = features[i * max_atoms : (i + 1) * max_atoms, :]
        x_full = torch.tensor(x_full)
        one_data = {
            'num_atom': max_atoms,
            'atom_type': x_full,
            'bond_type': matrix_data,
            'logP_SA_cycle_normalized': torch.tensor([gaps[i]], dtype=torch.float64),
            'z': a_data
        }
        datasets.append(one_data)

    # datasets_train = datasets[: config.train_length]
    # datasets_test = datasets[config.train_length: config.train_length + config.test_length]
    # datasets_val = datasets[config.train_length + config.test_length:]

    # return datasets_train, datasets_test, datasets_val, max, min
    # return features
    return datasets


def genergare_all_features():
    feature_1 = []
    for i in range(1, 22):
        datasets = generate_dataset(f"data/1.8newdata/{i}WATER", "dataset/waterforce/", config_data)
        feature_1.append(datasets)
    feature_2 = []
    for i in range(2, 22):
        datasets = generate_dataset(f"data/1-21-train/{i}WATER", "dataset/waterforce/", config_data2)
        feature_2.append(datasets)
    feature1 = generate_dataset("data/DATA 1/", "dataset/waterforce/", config_data3, True)
    feature2 = generate_dataset("data/Full_data/", "dataset/waterforce/", config_data4)
    feature2_t = generate_dataset("data/Full_test_data/", "dataset/waterforce/", config_data5)
    feature_rm = generate_dataset("data/random_full/", "dataset/waterforce/", config_data6)
    feature_rm_e = generate_dataset("data/random_expand/", "dataset/waterforce/", config_data6)
    feature1_more = generate_dataset("data/random_expand/1WATER/", "dataset/waterforce/", config_data2)

    # feature2_t = generate_dataset("data/Full_test_data/DATA 21", "dataset/waterforce/", config_2)
    # feature3 = generate_dataset("data/1.5-cutoff/DATA 3/", "dataset/waterforce/", config, 9)
    # feature4 = generate_dataset("data/1.5-cutoff/DATA 4/", "dataset/waterforce/", config, 12)
    # feature5 = generate_dataset("data/1.5-cutoff/DATA 5/", "dataset/waterforce/", config, 15)
    # feature6 = generate_dataset("data/1.5-cutoff/DATA 6/", "dataset/waterforce/", config, 18)
    # feature7 = generate_dataset("data/1.5-cutoff/DATA 7/", "dataset/waterforce/", config, 21)
    # feature8 = generate_dataset("data/1.5-cutoff/DATA 8/", "dataset/waterforce/", config, 24)
    # feature9 = generate_dataset("data/1.5-cutoff/DATA 9/", "dataset/waterforce/", config, 27)
    # feature10 = generate_dataset("data/1.5-cutoff/DATA 10/", "dataset/waterforce/", config, 30)
    # feature11 = generate_dataset("data/1.5-cutoff/DATA 11/", "dataset/waterforce/", config, 33)
    # feature12 = generate_dataset("data/1.5-cutoff/DATA 12/", "dataset/waterforce/", config, 36)
    # feature13 = generate_dataset("data/1.5-cutoff/DATA 13/", "dataset/waterforce/", config, 39)
    # feature14 = generate_dataset("data/1.5-cutoff/DATA 14/", "dataset/waterforce/", config, 42)
    # feature15 = generate_dataset("data/1.5-cutoff/DATA 15/", "dataset/waterforce/", config, 45)
    # feature16 = generate_dataset("data/1.5-cutoff/DATA 16/", "dataset/waterforce/", config, 48)
    # feature17 = generate_dataset("data/1.5-cutoff/DATA 17/", "dataset/waterforce/", config, 51)
    # feature18 = generate_dataset("data/1.5-cutoff/DATA 18/", "dataset/waterforce/", config, 54)
    # feature19 = generate_dataset("data/1.5-cutoff/DATA 19/", "dataset/waterforce/", config, 57)
    # feature20 = generate_dataset("data/1.5-cutoff/DATA 20/", "dataset/waterforce/", config, 60)
    # feature21 = generate_dataset("data/1.5-cutoff/DATA 21/", "dataset/waterforce/", config_1, 63)
    # featurebook6 = generate_dataset("data/1.5-cutoff/DATA_BOOK6/", "dataset/waterforce/", config, 18)
    # featurecage6 = generate_dataset("data/1.5-cutoff/DATA_CAGE6/", "dataset/waterforce/", config, 18)
    # featurecy3 = generate_dataset("data/1.5-cutoff/DATA_CY3/", "dataset/waterforce/", config, 9)
    # featurecy4_1 = generate_dataset("data/1.5-cutoff/DATA_CY4_1/", "dataset/waterforce/", config, 12)
    # featurecy4_2 = generate_dataset("data/1.5-cutoff/DATA_CY4_2/", "dataset/waterforce/", config, 12)
    # featurecycy5 = generate_dataset("data/1.5-cutoff/DATA_CY5/", "dataset/waterforce/", config, 15)
    # featurecyclic6 = generate_dataset("data/1.5-cutoff/DATA_CYCLIC6/", "dataset/waterforce/", config, 18)
    # featurel2 = generate_dataset("data/1.5-cutoff/DATA_L2/", "dataset/waterforce/", config, 6)
    # featurep4 = generate_dataset("data/1.5-cutoff/DATA_P4/", "dataset/waterforce/", config, 4)
    # featureprisim6 = generate_dataset("data/1.5-cutoff/DATA_PRISIM6/", "dataset/waterforce/", config, 18)
    # featurew7_1 = generate_dataset("data/1.5-cutoff/DATA_W7_1/", "dataset/waterforce/", config, 21)
    # featurew7_2 = generate_dataset("data/1.5-cutoff/DATA_W7_2/", "dataset/waterforce/", config, 21)
    # featurew7_3 = generate_dataset("data/1.5-cutoff/DATA_W7_3/", "dataset/waterforce/", config, 21)
    # featurew7_4 = generate_dataset("data/1.5-cutoff/DATA_W7_4/", "dataset/waterforce/", config, 21)
    # featurew7_5 = generate_dataset("data/1.5-cutoff/DATA_W7_5/", "dataset/waterforce/", config, 21)
    # featurew8_1 = generate_dataset("data/1.5-cutoff/DATA_W8_1/", "dataset/waterforce/", config, 24)
    # featurew8_2 = generate_dataset("data/1.5-cutoff/DATA_W8_2/", "dataset/waterforce/", config, 24)
    # featurew8_3 = generate_dataset("data/1.5-cutoff/DATA_W8_3/", "dataset/waterforce/", config, 24)
    # featurew8_4 = generate_dataset("data/1.5-cutoff/DATA_W8_4/", "dataset/waterforce/", config, 24)
    # featurew8_5 = generate_dataset("data/1.5-cutoff/DATA_W8_5/", "dataset/waterforce/", config, 24)
    # featurew8_6 = generate_dataset("data/1.5-cutoff/DATA_W8_6/", "dataset/waterforce/", config, 24)
    # featurew8_7 = generate_dataset("data/1.5-cutoff/DATA_W8_7/", "dataset/waterforce/", config, 24)
    # featurew8_8 = generate_dataset("data/1.5-cutoff/DATA_W8_8/", "dataset/waterforce/", config, 24)
    # featurew8_9 = generate_dataset("data/1.5-cutoff/DATA_W8_9/", "dataset/waterforce/", config, 24)
    # featurew8_10 = generate_dataset("data/1.5-cutoff/DATA_W8_10/", "dataset/waterforce/", config, 24)
    # featurew8_11 = generate_dataset("data/1.5-cutoff/DATA_W8_11/", "dataset/waterforce/", config, 24)
    # featurew8_12 = generate_dataset("data/1.5-cutoff/DATA_W8_12/", "dataset/waterforce/", config, 24)
    # featurew8_13 = generate_dataset("data/1.5-cutoff/DATA_W8_13/", "dataset/waterforce/", config, 24)
    # featurew8_14 = generate_dataset("data/1.5-cutoff/DATA_W8_14/", "dataset/waterforce/", config, 24)
    # featurew8_15 = generate_dataset("data/1.5-cutoff/DATA_W8_15/", "dataset/waterforce/", config, 24)
    # featurew9_1 = generate_dataset("data/1.5-cutoff/DATA_W9_1/", "dataset/waterforce/", config, 27)
    # featurew9_2 = generate_dataset("data/1.5-cutoff/DATA_W9_2/", "dataset/waterforce/", config, 27)
    # featurew9_3 = generate_dataset("data/1.5-cutoff/DATA_W9_3/", "dataset/waterforce/", config, 27)
    # featurew9_4 = generate_dataset("data/1.5-cutoff/DATA_W9_4/", "dataset/waterforce/", config, 27)
    # featurew9_5 = generate_dataset("data/1.5-cutoff/DATA_W9_5/", "dataset/waterforce/", config, 27)
    # featurew9_6 = generate_dataset("data/1.5-cutoff/DATA_W9_6/", "dataset/waterforce/", config, 27)
    # featurew9_7 = generate_dataset("data/1.5-cutoff/DATA_W9_7/", "dataset/waterforce/", config, 27)
    # featurew9_8 = generate_dataset("data/1.5-cutoff/DATA_W9_8/", "dataset/waterforce/", config, 27)
    # featurew9_9 = generate_dataset("data/1.5-cutoff/DATA_W9_9/", "dataset/waterforce/", config, 27)
    # featurew9_10 = generate_dataset("data/1.5-cutoff/DATA_W9_10/", "dataset/waterforce/", config, 27)
    # featurew9_11 = generate_dataset("data/1.5-cutoff/DATA_W9_11/", "dataset/waterforce/", config, 27)
    # featurew9_12 = generate_dataset("data/1.5-cutoff/DATA_W9_12/", "dataset/waterforce/", config, 27)
    # featurew9_13 = generate_dataset("data/1.5-cutoff/DATA_W9_13/", "dataset/waterforce/", config, 27)
    # featurew9_14 = generate_dataset("data/1.5-cutoff/DATA_W9_14/", "dataset/waterforce/", config, 27)
    # featurew10_1 = generate_dataset("data/1.5-cutoff/DATA_W10_1/", "dataset/waterforce/", config, 30)
    # featurew10_2 = generate_dataset("data/1.5-cutoff/DATA_W10_2/", "dataset/waterforce/", config, 30)
    # featurew10_3 = generate_dataset("data/1.5-cutoff/DATA_W10_3/", "dataset/waterforce/", config, 30)
    # featurew10_4 = generate_dataset("data/1.5-cutoff/DATA_W10_4/", "dataset/waterforce/", config, 30)
    # featurew10_5 = generate_dataset("data/1.5-cutoff/DATA_W10_5/", "dataset/waterforce/", config, 30)
    # featurew10_6 = generate_dataset("data/1.5-cutoff/DATA_W10_6/", "dataset/waterforce/", config, 30)
    # featurew10_7 = generate_dataset("data/1.5-cutoff/DATA_W10_7/", "dataset/waterforce/", config, 30)
    # featurew10_8 = generate_dataset("data/1.5-cutoff/DATA_W10_8/", "dataset/waterforce/", config, 30)
    # featurew10_9 = generate_dataset("data/1.5-cutoff/DATA_W10_9/", "dataset/waterforce/", config, 30)
    # featurew10_10 = generate_dataset("data/1.5-cutoff/DATA_W10_10/", "dataset/waterforce/", config, 30)
    # featurew10_11 = generate_dataset("data/1.5-cutoff/DATA_W10_11/", "dataset/waterforce/", config, 30)
    # featurew10_12 = generate_dataset("data/1.5-cutoff/DATA_W10_12/", "dataset/waterforce/", config, 30)
    # featurew10_13 = generate_dataset("data/1.5-cutoff/DATA_W10_13/", "dataset/waterforce/", config, 30)
    # featurew10_14 = generate_dataset("data/1.5-cutoff/DATA_W10_14/", "dataset/waterforce/", config, 30)
    # featurew11_1 = generate_dataset("data/1.5-cutoff/DATA_W11_1/", "dataset/waterforce/", config, 33)
    # featurew11_2 = generate_dataset("data/1.5-cutoff/DATA_W11_2/", "dataset/waterforce/", config, 33)
    # featurew11_3 = generate_dataset("data/1.5-cutoff/DATA_W11_3/", "dataset/waterforce/", config, 33)
    # featurew11_4 = generate_dataset("data/1.5-cutoff/DATA_W11_4/", "dataset/waterforce/", config, 33)
    # featurew11_5 = generate_dataset("data/1.5-cutoff/DATA_W11_5/", "dataset/waterforce/", config, 33)
    # featurew11_6 = generate_dataset("data/1.5-cutoff/DATA_W11_6/", "dataset/waterforce/", config, 33)
    # featurew11_7 = generate_dataset("data/1.5-cutoff/DATA_W11_7/", "dataset/waterforce/", config, 33)
    # featurew11_8 = generate_dataset("data/1.5-cutoff/DATA_W11_8/", "dataset/waterforce/", config, 33)
    # featurew11_9 = generate_dataset("data/1.5-cutoff/DATA_W11_9/", "dataset/waterforce/", config, 33)
    # featurew11_10 = generate_dataset("data/1.5-cutoff/DATA_W11_10/", "dataset/waterforce/", config, 33)
    # featurew11_11 = generate_dataset("data/1.5-cutoff/DATA_W11_11/", "dataset/waterforce/", config, 33)
    # featurew11_12 = generate_dataset("data/1.5-cutoff/DATA_W11_12/", "dataset/waterforce/", config, 33)
    # featurew11_13 = generate_dataset("data/1.5-cutoff/DATA_W11_13/", "dataset/waterforce/", config, 33)
    # featurew11_14 = generate_dataset("data/1.5-cutoff/DATA_W11_14/", "dataset/waterforce/", config, 33)
    # featurew12_1 = generate_dataset("data/1.5-cutoff/DATA_W12_1/", "dataset/waterforce/", config, 36)
    # featurew12_2 = generate_dataset("data/1.5-cutoff/DATA_W12_2/", "dataset/waterforce/", config, 36)
    # featurew12_3 = generate_dataset("data/1.5-cutoff/DATA_W12_3/", "dataset/waterforce/", config, 36)
    # featurew12_4 = generate_dataset("data/1.5-cutoff/DATA_W12_4/", "dataset/waterforce/", config, 36)
    # featurew12_5 = generate_dataset("data/1.5-cutoff/DATA_W12_5/", "dataset/waterforce/", config, 36)
    # featurew12_6 = generate_dataset("data/1.5-cutoff/DATA_W12_6/", "dataset/waterforce/", config, 36)
    # featurew12_7 = generate_dataset("data/1.5-cutoff/DATA_W12_7/", "dataset/waterforce/", config, 36)
    # featurew13_1 = generate_dataset("data/1.5-cutoff/DATA_W13_1/", "dataset/waterforce/", config, 39)
    # featurew13_2 = generate_dataset("data/1.5-cutoff/DATA_W13_2/", "dataset/waterforce/", config, 39)
    # featurew13_3 = generate_dataset("data/1.5-cutoff/DATA_W13_3/", "dataset/waterforce/", config, 39)
    # featurew13_4 = generate_dataset("data/1.5-cutoff/DATA_W13_4/", "dataset/waterforce/", config, 39)
    # featurew13_5 = generate_dataset("data/1.5-cutoff/DATA_W13_5/", "dataset/waterforce/", config, 39)
    # featurew13_6 = generate_dataset("data/1.5-cutoff/DATA_W13_6/", "dataset/waterforce/", config, 39)
    # featurew13_7 = generate_dataset("data/1.5-cutoff/DATA_W13_7/", "dataset/waterforce/", config, 39)
    # featurew13_8 = generate_dataset("data/1.5-cutoff/DATA_W13_8/", "dataset/waterforce/", config, 39)
    # featurew13_9 = generate_dataset("data/1.5-cutoff/DATA_W13_9/", "dataset/waterforce/", config, 39)
    # featurew14_1 = generate_dataset("data/1.5-cutoff/DATA_W14_1/", "dataset/waterforce/", config, 42)
    # featurew14_2 = generate_dataset("data/1.5-cutoff/DATA_W14_2/", "dataset/waterforce/", config, 42)
    # featurew14_3 = generate_dataset("data/1.5-cutoff/DATA_W14_3/", "dataset/waterforce/", config, 42)
    # featurew14_4 = generate_dataset("data/1.5-cutoff/DATA_W14_4/", "dataset/waterforce/", config, 42)
    # featurew14_5 = generate_dataset("data/1.5-cutoff/DATA_W14_5/", "dataset/waterforce/", config, 42)
    # featurew14_6 = generate_dataset("data/1.5-cutoff/DATA_W14_6/", "dataset/waterforce/", config, 42)
    # featurew14_7 = generate_dataset("data/1.5-cutoff/DATA_W14_7/", "dataset/waterforce/", config, 42)
    # featurew14_8 = generate_dataset("data/1.5-cutoff/DATA_W14_8/", "dataset/waterforce/", config, 42)
    # featurew14_9 = generate_dataset("data/1.5-cutoff/DATA_W14_9/", "dataset/waterforce/", config, 42)
    # featurew14_10 = generate_dataset("data/1.5-cutoff/DATA_W14_10/", "dataset/waterforce/", config, 42)
    # featurew14_11 = generate_dataset("data/1.5-cutoff/DATA_W14_11/", "dataset/waterforce/", config, 42)
    # featurew15_1 = generate_dataset("data/1.5-cutoff/DATA_W15_1/", "dataset/waterforce/", config, 45)
    # featurew15_2 = generate_dataset("data/1.5-cutoff/DATA_W15_2/", "dataset/waterforce/", config, 45)
    # featurew15_3 = generate_dataset("data/1.5-cutoff/DATA_W15_3/", "dataset/waterforce/", config, 45)
    # featurew15_4 = generate_dataset("data/1.5-cutoff/DATA_W15_4/", "dataset/waterforce/", config, 45)
    # featurew15_5 = generate_dataset("data/1.5-cutoff/DATA_W15_5/", "dataset/waterforce/", config, 45)
    # featurew15_6 = generate_dataset("data/1.5-cutoff/DATA_W15_6/", "dataset/waterforce/", config, 45)
    # featurew15_7 = generate_dataset("data/1.5-cutoff/DATA_W15_7/", "dataset/waterforce/", config, 45)
    # featurew15_8 = generate_dataset("data/1.5-cutoff/DATA_W15_8/", "dataset/waterforce/", config, 45)
    # featurew15_9 = generate_dataset("data/1.5-cutoff/DATA_W15_9/", "dataset/waterforce/", config, 45)
    # featurew15_10 = generate_dataset("data/1.5-cutoff/DATA_W15_10/", "dataset/waterforce/", config, 45)
    # featurew15_11 = generate_dataset("data/1.5-cutoff/DATA_W15_11/", "dataset/waterforce/", config, 45)
    # featurew15_12 = generate_dataset("data/1.5-cutoff/DATA_W15_12/", "dataset/waterforce/", config, 45)
    # featurew15_13 = generate_dataset("data/1.5-cutoff/DATA_W15_13/", "dataset/waterforce/", config, 45)
    # featurew15_14 = generate_dataset("data/1.5-cutoff/DATA_W15_14/", "dataset/waterforce/", config, 45)
    # featurew16_1 = generate_dataset("data/1.5-cutoff/DATA_W16_1/", "dataset/waterforce/", config, 48)
    # featurew16_2 = generate_dataset("data/1.5-cutoff/DATA_W16_2/", "dataset/waterforce/", config, 48)
    # featurew16_3 = generate_dataset("data/1.5-cutoff/DATA_W16_3/", "dataset/waterforce/", config, 48)
    # featurew16_4 = generate_dataset("data/1.5-cutoff/DATA_W16_4/", "dataset/waterforce/", config, 48)
    # featurew16_5 = generate_dataset("data/1.5-cutoff/DATA_W16_5/", "dataset/waterforce/", config, 48)
    # featurew16_6 = generate_dataset("data/1.5-cutoff/DATA_W16_6/", "dataset/waterforce/", config, 48)
    # featurew16_7 = generate_dataset("data/1.5-cutoff/DATA_W16_7/", "dataset/waterforce/", config, 48)
    # featurew16_8 = generate_dataset("data/1.5-cutoff/DATA_W16_8/", "dataset/waterforce/", config, 48)
    # featurew16_9 = generate_dataset("data/1.5-cutoff/DATA_W16_9/", "dataset/waterforce/", config, 48)
    # featurew16_10 = generate_dataset("data/1.5-cutoff/DATA_W16_10/", "dataset/waterforce/", config, 48)
    # featurew16_11 = generate_dataset("data/1.5-cutoff/DATA_W16_11/", "dataset/waterforce/", config, 48)
    # featurew16_12 = generate_dataset("data/1.5-cutoff/DATA_W16_12/", "dataset/waterforce/", config, 48)
    # featurew16_13 = generate_dataset("data/1.5-cutoff/DATA_W16_13/", "dataset/waterforce/", config, 48)
    # featurew16_14 = generate_dataset("data/1.5-cutoff/DATA_W16_14/", "dataset/waterforce/", config, 48)
    # featurew16_15 = generate_dataset("data/1.5-cutoff/DATA_W16_15/", "dataset/waterforce/", config, 48)
    # featurew16_16 = generate_dataset("data/1.5-cutoff/DATA_W16_16/", "dataset/waterforce/", config, 48)
    # featurew16_17 = generate_dataset("data/1.5-cutoff/DATA_W16_17/", "dataset/waterforce/", config, 48)
    # featurew16_18 = generate_dataset("data/1.5-cutoff/DATA_W16_18/", "dataset/waterforce/", config, 48)
    # featurew16_19 = generate_dataset("data/1.5-cutoff/DATA_W16_19/", "dataset/waterforce/", config, 48)
    # featurew16_20 = generate_dataset("data/1.5-cutoff/DATA_W16_20/", "dataset/waterforce/", config, 48)
    # featurew16_21 = generate_dataset("data/1.5-cutoff/DATA_W16_21/", "dataset/waterforce/", config, 48)
    # featurew16_22 = generate_dataset("data/1.5-cutoff/DATA_W16_22/", "dataset/waterforce/", config, 48)
    # featurew16_23 = generate_dataset("data/1.5-cutoff/DATA_W16_23/", "dataset/waterforce/", config, 48)
    # featurew16_24 = generate_dataset("data/1.5-cutoff/DATA_W16_24/", "dataset/waterforce/", config, 48)
    # featurew16_25 = generate_dataset("data/1.5-cutoff/DATA_W16_25/", "dataset/waterforce/", config, 48)
    # featurew17_1 = generate_dataset("data/1.5-cutoff/DATA_W17_1/", "dataset/waterforce/", config, 51)
    # featurew17_2 = generate_dataset("data/1.5-cutoff/DATA_W17_2/", "dataset/waterforce/", config, 51)
    # featurew17_3 = generate_dataset("data/1.5-cutoff/DATA_W17_3/", "dataset/waterforce/", config, 51)
    # featurew17_4 = generate_dataset("data/1.5-cutoff/DATA_W17_4/", "dataset/waterforce/", config, 51)
    # featurew17_5 = generate_dataset("data/1.5-cutoff/DATA_W17_5/", "dataset/waterforce/", config, 51)
    # featurew17_6 = generate_dataset("data/1.5-cutoff/DATA_W17_6/", "dataset/waterforce/", config, 51)
    # featurew17_7 = generate_dataset("data/1.5-cutoff/DATA_W17_7/", "dataset/waterforce/", config, 51)
    # featurew17_8 = generate_dataset("data/1.5-cutoff/DATA_W17_8/", "dataset/waterforce/", config, 51)
    # featurew17_9 = generate_dataset("data/1.5-cutoff/DATA_W17_9/", "dataset/waterforce/", config, 51)
    # featurew17_10 = generate_dataset("data/1.5-cutoff/DATA_W17_10/", "dataset/waterforce/", config, 51)
    # featurew17_11 = generate_dataset("data/1.5-cutoff/DATA_W17_11/", "dataset/waterforce/", config, 51)
    # featurew17_12 = generate_dataset("data/1.5-cutoff/DATA_W17_12/", "dataset/waterforce/", config, 51)
    # featurew17_13 = generate_dataset("data/1.5-cutoff/DATA_W17_13/", "dataset/waterforce/", config, 51)
    # featurew17_14 = generate_dataset("data/1.5-cutoff/DATA_W17_14/", "dataset/waterforce/", config, 51)
    # featurew17_15 = generate_dataset("data/1.5-cutoff/DATA_W17_15/", "dataset/waterforce/", config, 51)
    # featurew17_16 = generate_dataset("data/1.5-cutoff/DATA_W17_16/", "dataset/waterforce/", config, 51)
    # featurew17_17 = generate_dataset("data/1.5-cutoff/DATA_W17_17/", "dataset/waterforce/", config, 51)
    # featurew17_18 = generate_dataset("data/1.5-cutoff/DATA_W17_18/", "dataset/waterforce/", config, 51)
    # featurew17_19 = generate_dataset("data/1.5-cutoff/DATA_W17_19/", "dataset/waterforce/", config, 51)
    # featurew17_20 = generate_dataset("data/1.5-cutoff/DATA_W17_20/", "dataset/waterforce/", config, 51)
    # featurew17_21 = generate_dataset("data/1.5-cutoff/DATA_W17_21/", "dataset/waterforce/", config, 51)

    features = np.vstack((feature1, feature2, feature2_t, feature1_more, feature_rm, feature_rm_e))
    feature2 = np.vstack((feature_1[1], feature_1[2], feature_1[3], feature_1[4], feature_1[5],
                          feature_1[6], feature_1[7], feature_1[8], feature_1[9], feature_1[10], feature_1[11], feature_1[12], feature_1[13], feature_1[14], feature_1[15], feature_1[16],
                          feature_1[17], feature_1[18], feature_1[19], feature_1[20]))
    feature3 = np.vstack((feature_1[0], feature_2[2], feature_2[3], feature_2[4], feature_2[5],
                          feature_2[6], feature_2[7], feature_2[8], feature_2[9], feature_2[10], feature_2[11], feature_2[12], feature_2[13], feature_2[14], feature_2[15], feature_2[16],
                          feature_2[17], feature_2[18], feature_2[19], feature_2[0], feature_2[1]))
    max_values1 = np.max(features, axis=0)
    min_values1 = np.min(features, axis=0)
    max_values2 = np.max(feature2, axis=0)
    min_values2 = np.min(feature2, axis=0)
    max_values3 = np.max(feature3, axis=0)
    min_values3 = np.min(feature3, axis=0)
    # 得到三个最大值数组中的最大值
    max_values = np.max([max_values1, max_values2, max_values3], axis=0)
    # 得到三个最小值数组中的最小值
    min_values = np.min([min_values1, min_values2, min_values3], axis=0)

    np.savetxt("max_values_force_onehot_expand.txt", max_values, delimiter=",")  # 保存最大值到文件
    np.savetxt("min_values_force_onehot_expand.txt", min_values, delimiter=",")  # 保存最小值到文件


# def generate_soap(structure, species, numbers):
#     soap = SOAP(
#         species=species,
#         r_cut=10.0,
#         n_max=10,
#         l_max=5,
#         periodic=False,
#         average="off"
#     )
#     atomic_numbers = structure.get_atomic_numbers()
#     soap_descriptors = soap.create(structure, n_jobs=1)
#     one_hot_encoded = np.zeros((len(atomic_numbers), 2))
#     one_hot_encoded[atomic_numbers == 1, 0] = 1
#     one_hot_encoded[atomic_numbers == 8, 1] = 1
#     soap_descriptors = np.hstack((one_hot_encoded, soap_descriptors))
#     # soap_descriptors = np.hstack((atomic_numbers[:, np.newaxis], soap_descriptors))
#     # soap_descriptors = torch.tensor(soap_descriptors)
#     return soap_descriptors
#
#
# def generate_soap1(pos):
#     soap = SOAP(
#         species=["H", "O"],
#         r_cut=10.0,
#         n_max=10,
#         l_max=5,
#         periodic=False,
#         average="off"
#     )
#     system = Atoms(symbols=["O", "H", "H"], positions=pos)
#     atomic_numbers = np.asarray([8, 1, 1])
#     soap_descriptors = soap.create(system)
#     one_hot_encoded = np.zeros((len(atomic_numbers), 2))
#     one_hot_encoded[atomic_numbers == 1, 0] = 1
#     one_hot_encoded[atomic_numbers == 8, 1] = 1
#     soap_descriptors = np.hstack((one_hot_encoded, soap_descriptors))
#     # soap_descriptors = np.hstack((atomic_numbers[:, np.newaxis], soap_descriptors))
#     # soap_descriptors = torch.tensor(soap_descriptors)
#     return soap_descriptors


def generate_soap(structure, species, numbers):
    soap_fea = []
    soap = SOAP(
        species=species,
        r_cut=10.0,
        n_max=10,
        l_max=5,
        periodic=False,
        average="cc"
    )
    atomic_numbers = structure.get_atomic_numbers()
    for i in range(numbers):
        soap_descriptors = soap.create(structure, n_jobs=1, centers=[i])
        one_hot_encoded = np.zeros(2)
        if atomic_numbers[i] == 1:
            one_hot_encoded = np.array([1, 0])
        if atomic_numbers[i] == 8:
            one_hot_encoded = np.array([0, 1])
        soap_descriptors = np.hstack((one_hot_encoded, soap_descriptors))
        # soap_descriptors = torch.tensor(soap_descriptors)
        soap_fea.append(soap_descriptors)
    # soap_feature = torch.stack(soap_fea)
    # return soap_feature
    return soap_fea

def generate_soap1(pos):
    soap_fea = []
    soap = SOAP(
        species=["H", "O"],
        r_cut=10.0,
        n_max=10,
        l_max=5,
        periodic=False,
        average="cc"
    )
    system = Atoms(symbols=["O", "H", "H"], positions=pos)
    atomic_numbers = [8, 1, 1]
    for i in range(3):
        soap_descriptors = soap.create(system, centers=[i])
        one_hot_encoded = np.zeros(2)
        if atomic_numbers[i] == 1:
            one_hot_encoded = np.array([1, 0])
        if atomic_numbers[i] == 8:
            one_hot_encoded = np.array([0, 1])
        soap_descriptors = np.hstack((one_hot_encoded, soap_descriptors))
        # soap_descriptors = torch.tensor(soap_descriptors)
        soap_fea.append(soap_descriptors)
    # soap_feature = torch.stack(soap_fea)
    return soap_fea

if __name__ == '__main__':
    genergare_all_features()