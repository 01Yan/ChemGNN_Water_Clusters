#from torch_scatter import scatter
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
from config import config
from dscribe.descriptors import SOAP
from ase import Atoms
from torch.utils import data
from torch_geometric.nn import BatchNorm, global_add_pool
from ChemGNN import CEALConv
import torch.nn as nn
import torch.nn.init as init
# import seaborn as sns

def split_dataset(dataset, train_p=60, val_p=20, shuffle=True):
   """
   dataset: dataset to split
   train_p: training set percentage
   val_p: validating set percentage

   """

   if shuffle:
      ############## Change this if not to seek reproducibility ####
      #random.seed(218632)
      idx = list(range(dataset.len()))
      random.shuffle(idx)
      idx = np.array(idx)
   else:
      idx = np.array(range(dataset.len()))

   len_train = int(len(idx) * train_p)
   len_val = int(len(idx) * val_p)

   train_dataset = dataset[idx[0:len_train]]
   val_dataset = dataset[idx[len_train:len_train+len_val]]
   test_dataset = dataset[idx[len_train+len_val:]]
   return train_dataset, val_dataset, test_dataset

def split_dataset_balance(dataset, train_len=501, val_len=200, test_len=300):
    idx = np.array(range(dataset.len()))
    t = dataset.len() / (train_len + test_len + val_len)
    train_dataset = []
    test_dataset = []
    val_dataset = []
    for i in range(int(t)):
        train_dataset.append(dataset[idx[i*(train_len+val_len+test_len) : i*(train_len+val_len+test_len)+train_len]])
        test_dataset.append(dataset[idx[i*(train_len+val_len+test_len)+train_len : i*(train_len+val_len+test_len)+train_len+test_len]])
        val_dataset.append(dataset[idx[i*(train_len+val_len+test_len)+train_len+test_len : (i+1)*(train_len+val_len+test_len)]])
    train = data.ConcatDataset(train_dataset)
    test = data.ConcatDataset(test_dataset)
    val = data.ConcatDataset(val_dataset)
    return train, test, val


def draw_two_dimension(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :return:
    """
    assert len(y_lists[0]) == len(x_list), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_two_dimension_regression(
        y_lists,
        x_lists,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=20,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=20,
        y_label_size=20,
        number_label_size=20,
        fig_size=(8, 6)
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value
    :param x_lists: (list[list]) x value
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :return:
    """
    y_count = len(y_lists)
    for i in range(y_count):
        assert len(y_lists[i]) == len(x_lists[i]), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(x_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"

    plt.figure(figsize=fig_size)
    for i in range(y_count):
        # plt.plot(x_lists[i], y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
        fit = np.polyfit(x_lists[i], y_lists[i], 1)
        line_fn = np.poly1d(fit)
        y_line = line_fn(x_lists[i])
        plt.scatter(x_lists[i], y_lists[i])
        plt.plot(x_lists[i], y_line, markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    #plt.tight_layout()
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def tensor_min_max_scaler(data, new_min=0.0, new_max=1.0):
    assert isinstance(data, torch.Tensor)
    data_min = torch.min(data,0).values
    data_max = torch.max(data,0).values
    #data_min=np.min(data.reshape(-1,1).tolist())
    #data_max = np.max(data.reshape(-1, 1).tolist())
    #assert 0.0 not in (data_max - data_min)
    core = (data - data_min) / (data_max - data_min)
    regre=data_max-data_min
    #if regre[3]==0.0000:
    #    core[:,3]=1.0000
    #core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new


def numpy_min_max_scaler_1d(data, new_min=0.0, new_max=1.0):
    assert isinstance(data, np.ndarray)
    data_min = np.min(data)
    data_max = np.max(data)
    assert data_max - data_min > 0
    core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new

# -361.77515914 -17.19880547 -3721.31782775 -3720.69049016
def reverse_min_max_scaler_1d(data_normalized, data_min=-361.77515914, data_max=-17.19880547, new_min=0.0,
                                  new_max=1.0):

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


def load_one_charge(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    # print(file_path)
    data = [line.split()[1:] for line in data]
    data = np.asarray(data, dtype=float)
    lengths = [len(item) for item in data]
    assert min(lengths) == max(lengths) == 7, "load data error in file {}".format(file_path)
    return data


def load_one_coordinate(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    # print(data.shape, data)
    return data


def generate_gap_file(folder, save_path, length, file_format, L, overwrite_flag=False):
    # files = os.listdir(folder)
    if osp.exists(save_path) and not overwrite_flag:
        print("Gap file {} exists. Skip generating ...".format(save_path))
        gaps = np.load(save_path)
        return gaps
    else:
        print("{}: {} files".format(folder, length))
        # files.sort()
        gaps = []
        # print(files[:30])
        for i in range(length):
            j=L[i]
            filename = osp.join(folder, file_format.format(j))
            with open(filename, "r") as f:
                lines = f.readlines()
                one_gap = float(lines[0])
            gaps.append(one_gap)
        gaps = np.asarray(gaps)
    min = np.min(gaps)
    max = np.max(gaps)
    '''
    min_gaps=np.min(gaps)
    max_gaps=np.max(gaps)
    gap_s=[]
    for i in range(length):
        core = (gaps[i] - min_gaps) / (max_gaps - min_gaps)
        gap_s.append(core)
    gaps=np.asarray(gap_s)
    '''
    # np.save(save_path, gaps)
    return gaps, max, min

def generate_feature(folder, length, file_format, L):
    features = []
    for i in tqdm(range(length)):
        j = L[i]
        c_path = osp.join(folder, file_format.format(j))
        c_data = load_one_coordinate(c_path)
        one_feature = generate_soap(c_data)
        features.append(one_feature)

    features = np.asarray(features)
    features = features.reshape(-1, 1260)
    max_values = np.max(features, axis=0)  # 沿着列的方向计算最大值
    min_values = np.min(features, axis=0)  # 沿着列的方向计算最小值
    np.savetxt("/Users/yanhongyu/Desktop/SOAP_ChemGNN-master/max_values_force.txt", max_values, delimiter=",")  # 保存最大值到文件
    np.savetxt("/Users/yanhongyu/Desktop/SOAP_ChemGNN-master/min_values_force.txt", min_values, delimiter=",")  # 保存最小值到文件
    for i in range(1260):
        features[:, i] = numpy_min_max_scaler_1d(features[:, i])
    return features

def generate_atom_edges(edge_index_0, atom_num):
    item_list, item_count = edge_index_0.unique(return_counts=True)
    dic = {int(item_list[i]): int(item_count[i]) for i in range(item_list.shape[0])}
    atom_edges_array = torch.tensor([[dic[i]] if i in dic else [0] for i in range(atom_num)], dtype=torch.float32)
    # print(atom_edges_array)
    return atom_edges_array

def broadcast( src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def generate_dataset(input_path, output_path, config):
    assert osp.exists(input_path)
    if not osp.exists(osp.join(output_path, "raw")):
        print("Created new folder: {}".format(osp.join(output_path, "raw")))
        os.makedirs(osp.join(output_path, "raw"))
    random_list=[]
    for i in range(config.length):
        random_list.append(random.randint(0, config.length))
    random_list=list(range(config.length))
    random.shuffle(random_list)
    gaps, max, min = generate_gap_file(osp.join(input_path, "MLENERGY"), osp.join(input_path, "{}_gaps.npy".format(config.dataset)), config.length, config.format_eigen, random_list, overwrite_flag=True)
    gaps = numpy_min_max_scaler_1d(gaps)
    features = generate_feature(osp.join(input_path, "CONFIGS"), config.length, config.format_conf, random_list)

    datasets = []
    for i in tqdm(range(config.length)):
        j = random_list[i]
        bt_path = osp.join(input_path, "BTMATRIXES", config.format_bmat.format(j))
        bt_data = load_one_map(bt_path)
        bt_data = torch.tensor(bt_data)
        a_path = osp.join(input_path, "AIFORCE", config.format_force.format(j))
        a_data = load_one_coordinate(a_path)
        a_data = torch.tensor(a_data,dtype=torch.float32)
        d_path = osp.join(input_path, "DMATRIXES", config.format_dmat.format(j))
        d_data = load_one_map(d_path)
        d_data = torch.tensor(d_data, dtype=torch.float32)
        c_path = osp.join(input_path, "CONFIGS", config.format_conf.format(j))
        c_data = load_one_coordinate(c_path)
        c_data = torch.tensor(c_data, dtype=torch.float32)
        at_data=torch.tensor([8,1,1],dtype=torch.float32)
        at_data=at_data.resize(3,1)
        #charge_path = osp.join(input_path, "CHARGES", config.format_charge.format(j + 1))
        #charge_data = load_one_charge(charge_path)
        #charge_data = charge_data[:, [0, 2, 3, 4, 5, 6]]
        #charge_data = torch.tensor(charge_data)




        #matrix_data = d_data * bt_data  # 4.0 * (d_data - 1.2) * bt_data
        matrix_data =bt_data*d_data

        edge_index = bt_data.nonzero().t().contiguous()
        # edge_attr_from_dis = d_data[edge_index[0], edge_index[1]].to(torch.float32)
        # edge_attr = torch.tensor([[edge_attr_from_dis[i], 0.0, 0.0, 0.0] for i in range(edge_attr_from_dis.shape[0])],
        #                          dtype=torch.float32)
        node_degree = generate_atom_edges(edge_index[0], config.max_natoms)
        assert node_degree.shape[0] == c_data.shape[0], "Need to match max_natoms!"

        atom_num_array = [6] * 54 + [7] * 71 + [15] * 1
        electron_num_array = [4] * 54 + [5] * 72
        atom_type_dict = {
            6: [1, 0, 0],
            7: [0, 1, 0],
            15: [0, 0, 1]
        }
        atom_type = torch.Tensor([atom_type_dict.get(item) for item in atom_num_array])
        atomic_number = torch.Tensor([[item] for item in atom_num_array])
        electron_number = torch.Tensor([[item] for item in electron_num_array])

        # x_full = torch.cat([c_data,at_data], 1)  # pos 3 cols + atom edge 1 col = 4 cols of feature + 6 cols of charges
        # x_full = generate_soap(c_data)
        x_full = features[i * config.max_natoms: (i + 1) * config.max_natoms, :]
        x_full = torch.tensor(x_full)
        # for j in range(3):
        #     x_full[j] = tensor_min_max_scaler(x_full[j])

        #x_full = tensor_min_max_scaler(x_full)
        #force = tensor_min_max_scaler(a_data)
        # x_full = torch.cat([c_data, node_degree, atom_type, atomic_number, electron_number], 1)  # 9 cols of feature


        # np.set_printoptions(threshold=np.inf)
        # print(matrix_data)
        # print(bt_data[0])
        # print(d_data[0])
        # for ii in range(126):
        #     print("####" if bt_data[1][ii] == 1 else "", bt_data[1][ii], d_data[1][ii])
        # atom_num_array = torch.Tensor([[6]] * 54 + [[7]] * 71 + [[15]] * 1)
        # x_full = torch.cat([c_data, atom_num_array], 1)

        '''
        one_data = {
            'num_atom': config.max_natoms,
            'atom_type': x_full,
            'bond_type': matrix_data,
            'logP_SA_cycle_normalized': torch.Tensor([gaps[i]])
        }
        '''
        one_data = {
            'num_atom': config.max_natoms,
            'atom_type': x_full,
            'bond_type': matrix_data,
            'logP_SA_cycle_normalized': torch.Tensor([gaps[i]]),
            'z': a_data
        }
        datasets.append(one_data)
    # return datasets
        # print(one_data['logP_SA_cycle_normalized'])
    print("Finished! Train: {} Test: {} Val: {}".format(config.train_length, config.test_length, config.val_length))
    datasets_train = datasets[: config.train_length]
    datasets_test = datasets[config.train_length: config.train_length + config.test_length]
    datasets_val = datasets[config.train_length + config.test_length:]
    with open(osp.join(output_path, "raw/train.pickle"), "wb") as f:
        pickle.dump(datasets_train, f)
    with open(osp.join(output_path, "raw/test.pickle"), "wb") as f:
        pickle.dump(datasets_test, f)
    with open(osp.join(output_path, "raw/val.pickle"), "wb") as f:
        pickle.dump(datasets_val, f)
    with open(osp.join(output_path, "raw/all.pickle"), "wb") as f:
        pickle.dump(datasets, f)
    np.save(osp.join(output_path, "raw/train.pickle"), datasets_train, allow_pickle=True)
    np.save(osp.join(output_path, "raw/test.pickle"), datasets_test, allow_pickle=True)
    np.save(osp.join(output_path, "raw/val.pickle"), datasets_val, allow_pickle=True)
    np.save(osp.join(output_path, "raw/all.pickle"), datasets, allow_pickle=True)
    return max, min

def generate_soap(pos):
    soap = SOAP(
        species=["H", "O"],
        r_cut=10.0,
        n_max=10,
        l_max=5,
        periodic=False,
        average="off"
    )
    system = Atoms(symbols=["O", "H", "H"], positions=pos)

    soap_descriptors = soap.create(system)
    # soap_descriptors = torch.tensor(soap_descriptors)
    return soap_descriptors
# def generate_soap(pos):
#     soap = SOAP(
#         species=["H", "O"],
#         r_cut=10.0,
#         n_max=10,
#         l_max=5,
#         periodic=False,
#         average="outer"
#     )
#     tem = []
#     system = Atoms(symbols=["O", "H", "H"], positions=pos)
#     for iatom in range(3):
#         soap_descriptors = soap.create(system, centers=[iatom])
#         soap_descriptors = torch.tensor(soap_descriptors)
#         tem.append(soap_descriptors)
#     final_soap_descriptors = torch.stack(tem)
#     return final_soap_descriptors


def generate_fept_dataset(input_path, output_path, config):
    assert osp.exists(input_path)
    if not osp.exists(output_path):
        print("Created new folder: {}".format(output_path))
        os.makedirs(output_path)
    gaps = generate_gap_file(osp.join(input_path, "EIGENVALS_A"), osp.join(input_path, "{}_gaps.npy".format(config.dataset)), config.length, config.format_eigen)
    # datasets = []
    for i in tqdm(range(config.length)):
        bt_path = osp.join(input_path, "BTMATRIXES", config.format_bmat.format(i + 1))
        bt_data = load_one_map(bt_path)
        bt_data = torch.tensor(bt_data)
        d_path = osp.join(input_path, "DMATRIXES", config.format_dmat.format(i + 1))
        d_data = load_one_map(d_path)
        # d_data = torch.tensor(d_data)
        c_path = osp.join(input_path, "CONFIGS", config.format_conf.format(i + 1))
        c_data_raw = load_one_coordinate(c_path)
        c_data = torch.tensor(c_data_raw, dtype=torch.float32)
        # matrix_data = d_data * bt_data  # 4.0 * (d_data - 1.2) * bt_data

        edge_index = bt_data.nonzero().t().contiguous()
        edge_attr_from_dis = d_data[edge_index[0], edge_index[1]]

        node_degree = generate_atom_edges(edge_index[0], config.max_natoms)
        assert node_degree.shape[0] == c_data.shape[0], "Need to match max_natoms!"

        atom_num_array = [6] * 54 + [7] * 71 + [15] * 1
        electron_num_array = [4] * 54 + [5] * 72
        atom_type_dict = {
            6: [1, 0, 0],
            7: [0, 1, 0],
            15: [0, 0, 1]
        }
        atom_type = torch.Tensor([atom_type_dict.get(item) for item in atom_num_array])
        atomic_number = torch.Tensor([[item] for item in atom_num_array])
        electron_number = torch.Tensor([[item] for item in electron_num_array])

        # x_full = torch.cat([c_data, node_degree], 1)  # pos 3 cols + atom edge 1 col = 4 cols of feature
        x_full = torch.cat([c_data, node_degree, atom_type, atomic_number, electron_number], 1)  # 9 cols of feature

        node_distance_dict = dict()
        for j in range(126):
            node_distance_dict[j] = []
        for j in range(edge_attr_from_dis.shape[0]):
            node_distance_dict[edge_index.cpu().detach().numpy()[0][j]] += [edge_attr_from_dis[j]]
            node_distance_dict[edge_index.cpu().detach().numpy()[1][j]] += [edge_attr_from_dis[j]]
        # node_distance_avg = [sum(node_distance_dict[j]) / len(node_distance_dict[j]) for j in range(126)]
        node_distance_avg = [np.asarray(node_distance_dict[j]).mean() for j in range(126)]
        # print(node_distance_dict[2])
        # print("{:.12f}".format(node_distance_avg[2]))

        # print([len(node_distance_dict[j]) for j in range(126)])

        one_data = {
            "atom_num": atom_num_array,
            "atom_index": range(126),
            "x": c_data_raw[:, 0],
            "y": c_data_raw[:, 1],
            "z": c_data_raw[:, 2],
            "node_degree": [int(item[0]) for item in node_degree.cpu().detach().numpy()],
            "electron_num": [int(item[0]) for item in electron_number.cpu().detach().numpy()],
            "atom_type_0": [int(item[0]) for item in atom_type.cpu().detach().numpy()],
            "atom_type_1": [int(item[1]) for item in atom_type.cpu().detach().numpy()],
            "atom_type_2": [int(item[2]) for item in atom_type.cpu().detach().numpy()],
            "node_distance_avg": node_distance_avg,
        }

        one_data_format = {
            "atom_num": "{}",
            "atom_index": "{}",
            "x": "{:.15f}",
            "y": "{:.15f}",
            "z": "{:.15f}",
            "node_degree": "{}",
            "electron_num": "{}",
            "atom_type_0": "{}",
            "atom_type_1": "{}",
            "atom_type_2": "{}",
            "node_distance_avg": "{:.15f}",
        }

        one_data_format_rjust = {
            "atom_num": 3,
            "atom_index": 3,
            "x": 18,
            "y": 18,
            "z": 18,
            "node_degree": 3,
            "electron_num": 3,
            "atom_type_0": 3,
            "atom_type_1": 3,
            "atom_type_2": 3,
            "node_distance_avg": 18,
        }

        one_data_y = gaps[i]

        one_data_keys = ["atom_num", "atom_index", "x", "y", "z", "node_degree", "electron_num", "atom_type_0", "atom_type_1", "atom_type_2", "node_distance_avg"]

        string = ""
        string += "{:.15f}".format(one_data_y) + "\n"
        for j in range(126):
            string += "\t".join([one_data_format[one_key].format(one_data[one_key][j]).rjust(one_data_format_rjust[one_key]) for one_key in one_data_keys])
            string += "\n"
        with open(osp.join(output_path, "out_{}".format(i)), "w") as f:
            f.write(string)

        # datasets.append(one_data)
    # print("Finished! Train: {} Test: {} Val: {}".format(config.train_length, config.test_length, config.val_length))
    # datasets_train = datasets[: config.train_length]
    # datasets_test = datasets[config.train_length: config.train_length + config.test_length]
    # datasets_val = datasets[config.train_length + config.test_length:]
    # with open(osp.join(output_path, "raw/train.pickle"), "wb") as f:
    #     pickle.dump(datasets_train, f)
    # with open(osp.join(output_path, "raw/test.pickle"), "wb") as f:
    #     pickle.dump(datasets_test, f)
    # with open(osp.join(output_path, "raw/val.pickle"), "wb") as f:
    #     pickle.dump(datasets_val, f)
    # np.save(osp.join(output_path, "raw/train.pickle"), datasets_train, allow_pickle=True)
    # np.save(osp.join(output_path, "raw/test.pickle"), datasets_test, allow_pickle=True)
    # np.save(osp.join(output_path, "raw/val.pickle"), datasets_val, allow_pickle=True)

def worker_init_fn(worker_id, seed=0):
    random.seed(seed + worker_id)


def generate_deg(dataset):
    max_degree = -1
    for data in dataset:
        # print("data.num_nodes:", data.num_nodes)
        # print("data.edge_index[1]:", data.edge_index[1].shape)
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def compute_correlation(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0, len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar ** 2
        varY += difYYbar ** 2
    SST = math.sqrt(varX * varY)
    if SST == 0.0:
        return -1
    return SSR / SST

def Metric_for_Smoothness(H,M):
    # calculate_cosine_distance_matrix
    n = H.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_distance = 1 - np.dot(H[i, :], H[j, :]) / (np.linalg.norm(H[i, :]) * np.linalg.norm(H[j, :]))
            D[i, j] = cos_distance
    # print("D",D)
    # element_wise_multiplication
    if D.shape != M.shape:
        raise print("Shapes of D and M must be the same.")
    D_tgt = D * M
    # print("D_tgt",D_tgt)
    # average_Distence
    row_sums = np.sum(D_tgt, axis=1)
    row_counts = np.sum(D_tgt > 0, axis=1)
    ave_D_tgt = row_sums / row_counts
    # print("ave_D_tgt",ave_D_tgt)
    # calculate_mad
    sum_D_tgt = np.sum(ave_D_tgt)
    counts = np.sum(ave_D_tgt > 0)
    MAD_tgt = sum_D_tgt / counts
    return MAD_tgt

def save_checkpoint(skp_path, model, epoches, optimizer, global_step):
    checkpoint = {'epoch': epoches,
                  'global_step': global_step,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, skp_path)

# Kaiming初始化的函数
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
def custom_std(data):
    """
    计算类似于标准差的度量，但是采用四次方和四次方根。
    data: 包含数据点的数组或列表
    """
    mean = np.mean(data)
    sum_of_fourth_powers = np.sum((data - mean) ** 4)
    n = len(data)
    return (sum_of_fourth_powers / n) ** (1 / 4)

if __name__ == "__main__":
    # data = pd.read_csv("../data/ZINC/raw/atom_dict.pickle")
    # print(data)
    # with open("../data/ZINC/raw/atom_dict.pickle", "rb") as f:
    #     data_atom = pickle.load(f)
    # print(type(data_atom), data_atom.shape)
    # print(data_atom)

    # with open("../data/ZINC/raw/train.pickle", 'r') as f:
    #     data = [x.split('\t') for x in f.read().split('\n')[1:-1]]
    #
    #     rows, cols = [], []
    #     for n_id, col, _ in data:
    #         col = [int(x) for x in col.split(',')]
    #         rows += [int(n_id)] * len(col)
    #         cols += col
    #     x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
    #     x = x.to_dense()
    #
    # print(x)
    # data = np.load("../data/ZINC/raw/test.pickle", allow_pickle=True)
    #
    # print(len(data))

    # with open("../data/GCN_N3P/raw/train.pickle", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))
    # # data = np.load("../data/ZINC/raw/test.index", allow_pickle=True)
    # # with open("../data/ZINC/raw/val.index", 'r') as f:
    # #     data = f.readline()
    # # print(data)
    # # print(len(data))
    # # data = data.split(",")
    # # print(data)
    # # print(len(data))
    # #
    # # data = np.load("../data/ZINC/raw/val.pickle", allow_pickle=True)
    # print(type(data))
    # print(len(data))
    # for i in range(5):
    #     # print(type(data[i]))
    #     # print(data[i].keys())
    # # print(type(data[-1]))
    # # print(data[-1].keys())
    #     if data[i]["num_atom"] != 24:
    #         print(i)
    #         print(data[i])

    # with open("../data/ZINC/raw/val.index", 'w') as f:
    #     f.write("abcdefg")
    # generate_dataset("../../MLGCN/data/GCN_N3P/")
    # load_one_coordinate("../../MLGCN/data/GCN_N3P/CONFIGS/COORD_1")
    # generate_gap_file("../../MLGCN/data/GCN_N3P/EIGENVALS/", "../../MLGCN/data/GCN_N3P/GCN_N3P_gaps.npy")
    # a = [1,2,3,4,5,6,7]
    #
    # print(a[slice([2,4,5])])
    # print(type({"1":222, "2":333}))

    max, min = generate_dataset("data/waterforce/", "dataset/waterforce/", config)
    print(max)
    print(min)
    # with open("dataset/waterforce/raw/val.pickle", "rb") as f:
    #     a = pickle.load(f)
    #     print(a)

