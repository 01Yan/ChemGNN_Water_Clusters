# import os
# import os.path as osp
# import pickle
# import shutil
# import numpy as np
# import torch
# from tqdm import tqdm
# from numpy.lib.format import read_array
# import h5py
#
# from torch_geometric.data import (
#     Data,
#     InMemoryDataset,
#     download_url,
#     extract_zip,
# )
#
# def load_datasets_from_h5py(raw_dir):
#     train_data = []  # 训练集数据
#     test_data = []  # 测试集数据
#     val_data = []  # 验证集数据
#     with h5py.File(osp.join(raw_dir, "1-H2O_energy.h5"), 'r') as hf:
#         for split in ['train', 'test', 'val']:
#             group = hf[split]
#
#             for group_name in group:
#                 data_group = group[group_name]
#
#                 num_atom = data_group['num_atom'][()]  # 获取标量数据
#                 atom_type_bytes = data_group['atom_type'][()].replace(b'\r', b'\x00')  # 获取 bytes 数据
#                 bond_type_bytes = data_group['bond_type'][()].replace(b'\r', b'\x00')  # 获取 bytes 数据
#                 logP_SA_cycle_normalized_bytes = data_group['logP_SA_cycle_normalized'][()].replace(b'\r', b'\x00')  # 获取 bytes 数据
#                 z_bytes = data_group['z'][()].replace(b'\r', b'\x00')  # 获取 bytes 数据
#
#                 # 将 bytes 数据还原为 torch.Tensor
#                 atom_type = torch.from_numpy(np.frombuffer(atom_type_bytes, dtype=np.float64).reshape(-1, 1260))
#                 bond_type = torch.from_numpy(np.frombuffer(bond_type_bytes, dtype=np.float64).reshape(-1, 3))
#                 logP_SA_cycle_normalized = torch.from_numpy(np.frombuffer(logP_SA_cycle_normalized_bytes, dtype=np.float32).reshape(-1, 1))
#                 z = torch.from_numpy(np.frombuffer(z_bytes, dtype=np.float32).reshape(3, 3))
#
#                 data_item = {
#                     'num_atom': num_atom,
#                     'atom_type': atom_type,
#                     'bond_type': bond_type,
#                     'logP_SA_cycle_normalized': logP_SA_cycle_normalized,
#                     'z': z
#                 }
#
#                 # 将数据项添加到对应的集合中
#                 if split == 'train':
#                     train_data.append(data_item)
#                 elif split == 'test':
#                     test_data.append(data_item)
#                 elif split == 'val':
#                     val_data.append(data_item)
#     return train_data, test_data, val_data
#
# def load_datasets_from_npy(raw_dir):
#
#     datasets_val = np.load(osp.join(raw_dir, "val.pickle.npy"), allow_pickle=True)
#     datasets_test = np.load(osp.join(raw_dir, "test.pickle.npy"), allow_pickle=True)
#     datasets_train = np.load(osp.join(raw_dir, "train.pickle.npy"), allow_pickle=True)
#
#     # datasets = np.load(osp.join(raw_dir, "all.npy"), allow_pickle=True)
#     return datasets_train, datasets_test, datasets_val
#
# class MyDataset(InMemoryDataset):
#     url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
#     split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
#                  'benchmarking-gnns/master/data/molecules/{}.index')
#
#     def __init__(self, root, subset=False, split='train', transform=None,
#                  pre_transform=None, pre_filter=None):
#         self.subset = subset
#         assert split in ['train', 'val', 'test']
#         super().__init__(root, transform, pre_transform, pre_filter)
#         path = osp.join(self.processed_dir, f'{split}.pt')
#         self.data, self.slices = torch.load(path)
#         # print("1")
#
#     @property
#     def raw_file_names(self):
#         return [
#             # 'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
#             'train_h2o', 'val_h2o', 'test_h2o', 'train.index',
#             'val.index', 'test.index'
#         ]
#
#     @property
#     def processed_dir(self):
#         name = 'subset' if self.subset else 'full'
#         return osp.join(self.root, name, 'processed')
#
#     @property
#     def processed_file_names(self):
#         return ['train.pt', 'val.pt', 'test.pt']
#
#     def download(self):
#         # print("Data not found!")
#         pass
#         #raise RuntimeError
#         #shutil.rmtree(self.raw_dir)
#         #path = download_url(self.url, self.root)
#         #extract_zip(path, self.root)
#         #os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
#         #os.unlink(path)
#
#         #for split in ['train', 'val', 'test']:
#         #    download_url(self.split_url.format(split), self.raw_dir)
#
#
#
#     def process(self):
#
#         datasets_train, datasets_test, datasets_val = load_datasets_from_npy(self.raw_dir)
#         # datasets_train, datasets_test, datasets_val = load_datasets_from_h5py(self.raw_dir)
#
#         for split, data_list in [('train', datasets_train), ('val', datasets_val), ('test', datasets_test)]:
#             print("finding", osp.join(self.raw_dir, f'{split}.npy'))
#
#             # 遍历数据并转换为 PyTorch 的 Data 对象
#             pbar = tqdm(total=len(data_list))
#             pbar.set_description(f'Processing {split} dataset')
#
#             data_list_pt = []  # 保存转换后的 PyTorch Data 对象
#             for mol in data_list:
#                 # mol = data.item()
#
#                 x = torch.tensor(mol['atom_type'], dtype=torch.float32).view(-1, mol['atom_type'].shape[1])
#                 y = torch.tensor(mol['logP_SA_cycle_normalized'], dtype=torch.float32)
#                 z = torch.tensor(mol['z'], dtype=torch.float32)
#
#                 adj = torch.tensor(mol['bond_type'], dtype=torch.float32)
#                 edge_index = adj.nonzero(as_tuple=False).t().contiguous()
#                 edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
#
#                 data_pt = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z)
#
#                 if self.pre_filter is not None and not self.pre_filter(data_pt):
#                     continue
#
#                 if self.pre_transform is not None:
#                     data_pt = self.pre_transform(data_pt)
#
#                 data_list_pt.append(data_pt)
#                 pbar.update(1)
#
#             pbar.close()
#
#             # 保存数据为 .pt 文件（使用 PyTorch 的 torch.save() 函数）
#             torch.save(self.collate(data_list_pt), osp.join(self.processed_dir, f'{split}.pt'))

        # for split in ['train', 'val', 'test']:
        #     print("finding", osp.join(self.raw_dir, f'{split}.pickle'))
        #
        #     with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
        #         mols = pickle.load(f)
        #
        #     indices = range(len(mols))
        #
        #     if self.subset:
        #         with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
        #             indices = [int(x) for x in f.read()[:-1].split(',')]
        #
        #     pbar = tqdm(total=len(indices))
        #     pbar.set_description(f'Processing {split} dataset')
        #
        #     data_list = []
        #     for idx in indices:
        #         mol = mols[idx]
        #
        #
        #         x = mol['atom_type'].to(torch.float32).view(-1, mol['atom_type'].shape[1]) #  x = mol['atom_type'].to(torch.long).view(-1, 1)
        #         y = mol['logP_SA_cycle_normalized'].to(torch.float)
        #         z = mol['z'].to(torch.float)
        #
        #         adj = mol['bond_type']
        #         edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        #         edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)  # torch.long
        #         # print("x:", x)
        #         data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
        #                     y=y,z=z)
        #
        #         if self.pre_filter is not None and not self.pre_filter(data):
        #             continue
        #
        #         if self.pre_transform is not None:
        #             data = self.pre_transform(data)
        #
        #         data_list.append(data)
        #         pbar.update(1)
        #
        #     pbar.close()
        #
        #     torch.save(self.collate(data_list),
        #                osp.join(self.processed_dir, f'{split}.pt'))
            # print(2)

# import os.path as osp
# import pickle
# import shutil
# import numpy as np
# import torch
# from tqdm import tqdm
# # from Full_H2O_utils import *
# # from config import config_1
# from config import config
#
# from torch_geometric.data import (
#     Data,
#     InMemoryDataset,
#     download_url,
#     extract_zip,
# )
#
# class MyDataset(InMemoryDataset):
#     url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
#     split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
#                  'benchmarking-gnns/master/data/molecules/{}.index')
#
#     def __init__(self, root, split='1-H2O', transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(root, transform, pre_transform, pre_filter)
#         path = osp.join(self.processed_dir, f'{split}.pt')
#         self.data, self.slices = torch.load(path)
#         # print("1")
#
#     @property
#     def raw_file_names(self):
#         return ['train.pickle.npy']
#
#     @property
#     def processed_dir(self):
#         # name = 'subset' if self.subset else 'full'
#         return osp.join(self.root, 'full', 'processed')
#
#     @property
#     def processed_file_names(self):
#         return ['1-H2O.pt', '2-H2O.pt', '3-H2O.pt', '4-H2O.pt', '5-H2O.pt', '6-H2O.pt', '7-H2O.pt', '8-H2O.pt', '9-H2O.pt', '10-H2O.pt',
#                 '11-H2O.pt', '12-H2O.pt', '13-H2O.pt', '14-H2O.pt', '15-H2O.pt', '16-H2O.pt', '17-H2O.pt', '18-H2O.pt', '19-H2O.pt', '20-H2O.pt', '21-H2O.pt', '216-H2O.pt',
#                 '1-H2O_test.pt', 'full-H2O.pt', 'full-H2O_test.pt', 'Full.pt', 'Full_test.pt', '1.pt', '1_test.pt', 'random.pt']
#
#     def download(self):
#         pass
#
#     def process(self):
#         pass

import os
import os.path as osp
import pickle
import shutil
import numpy as np
import torch
from tqdm import tqdm
from process_dataset import *
# from utils_216H2O import *
# from utils import *
from config import *


from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class MyDataset(InMemoryDataset):
    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root, split='1-H2O', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        # print("1")

    @property
    def raw_file_names(self):
        return ['train.pickle.npy']

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        return osp.join(self.root, 'full', 'processed')

    @property
    def processed_file_names(self):
        return ['1-H2O.pt', '2-H2O.pt', '3-H2O.pt', '4-H2O.pt', '5-H2O.pt', '6-H2O.pt', '7-H2O.pt', '8-H2O.pt', '9-H2O.pt', '10-H2O.pt',
                '11-H2O.pt', '12-H2O.pt', '13-H2O.pt', '14-H2O.pt', '15-H2O.pt', '16-H2O.pt', '17-H2O.pt', '18-H2O.pt', '19-H2O.pt', '20-H2O.pt', '21-H2O.pt', '216-H2O.pt']

    def download(self):
        pass

    def process(self):
        pass
        # for i in range(2, 22):
        #     datasets = generate_dataset(f"data/random_full/{i}WATER", "dataset/waterforce/", config_data)

            # datasets = generate_dataset("data/DATA 1/", "dataset/waterforce/", config_data, True)
            # datasets = generate_dataset("data/Full_data/", "dataset/waterforce/", config_data)
            # datasets = generate_dataset("data/Full_test_data/", "dataset/waterforce/", config_data)
            # datasets = generate_dataset("data/1-21-train", "dataset/waterforce/", config_4)
            # datasets = generate_dataset("data/random_full", "dataset/waterforce/", config_4)
            # datasets = generate_dataset("data/random_expand", "dataset/waterforce/", config_4)
            # datasets = generate_dataset("data/random_expand/1WATER", "dataset/waterforce/", config_5)
            # datasets = generate_dataset("data/1.8newdata/WATER_2", "dataset/waterforce/", config_6)
        #
        #
        #     # datasets = generate_dataset("data/data-216H2O/", "dataset/waterforce/", config_216)
        #     # datasets = generate_dataset("data/data_21H2O/", "dataset/waterforce/", config)
        #
        #     # datasets = generate_dataset("data/data-216H2O/", "dataset/waterforce/", config_216)
        #     # datasets_train, datasets_test, datasets_val = load_datasets_from_npy(self.raw_dir)
        #
            # for split, data_list in [('train', datasets_train), ('val', datasets_val), ('test', datasets_test)]:

            #     # 遍历数据并转换为 PyTorch 的 Data 对象
            # pbar = tqdm(total=len(datasets))
            # pbar.set_description(f'Processing 1-H2O dataset')
            #
            # data_list_pt = []  # 保存转换后的 PyTorch Data 对象
            # for idx, mol in enumerate(datasets):
            #     # mol = data.item()
            #     try:
            #         x = torch.tensor(mol['atom_type'], dtype=torch.float32).view(-1, mol['atom_type'].shape[1])
            #         y = torch.tensor(mol['logP_SA_cycle_normalized'], dtype=torch.float64)
            #         z = torch.tensor(mol['z'], dtype=torch.float32)
            #
            #         adj = torch.tensor(mol['bond_type'], dtype=torch.float32)
            #         edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            #         edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
            #         assert z.shape[0] == edge_index.max().item()+1
            #     except AssertionError:
            #         print(f"Data at index {idx} has an issue!")
            #     data_pt = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z)
            #
            #     if self.pre_filter is not None and not self.pre_filter(data_pt):
            #         continue
            #
            #     if self.pre_transform is not None:
            #         data_pt = self.pre_transform(data_pt)
            #
            #     data_list_pt.append(data_pt)
            #     pbar.update(1)
            #
            # pbar.close()
            #
            # # 保存数据为 .pt 文件（使用 PyTorch 的 torch.save() 函数）
            # torch.save(self.collate(data_list_pt), osp.join(self.processed_dir, f'Random3_{i}water_force.pt'))
if __name__ == '__main__':
    train_dataset = MyDataset('/Users/yanhongyu/Documents/课程资料/summer_project/soap_1-21-water_energy/1/dataset/waterforce')
    # train_dataset = MyDataset('/Users/yanhongyu/Desktop/SOAP_ChemGNN-21H2O/dataset/waterforce')