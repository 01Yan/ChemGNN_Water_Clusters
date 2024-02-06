import os.path as osp
import math
from utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
# from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, global_add_pool
from ChemGNN import CEALConv
from torch_geometric.utils import degree
import time

from dataset import MyDataset
from tqdm import tqdm


class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.edge_emb = Embedding(20, 10)  # self.edge_emb = Embedding(4, 50)

        self.conv_num = 1
        self.in_num = 1262
        self.CP2K_flag = False

        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        scalers = ['identity']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.std_weight = nn.Parameter(torch.rand(1))

        for _ in range(self.conv_num):
            if _ == 0:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, scalers=scalers,
                                edge_dim=10, towers=1, pre_layers=5, post_layers=5,
                                divide_input=False)  # model1/model1
                norms = BatchNorm(self.in_num)
            else:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, scalers=scalers,
                                edge_dim=10, towers=1, pre_layers=5, post_layers=5,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            self.convs.append(conv)
            self.batch_norms.append(norms)

        # self.mlp1 = Sequential(Linear(4, 25), ReLU(), Linear(25, 50), ReLU(), Linear(50, 100))
        # self.mlp2 = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 1))
        # self.mlp1 = Sequential(Linear(1260, 630, bias=False), ReLU(), Linear(630, 300, bias=False), ReLU(), Linear(300, 150, bias=False), ReLU(), Linear(150, self.in_num, bias=False))
        # self.mlp1 = Sequential(Linear(1260, 600), ReLU(), Linear(600, 300), ReLU(), Linear(300, self.in_num))

        self.mlp1 = Sequential(Linear(1262, 1262), ReLU())
        # self.mlp2 = Sequential(Linear(1260, 600), ReLU(), Linear(600, 300), ReLU(), Linear(310, 150))
        self.mlp2 = Sequential(Linear(self.in_num, 600), ReLU(), Linear(600, 100), ReLU(), Linear(100, 1))
        # self.mlp2 = Sequential(Linear(self.in_num, 10), ReLU(), Linear(10, 1))
        # self.mlp3 = Sequential(Linear(1260, 630), ReLU(), Linear(630, 310), ReLU(), Linear(310, 150), ReLU(),
        #                        Linear(150, 100), ReLU(), Linear(100, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 10), ReLU(), Linear(10, 3))
        # self.mlp3 = Sequential(Linear(self.in_num, 10), ReLU(), Linear(10, 3))
        self.mlp3 = Sequential(Linear(self.in_num, 600), ReLU(), Linear(600, 100), ReLU(), Linear(100, 3))
        # self.weight_generator = Linear(1, 1)
        # self.weight_generator2 = ReLU()
        # self.mlp1 = Sequential(Linear(4, self.in_num))
        # self.mlp2 = Sequential(Linear(self.in_num, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 3))
        # self.fc = Sequential(Linear(3, 1))

    def forward(self, input_dict, mad_flag=False):
        """
                :param input_dict:
                key "x" is used for training (with sub-keys "x", "edge_index", "edge_attr", "batch", etc. involved);
                keys "atom_info" is reserved for CP2K interface.
                Note that in training we generate items "x", "edge_index", "edge_attr", "batch", etc. in advance for training efficiency. However, when being tested by CP2K interfaces, we can call a one-time function to generate them using "pos" and "number".
                :param mad_flag:
                When set to "True", the value of mad is calculated
                :return: output_dict:
                key "y" is used for training;
                keys "force" and "energy" are reserved for CP2K interface.
                """
        if self.CP2K_flag:
            assert "atom_info" in input_dict
            atom_info = input_dict["atom_info"]
            number, pos = torch.split(atom_info, [1, 3], dim=1)
            x = self.one_time_generate_forward_input(number=number, pos=pos)
            # output_dict, mad_value = self.inner_forward(x)
            output_dict = self.inner_forward(x)
        elif mad_flag:
            x = input_dict
            # output_dict, mad_value = self.inner_forward_mad(x)
            output_dict = self.inner_forward(x)
        else:
            x = input_dict
            # output_dict, mad_value = self.inner_forward(x)
            output_dict = self.inner_forward(x)
        #if self.CP2K_flag:
            #del output_dict["y"]
        # return output_dict, mad_value
        return output_dict

    def inner_forward(self, x):
        """
        The real forward call
        :param x: a dict having ["x", "edge_index", "edge_attr", "batch"] as its keys
        :return: res_dict, a dict having ["y", "force" and "energy"] as its keys
        """
        assert {"x", "edge_index", "edge_attr", "batch"}.issubset(x.keys())
        x, edge_index, edge_attr, batch = iter([x.get(one_key) for one_key in ["x", "edge_index", "edge_attr", "batch"]])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # THIS FUNCTION IS AN EXAMPLE
        x = self.mlp1(x)
        agg_weights = torch.nn.functional.softmax(self.weights, dim=0)
        # print("x_cp3: {}".format(x.shape))
        # edge_attr=edge_attr.reshape(-1,1)
        # edge_attr=self.edge_emb1(edge_attr)
        edge_attr = self.edge_emb(edge_attr)

        # w = torch.nn.functional.relu(self.std_weight)
        w = config.weight



        """
        x_cp1: torch.Size([8064, 1])
        x_cp2: torch.Size([8064, 75])
        x_cp3: torch.Size([8064, 75])
        x_before: torch.Size([8064, 75])
        x_after: torch.Size([64, 75])
        """
        # for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
        #     if i == 0:
        #         x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))
        #     else:
        #         x = F.dropout(x, p=0.2, training=self.training)
        #         x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            # x = F.dropout(x, p=0.01, training=self.training)
            x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))

        # x_array = x.cpu().detach().numpy()
        # m = np.ones((x_array.shape[0], x_array.shape[0]))
        # self.mad = Metric_for_Smoothness(x_array, m)

        # print("x_before: {}".format(x.shape))
        # print(x)
        # print("batch: {}".format(batch.shape))
        # print(batch)
        # x_energy = self.mlp2(x)
        x_energy = global_add_pool(x, batch)
        x_energy = self.mlp2(x_energy)
        x_energy = x_energy
        x_force = self.mlp3(x)
        if self.CP2K_flag:
            res_dict = dict({
                # "y": torch.ones([10, 10]),
                "force": x_force,
                "energy": x_energy,
            })
        else:
            res_dict = dict({
                # "y": torch.ones([10, 10]),
                "force": x_force,
                "energy": x_energy,
            })

        # return res_dict, None
        return res_dict, w

    def inner_forward_mad(self, x):
        """
        The real forward call
        :param x: a dict having ["x", "edge_index", "edge_attr", "batch"] as its keys
        :return: res_dict, a dict having ["y", "force" and "energy"] as its keys
        """
        assert {"x", "edge_index", "edge_attr", "batch"}.issubset(x.keys())
        x, edge_index, edge_attr, batch = iter([x.get(one_key) for one_key in ["x", "edge_index", "edge_attr", "batch"]])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # x = self.mlp1(x)
        agg_weights = torch.nn.functional.softmax(self.weights, dim=0)
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))

        x_array = x.cpu().detach().numpy()
        m = np.ones((x_array.shape[0], x_array.shape[0]))
        mad_value = Metric_for_Smoothness(x_array, m)

        x_energy = global_add_pool(x, batch)
        x_energy = self.mlp2(x_energy)
        x_energy = x_energy
        x_force = self.mlp3(x)
        if self.CP2K_flag:
            res_dict = dict({
                "force": x_force,
                "energy": x_energy,
            })
        else:
            res_dict = dict({
                "force": x_force,
                "energy": x_energy,
            })

        # return res_dict, mad_value
        return res_dict
    def one_time_generate_forward_input(self, number, pos):
        """
        In the training we generate items (like "x", "edge_index", "edge_attr", "batch", etc.) in advance for training efficiency. However, when being tested by CP2K interfaces, we can call a one-time function to generate them using "pos" and "number".
        :param number: in shape [Natom, 1]
        :param pos: in shape [Natom, 3]
        :return: x, which is functionally used in the module for the forwarding call
        """
        # THIS FUNCTION IS AN EXAMPLE
        device=torch.device("cuda")
        x_full = generate_soap(pos).to(device)
        x_full = x_full.to(torch.float32)
        # x_full = torch.cat([pos, number], 1).to(device)
        lines=pos
        DMA = np.zeros((3, 3))
        for x in range(3):
            for y in range(3):
                dis = math.sqrt((lines[x, 0] - lines[y, 0]) ** 2 + (lines[x, 1] - lines[y, 1]) ** 2 + (
                            lines[x, 2] - lines[y, 2]) ** 2)
                DMA[x, y] = dis
        DMA=torch.tensor(DMA,dtype=torch.float32)
        BTMA=torch.tensor(np.array([[0,1,1],[1,0,1],[1,1,0]]))
        adj=DMA*BTMA
        edge_index = adj.nonzero(as_tuple=False).t().contiguous().to(device)
        edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long).to(device)
        c=int(pos.shape[0])
        batch=[]
        for i in range(int(c/3)):
            batch+=[i]*3
        batch=torch.tensor(batch).to(device)
        x = dict({
            "x": x_full,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "batch": batch
        })
        return x

    def set_cp2k_flag(self, val=True):
        self.CP2K_flag = val


def train(model, args, train_loader, optimizer):
    model.train()
    # print("length: {}".format(len(train_loader)))
    total_loss = 0
    mad_flag = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch_i, data in enumerate(train_loader):  # for batch_i, data in tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / args.batch_size)):
        data = data.to(args.device)
        optimizer.zero_grad()
        # print("data.x:", data.x.shape)
        # print("data.edge_index:", data.edge_index.shape)
        # print("data.edge_attr:", data.edge_attr.shape)
        # print("data.batch:", data.batch.shape)
        # loss=0
        # x=out.shape[0]
        # for i in range(x):
        # loss_vector=math.sqrt((out[i,0].squeeze()-data.y[i,0])**2+(out[i,1].squeeze()-data.y[i,1])**2+(out[i,2].squeeze()-data.y[i,2])**2)
        # loss+=loss_vector
        # loss=torch.tensor(loss/x,requires_grad=True).to(args.device)
        input_dict=dict({
            "x":data.x,
            "edge_index":data.edge_index,
            "edge_attr":data.edge_attr,
            "batch":data.batch
        })
        # out, _ = model(input_dict, mad_flag)
        out, weight = model(input_dict, mad_flag)

        """
        计算batch中原子的个数
        """
        batch = input_dict["batch"].cpu().numpy()
        unique, counts = np.unique(batch, return_counts=True)
        node_counts = torch.tensor(counts, dtype=torch.float32)
        node_counts = node_counts.to(device)


        # loss=(((out["force"].squeeze()-data.z)**2).mean(1).sqrt()).mean()
        loss1 = ((out["energy"].squeeze() - data.y)).abs()
        loss0 = ((out["energy"].squeeze() - data.y) / node_counts).abs()
        losses_per_molecule_type = {i: [] for i in range(1, 22)}
        # 初始化一个字典来保存每种分子类型的损失值
        for loss, molecule_type in zip(loss0, node_counts):
            # 将损失值添加到对应分子类型的列表中
            losses_per_molecule_type[molecule_type.item() / 3].append(loss)
        # 计算每种分子类型的 MAE
        mae_per_molecule_type = {molecule_type: torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)
                                 for molecule_type, losses in losses_per_molecule_type.items()}
        # number_per_molecule_type = list(mae_per_molecule_type.keys())
        mae_per_molecule_type = list(mae_per_molecule_type.values())
        mae_per_molecule_type = [x.item() for x in mae_per_molecule_type]
        # mae_per_molecule_type_atom = [x/(y*3) for x, y in zip(mae_per_molecule_type, number_per_molecule_type)]

        maes = np.array(mae_per_molecule_type)
        # std_value = np.std(maes)
        std_value = custom_std(maes)

        # loss = loss0.mean() + config.weight * std_value
        loss = loss0.mean() + weight * std_value
        # loss = (((out["force"].squeeze()-data.z)**2).mean(1).sqrt()).mean()+(out["energy"].squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()


    loss = total_loss / len(train_loader.dataset)
    return model, loss, weight

def train_last(model, args, train_loader, optimizer, weight):
    model.train()
    total_loss = 0
    mad_list = []
    mad_flag = False
    for batch_i, data in enumerate(
            train_loader):  # for batch_i, data in tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / args.batch_size)):
        data = data.to(args.device)
        optimizer.zero_grad()

        input_dict = dict({
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "batch": data.batch
        })
        # out, mad_value = model(input_dict, mad_flag)
        # mad_list.append(mad_value)
        out = model(input_dict, weight, mad_flag)

        # loss=(((out["force"].squeeze()-data.z)**2).mean(1).sqrt()).mean()
        loss = (out["energy"].squeeze() - data.y).abs().mean()
        # loss = (((out["force"].squeeze()-data.z)**2).mean(1).sqrt()).mean()+(out["energy"].squeeze() - data.y).abs().mean()
        loss.backward(retain_graph=True)
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    mad = sum(mad_list) / len(mad_list)
    loss = total_loss / len(train_loader.dataset)
    return model, loss, mad

@torch.no_grad()
def test(model, args, loader):
    model.eval()
    total_error = 0
    mad_flag = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in loader:
        data = data.to(args.device)
        input_dict = dict({
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "batch": data.batch
        })
        # out, _ = model(input_dict, mad_flag)
        out, weight = model(input_dict, mad_flag)

        batch = input_dict["batch"].cpu().numpy()
        unique, counts = np.unique(batch, return_counts=True)
        node_counts = torch.tensor(counts, dtype=torch.float32)
        node_counts = node_counts.to(device)

        # loss=(((out["force"].squeeze()-data.z)**2).mean(1).sqrt()).mean()
        loss0 = ((out["energy"].squeeze() - data.y) / node_counts).abs()
        losses_per_molecule_type = {i: [] for i in range(1, 22)}
        # 初始化一个字典来保存每种分子类型的损失值
        for loss, molecule_type in zip(loss0, node_counts):
            # 将损失值添加到对应分子类型的列表中
            losses_per_molecule_type[molecule_type.item() / 3].append(loss)
        # 计算每种分子类型的 MAE
        mae_per_molecule_type = {molecule_type: torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)
                                 for molecule_type, losses in losses_per_molecule_type.items()}
        # number_per_molecule_type = list(mae_per_molecule_type.keys())
        mae_per_molecule_type = list(mae_per_molecule_type.values())
        mae_per_molecule_type = [x.item() for x in mae_per_molecule_type]
        # mae_per_molecule_type_atom = [x/(y*3) for x, y in zip(mae_per_molecule_type, number_per_molecule_type)]

        maes = np.array(mae_per_molecule_type)
        # std_value = np.std(maes)
        std_value = custom_std(maes)

        # total_error += (loss0.mean() + config.weight * std_value).item() * data.num_graphs
        total_error += (loss0.mean() + weight * std_value).item() * data.num_graphs
        # total_error += (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean() + (
        #             out["energy"].squeeze() - data.y).abs().mean()
        # total_error += (out["energy"].squeeze() - data.y).abs().mean().item() * data.num_graphs
        # total_error = (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean()
    loss = total_error / len(loader.dataset)
    return loss

@torch.no_grad()
def val(model, args, loader):
    model.eval()
    total_error = 0
    mad_flag = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in loader:
        data = data.to(args.device)
        input_dict = dict({
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "batch": data.batch
        })
        # out, _ = model(input_dict, mad_flag)
        out, weight = model(input_dict, mad_flag)

        batch = input_dict["batch"].cpu().numpy()
        unique, counts = np.unique(batch, return_counts=True)
        node_counts = torch.tensor(counts, dtype=torch.float32)
        node_counts = node_counts.to(device)

        # loss=(((out["force"].squeeze()-data.z)**2).mean(1).sqrt()).mean()
        loss0 = ((out["energy"].squeeze() - data.y) / node_counts).abs()
        losses_per_molecule_type = {i: [] for i in range(1, 22)}
        # 初始化一个字典来保存每种分子类型的损失值
        for loss, molecule_type in zip(loss0, node_counts):
            # 将损失值添加到对应分子类型的列表中
            losses_per_molecule_type[molecule_type.item() / 3].append(loss)
        # 计算每种分子类型的 MAE
        mae_per_molecule_type = {molecule_type: torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)
                                 for molecule_type, losses in losses_per_molecule_type.items()}
        # number_per_molecule_type = list(mae_per_molecule_type.keys())
        mae_per_molecule_type = list(mae_per_molecule_type.values())
        mae_per_molecule_type = [x.item() for x in mae_per_molecule_type]
        # mae_per_molecule_type_atom = [x/(y*3) for x, y in zip(mae_per_molecule_type, number_per_molecule_type)]

        maes = np.array(mae_per_molecule_type)
        std_value = np.std(maes)

        # total_error += (loss0.mean() + config.weight * std_value).item() * data.num_graphs
        total_error += (loss0.mean()).item() * data.num_graphs
    loss = total_error / len(loader.dataset)
    return loss

# if __name__ == "__main__":
    # # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GCN_N3P')
    # train_dataset = MyDataset(path, subset=False, split='train')
    # val_dataset = MyDataset(path, subset=False, split='val')
    # test_dataset = MyDataset(path, subset=False, split='test')
    #
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    # # Compute the maximum in-degree in the training data.
    # max_degree = -1
    # for data in train_dataset:
    #     # print("data.num_nodes:", data.num_nodes)
    #     # print("data.edge_index[1]:", data.edge_index[1].shape)
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     max_degree = max(max_degree, int(d.max()))
    #
    # # Compute the in-degree histogram tensor
    # deg = torch.zeros(max_degree + 1, dtype=torch.long)
    # for data in train_dataset:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MyNetwork(deg).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)
    #
    # t0 = time.time()
    # for epoch in range(100):
    #     loss = train(epoch)
    #     val_mae = test(val_loader)
    #     test_mae = test(test_loader)
    #     scheduler.step(val_mae)
    #     t = time.time()
    #     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, Test: {test_mae:.4f}, Time: {t - t0:.2f}s')
    #     t0 = t
