from utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from config import config
from torch_geometric.nn import BatchNorm, global_add_pool
from ChemGNN import CEALConv


class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_emb = Embedding(20, 10)  # self.edge_emb = Embedding(4, 50)
        self.conv_num = 1
        self.in_num = 1262
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.std_weight = nn.Parameter(torch.rand(1))

        for _ in range(self.conv_num):
            if _ == 0:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=5, post_layers=5,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            else:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=5, post_layers=5,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            self.convs.append(conv)
            self.batch_norms.append(norms)

        self.mlp1 = Sequential(Linear(1262, 1262), ReLU())
        self.mlp2 = Sequential(Linear(self.in_num, 600), ReLU(), Linear(600, 100), ReLU(), Linear(100, 1))
        self.mlp3 = Sequential(Linear(self.in_num, 600), ReLU(), Linear(600, 100), ReLU(), Linear(100, 3))

    def forward(self, input_dict):
        x = input_dict
        assert {"x", "edge_index", "edge_attr", "batch"}.issubset(x.keys())
        x, edge_index, edge_attr, batch = iter([x.get(one_key) for one_key in ["x", "edge_index", "edge_attr", "batch"]])
        # x = x.to(torch.float64) # float 64 mode
        x = self.mlp1(x)
        agg_weights = torch.nn.functional.softmax(self.weights, dim=0)
        edge_attr = self.edge_emb(edge_attr)
        w = config.weight
        '''
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            if i == 0:
                x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))
            else:
                x = F.dropout(x, p=0.2, training=self.training)
                x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))
        '''
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))

        x_energy = global_add_pool(x, batch)
        x_energy = self.mlp2(x_energy)
        x_energy = x_energy
        x_force = self.mlp3(x)
        res_dict = dict({
            "force": x_force,
            "energy": x_energy,
        })

        return res_dict, w


def train(model, args, train_loader, optimizer):
    model.train()
    total_loss = 0
    gradients_list = []
    for batch_i, data in enumerate(train_loader):
        data = data.to(args.device)
        optimizer.zero_grad()
        input_dict=dict({
            "x":data.x,
            "edge_index":data.edge_index,
            "edge_attr":data.edge_attr,
            "batch":data.batch
        })
        out, weight = model(input_dict)

        batch = input_dict["batch"]
        node_counts = torch.bincount(batch)

        loss0 = ((out["energy"].squeeze() - data.y) / node_counts).abs()
        """
        losses_per_molecule_type = {i: [] for i in range(1, 22)}
        for loss, molecule_type in zip(loss0, node_counts):
            losses_per_molecule_type[molecule_type.item() // 3].append(loss)
        mae_per_molecule_type = []
        for molecule_type, losses in losses_per_molecule_type.items():
            if losses:
                mae_per_molecule_type.append(torch.stack(losses).mean())
            else:
                mae_per_molecule_type.append(torch.tensor(0.0, device=args.device))
        mae_per_molecule_type = torch.stack(mae_per_molecule_type)
        std_value = custom_std(mae_per_molecule_type)
        """
        # loss = loss0.mean() + weight * std_value
        loss = loss0.mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs

        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name == 'mlp2.4.weight':
                gradients = torch.norm(parameter.grad, p=2)
                gradients = gradients.item()
                gradients_list.append(gradients)
                break
            # print(name)
        optimizer.step()
    loss = total_loss / len(train_loader.dataset)
    return model, loss, weight, sum(gradients_list)/len(gradients_list)

@torch.no_grad()
def test(model, args, loader):
    model.eval()
    total_error = 0
    for data in loader:
        data = data.to(args.device)
        input_dict = dict({
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "batch": data.batch
        })
        out, weight = model(input_dict)

        batch = input_dict["batch"]
        node_counts = torch.bincount(batch)

        loss0 = ((out["energy"].squeeze() - data.y) / node_counts).abs()
        """
        losses_per_molecule_type = {i: [] for i in range(1, 22)}
        for loss, molecule_type in zip(loss0, node_counts):
            losses_per_molecule_type[molecule_type.item() // 3].append(loss)
        mae_per_molecule_type = []
        for molecule_type, losses in losses_per_molecule_type.items():
            if losses:
                mae_per_molecule_type.append(torch.stack(losses).mean())
            else:
                mae_per_molecule_type.append(torch.tensor(0.0, device=args.device))
        mae_per_molecule_type = torch.stack(mae_per_molecule_type)
        std_value = custom_std(mae_per_molecule_type)
        """
        # total_error += (loss0.mean() + weight * std_value).item() * data.num_graphs
        total_error += (loss0.mean()).item() * data.num_graphs
    loss = total_error / len(loader.dataset)
    return loss

@torch.no_grad()
def val(model, args, loader):
    model.eval()
    total_error = 0
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
        out, weight = model(input_dict)

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


