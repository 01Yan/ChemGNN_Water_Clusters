import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.nn import BatchNorm, global_add_pool
from typing import Dict, Callable, List, Optional, Set, get_type_hints

import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential
from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, Size
from utils import *


from torch_geometric.nn.inits import reset


class CEALConv(MessagePassing):
    r"""Chemical Environment Adaptive Learning
    from the `"Chemical Environment Adaptive Learning for Optical Band Gap Prediction of Doped Graphitic Carbon Nitride Nanosheets"



    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    """

    def __init__(self, in_channels: int, out_channels: int, weights: Tensor,
                 aggregators: List[str],edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.weights = weights

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers


        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, weights: Tensor,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=weights, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor, weights: Tensor,
                  dim_size: Optional[int] = None,
                  ) -> Tensor:

        outs = []

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = weights[0] * scatter(inputs, index, 0, None, dim_size, reduce='sum')
                # out = weights[0] * scatter_sum(inputs, index, 0, None, dim_size=dim_size)
            elif aggregator == 'mean':
                out = weights[1] * scatter(inputs, index, 0, None, dim_size, reduce='mean')
                # out = weights[1] * scatter_mean(inputs, index, 0, None, dim_size=dim_size)
            elif aggregator == 'min':
                out = weights[2] * scatter(inputs, index, 0, None, dim_size, reduce='min')
                # out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = weights[3] * scatter(inputs, index, 0, None, dim_size, reduce='max')
                # out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                # mean = scatter_mean(inputs, index, 0, None, dim_size=dim_size)
                # mean_squares = scatter_mean(inputs * inputs, index, 0, None, dim_size=dim_size)
                # out_4=(mean_squares - mean * mean)

                out = (mean_squares - mean * mean)
                if aggregator == 'std':
                    out = weights[4] * torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        outs = []
        outs.append(out)
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')


class ChemGNNModel(torch.nn.Module):
    def __init__(self, _deg):
        super().__init__()
        self.model_name = "ChemGNN"
        self.edge_emb = Embedding(20, 10)
        self.conv_num = 10
        self.in_num = 100

        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights=torch.nn.Parameter(torch.rand(len(aggregators)))

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(self.conv_num):
            if _ == 0:
                conv = CEALConv(in_channels=11, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1,
                                post_layers=1, divide_input=False)
            else:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1,
                                post_layers=1, divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.in_num))
        self.mlp2 = Sequential(Linear(self.in_num, 50), ReLU(), Linear(50, 20), ReLU(), Linear(20, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        agg_weights = torch.nn.functional.softmax(self.weights,dim=0)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index,agg_weights,edge_attr)))
        x = self.mlp2(x)

        res = global_add_pool(x, batch)
        return res

# from torch.nn import Embedding, Linear
# from typing import Dict, Callable, List, Optional, Set, get_type_hints
# import torch
# from torch import Tensor
# from torch.nn import ModuleList, ReLU, Sequential
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.typing import Adj, OptTensor, Size
# from torch_geometric.nn.inits import reset
# from torch_geometric.nn import BatchNorm, global_add_pool
# import torch.nn.functional as F
#
#
# class CEALConv(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, weights: Tensor,
#                  aggregators: List[str], edge_dim: Optional[int] = None, towers: int = 1,
#                  pre_layers: int = 1, post_layers: int = 1,
#                  divide_input: bool = False, **kwargs):
#
#         kwargs.setdefault('aggr', None)
#         super().__init__(node_dim=0, **kwargs)
#
#         if divide_input:
#             assert in_channels % towers == 0
#         assert out_channels % towers == 0
#
#         device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.aggregators = aggregators
#         self.edge_dim = edge_dim
#         self.towers = towers
#         self.divide_input = divide_input
#         self.weights = weights
#
#         self.F_in = in_channels // towers if divide_input else in_channels
#         self.F_out = self.out_channels // towers
#
#         if self.edge_dim is not None:
#             self.edge_encoder = Linear(edge_dim, self.F_in)
#
#         self.pre_nns = ModuleList()
#         self.post_nns = ModuleList()
#         for _ in range(towers):
#             modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
#             for _ in range(pre_layers - 1):
#                 modules += [ReLU()]
#                 modules += [Linear(self.F_in, self.F_in)]
#             self.pre_nns.append(Sequential(*modules))
#
#             in_channels = (len(aggregators) + 1) * self.F_in
#             modules = [Linear(in_channels, self.F_out)]
#             for _ in range(post_layers - 1):
#                 modules += [ReLU()]
#                 modules += [Linear(self.F_out, self.F_out)]
#             self.post_nns.append(Sequential(*modules))
#
#         self.lin = Linear(out_channels, out_channels)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.edge_dim is not None:
#             self.edge_encoder.reset_parameters()
#         for nn in self.pre_nns:
#             reset(nn)
#         for nn in self.post_nns:
#             reset(nn)
#         self.lin.reset_parameters()
#
#     def forward(self, x: Tensor, edge_index: Adj, weights: Tensor,
#                 edge_attr: OptTensor = None
#                 ) -> Tensor:
#         """"""
#         device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
#         self.pre_nns = self.pre_nns.to(device)
#         self.post_nns = self.post_nns.to(device)
#         self.edge_encoder = self.edge_encoder.to(device)
#         self.lin = self.lin.to(device)
#
#         if self.divide_input:
#             x = x.view(-1, self.towers, self.F_in)
#         else:
#             x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
#
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=weights, size=None)
#
#         out = torch.cat([x, out], dim=-1)
#         outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
#         out = torch.cat(outs, dim=1)
#
#         return self.lin(out)
#
#     def message(self, x_i: Tensor, x_j: Tensor,
#                 edge_attr: OptTensor) -> Tensor:
#
#         h: Tensor = x_i  # Dummy.
#         if edge_attr is not None:
#             edge_attr = self.edge_encoder(edge_attr)
#             edge_attr = edge_attr.view(-1, 1, self.F_in)
#             edge_attr = edge_attr.repeat(1, self.towers, 1)
#             h = torch.cat([x_i, x_j, edge_attr], dim=-1)
#         else:
#             h = torch.cat([x_i, x_j], dim=-1)
#
#         hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
#         return torch.stack(hs, dim=1)
#
#     def aggregate(self, inputs: Tensor, index: Tensor, weights: Tensor,
#                   dim_size: Optional[int] = None,
#                   ) -> Tensor:
#
#         device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
#         outs = []
#
#         # unique values in the index tensor
#         unique_index = list(index.unique().size())[0]
#         # the second and third dimension sizes of the inputs tensor
#         ydim_inputs = inputs.size(dim=1)
#         zdim_inputs = inputs.size(dim=2)
#         # expand the one-dimensional index tensor to the same size as the three-dimensional inputs tensor
#         # note that pytorch_scatter_reduce requires the same dimension for the index and inputs tesors
#         tensor_index = index.unsqueeze(-1).unsqueeze(-1).expand(inputs.size())
#
#         for aggregator in self.aggregators:
#             if aggregator == 'sum':
#                 out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
#                 out = weights[0] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="sum",
#                                                       include_self=False)
#                 # out = weights[0] * scatter(inputs, index, 0, None, dim_size, reduce='sum')
#                 # out = weights[0] * scatter_sum(inputs, index, 0, None, dim_size=dim_size)
#             elif aggregator == 'mean':
#                 out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
#                 out = weights[1] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="mean",
#                                                       include_self=False)
#                 # out = weights[1] * scatter(inputs, index, 0, None, dim_size, reduce='mean')
#                 # out = weights[1] * scatter_mean(inputs, index, 0, None, dim_size=dim_size)
#             elif aggregator == 'min':
#                 out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
#                 out = weights[2] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="amin",
#                                                       include_self=False)
#                 # out = weights[2] * scatter(inputs, index, 0, None, dim_size, reduce='min')
#                 # out = scatter(inputs, index, 0, None, dim_size, reduce='min')
#             elif aggregator == 'max':
#                 out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
#                 out = weights[3] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="amax",
#                                                       include_self=False)
#                 # out = weights[3] * scatter(inputs, index, 0, None, dim_size, reduce='max')
#                 # out = scatter(inputs, index, 0, None, dim_size, reduce='max')
#             elif aggregator == 'var' or aggregator == 'std':
#                 mean = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
#                 mean = mean.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="mean", include_self=False)
#                 mean_squares = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
#                 mean_squares = mean_squares.scatter_reduce(dim=0, index=tensor_index, src=inputs * inputs,
#                                                            reduce="mean", include_self=False)
#                 # mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
#                 # mean_squares = scatter(inputs * inputs, index, 0, None,
#                 #                       dim_size, reduce='mean')
#                 out = (mean_squares - mean * mean)
#                 if aggregator == 'std':
#                     out = weights[4] * torch.sqrt(torch.relu(out) + 1e-5)
#             else:
#                 raise ValueError(f'Unknown aggregator "{aggregator}".')
#             outs.append(out)
#         out = torch.cat(outs, dim=-1)
#
#         outs = []
#         outs.append(out)
#         return torch.cat(outs, dim=-1)
#
#     def __repr__(self):
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, towers={self.towers}, '
#                 f'edge_dim={self.edge_dim})')
#
# class ChemGNNModel(torch.nn.Module):
#     def __init__(self, _deg):
#         super().__init__()
#         self.model_name = "ChemGNN"
#         self.edge_emb = Embedding(20, 10)
#         self.conv_num = 10
#         self.in_num = 100
#
#         aggregators = ['sum', 'mean', 'min', 'max', 'std']
#         self.weights=torch.nn.Parameter(torch.rand(len(aggregators)))
#
#         self.convs = ModuleList()
#         self.batch_norms = ModuleList()
#
#         for _ in range(self.conv_num):
#             if _ == 0:
#                 conv = CEALConv(in_channels=11, out_channels=self.in_num, weights=self.weights,
#                                 aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1,
#                                 post_layers=1, divide_input=False)
#             else:
#                 conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
#                                 aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1,
#                                 post_layers=1, divide_input=False)
#             self.convs.append(conv)
#             self.batch_norms.append(BatchNorm(self.in_num))
#         self.mlp2 = Sequential(Linear(self.in_num, 50), ReLU(), Linear(50, 20), ReLU(), Linear(20, 1))
#
#     def forward(self, x, edge_index, edge_attr, batch):
#         agg_weights = torch.nn.functional.softmax(self.weights,dim=0)
#         edge_attr = self.edge_emb(edge_attr)
#         for conv, batch_norm in zip(self.convs, self.batch_norms):
#             x = F.relu(batch_norm(conv(x, edge_index,agg_weights,edge_attr)))
#         x = self.mlp2(x)
#
#         res = global_add_pool(x, batch)
#         return res
