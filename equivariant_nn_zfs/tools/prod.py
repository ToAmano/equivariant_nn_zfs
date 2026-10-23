import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from equivariant_nn_zfs.tools.contract import ContractProduct3j


@compile_mode("script")
class Product3body(nn.Module):
    def __init__(self,
                 irreps_sh,
                 hidden_irreps,
                 node_attr_irreps,
                 ncor
                 ):
        super().__init__()

        self.contraction = ContractProduct3j(irreps_in1=irreps_sh,
                                             irreps_in2=irreps_sh,
                                             irreps_target=irreps_sh
                                             )
        # The input of linear_body is [x, x (x) x, (x (x) x) (x) x] concatenated along the irreps axis, i.e. the layout
        # irreps_sh + irreps_sh + irreps_sh (not sorted). The original used the sorted/simplified irreps here, which
        # mixes components of different l and breaks equivariance (pbc_zfs fix).
        ncor_irreps_sh = o3.Irreps(sum((list(irreps_sh) for _ in ncor), []))

        self.linear_body = Linear(irreps_in=ncor_irreps_sh,
                                  irreps_out=irreps_sh
                                  )
        # [nodes, channels, irreps_sh.dim] -> e3nn layout of hidden_irreps (channels x l blocks, channel-major inside a block)
        assert hidden_irreps == (irreps_sh * hidden_irreps.count(irreps_sh[0].ir)).sort()[0].simplify()
        self.sh_slices = [(sl.start, sl.stop) for sl in irreps_sh.slices()]

        self.linear = Linear(irreps_in=hidden_irreps,
                             irreps_out=hidden_irreps
                             )

        self.fctp_attributes = FullyConnectedTensorProduct(irreps_in1=node_attr_irreps,
                                                           irreps_in2=hidden_irreps,
                                                           irreps_out=hidden_irreps
                                                           )

        print(self.linear_body, self.linear, self.fctp_attributes)

    def forward(self,
                node_feature_i,
                node_attr,
                sc
                ):

        node2 = self.contraction(node_feature_i, node_feature_i)

        node3 = self.contraction(node2, node_feature_i)  # nodes, channels, irreps

        node_out = torch.cat([node_feature_i, node2, node3],
                             dim=-1
                             )  # nodes, channels, irreps*nbody

        node_out = self.linear_body(node_out)  # nodes, channels, irreps

        # flatten to the e3nn layout of hidden_irreps (the original concatenated component-major, which is not it)
        nodes = node_out.shape[0]
        node_out = torch.cat([node_out[..., a:b].reshape(nodes, -1) for a, b in self.sh_slices], dim=-1)

        node_out = self.linear(node_out)

        node_out = self.fctp_attributes(node_attr, node_out)

        return node_out + sc


@compile_mode("script")
class ReadoutL2(nn.Module):
    def __init__(self,
                 hidden_irreps,
                 out_irreps
                 ):
        super().__init__()

        self.linear_readout = Linear(irreps_in=hidden_irreps,
                                     irreps_out=out_irreps
                                     )

        print(self.linear_readout)

    def forward(self,
                node_features
                ):

        readout = self.linear_readout(node_features)

        return readout
