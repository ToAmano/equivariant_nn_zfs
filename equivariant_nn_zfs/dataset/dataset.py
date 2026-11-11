import numpy as np
import torch
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from mace import data, tools
from equivariant_nn_zfs.tools.convert_matrix import cartesian_to_spherical_irreps


def local_weights(struct, r_in, r_out, cn_r1=2.2, cn_r2=2.6, cn_mid=3.25, cn_width=0.03):
    """pbc_zfs_local: per-atom weights of the readout, w = 1 within r_in of the defect centre, a cosine switch to 0 at
    r_out. Everything is continuous in the atomic positions (no jump when atoms move, unlike a hard centre selection):
      smooth coordination CN_i = sum_j s(r_ij), s = 1 below cn_r1, cosine switch to 0 at cn_r2 (Si-C bond 1.89 A);
      defect weight g_i = 1 / (1 + exp((CN_i - cn_mid) / cn_width)) (~1 for the 6 dangling-bond atoms of V_Si V_C,
      CN <= 3.02, ~0 for the rest, CN >= 3.48 in all 532 SiC structures of d195c4 + test30, 300-1200 K);
      centre = sum_i g_i x_i / sum_i g_i (minimum image, atoms within 8 A of the lowest-CN atom).
    Rotating / translating the structure moves the centre with it, so the model stays equivariant."""
    i, d = neighbor_list("id", struct, cn_r2)
    s = 0.5 * (1.0 + np.cos(np.pi * np.clip((d - cn_r1) / (cn_r2 - cn_r1), 0.0, 1.0)))
    cn = np.bincount(i, weights=s, minlength=len(struct))
    g = 1.0 / (1.0 + np.exp(np.clip((cn - cn_mid) / cn_width, -50.0, 50.0)))
    ref = struct.positions[np.argmin(cn)]
    dv, dl = find_mic(struct.positions - ref, struct.cell, struct.pbc)
    near = dl < 8.0  # all 6 dangling-bond atoms (up to ~5 A apart); g of the other atoms ~1e-11
    centre = ref + (g[near, None] * dv[near]).sum(0) / g[near].sum()
    _, r = find_mic(struct.positions - centre, struct.cell, struct.pbc)
    x = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def local_weight_graph(struct, cn_r2=2.6, near_r=8.0):
    """pbc_zfs_local_dw: the discrete choices of local_weights at this structure, so that the model can recompute the
    weights in torch from the positions (differentiable; model.local_weights_torch). Same discrete choices as
    local_weights: CN neighbour pairs within cn_r2 (with their periodic shift vectors), the lowest-CN reference atom and the
    atoms within near_r of it (with their minimum-image shift vectors). Valid for small displacements (finite differences,
    autograd at this structure); for a new structure the dataset builds them again."""
    i, j, S = neighbor_list("ijS", struct, cn_r2)
    _, d = neighbor_list("id", struct, cn_r2)
    s = 0.5 * (1.0 + np.cos(np.pi * np.clip((d - 2.2) / (cn_r2 - 2.2), 0.0, 1.0)))
    cn = np.bincount(i, weights=s, minlength=len(struct))
    ref = int(np.argmin(cn))
    dv, dl = find_mic(struct.positions - struct.positions[ref], struct.cell, struct.pbc)
    near = np.where(dl < near_r)[0]
    dt = torch.get_default_dtype()
    return dict(lw_i=torch.tensor(i), lw_j=torch.tensor(j), lw_shift=torch.tensor(S @ struct.cell.array, dtype=dt),
                lw_ref=torch.tensor([ref]), lw_near=torch.tensor(near),
                lw_near_shift=torch.tensor(dv[near] - (struct.positions[near] - struct.positions[ref]), dtype=dt),
                lw_cell=torch.tensor(struct.cell.array, dtype=dt))


class MinimalDataset(Dataset):

    def __init__(self,
                 structures,
                 irreps_out,
                 radial_cutoff,
                 device,
                 target_key='target_L2',
                 target_scale=1.0,
                 local_rin=None,
                 local_rout=None
                 ):
        self.radial_cutoff = radial_cutoff
        self.structures = structures
        self.irreps_out = irreps_out
        # pbc_zfs_local: readout weighted by local_weights(r_in, r_out) when local_rout is set (None = original model)
        self.local_rin, self.local_rout = local_rin, local_rout
        # target_scale: the model learns target / target_scale (e.g. 1000 for a D tensor in MHz -> GHz)
        self.target_scale = target_scale
        self.targets = torch.stack([cartesian_to_spherical_irreps(torch.tensor(s.info[target_key].reshape(3, 3)) / target_scale,
                                                                  irreps=irreps_out) for s in structures], dim=0)
        self.device = device

        z_table = set()
        for s in structures:
            s_z_table = s.get_atomic_numbers()
            z_table.update(s_z_table)
        self.z_table = tools.AtomicNumberTable(sorted(z_table))  # sorted: same one-hot order for every dataset

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        struct = self.structures[idx]

        config = data.Configuration(
            atomic_numbers=struct.numbers,
            positions=struct.positions,
            properties={'positions': 'positions'},
            property_weights={'positions': 1},
            cell=struct.cell.array if struct.pbc.any() else None,  # periodic supercells: neighbours across the cell
            pbc=tuple(bool(x) for x in struct.pbc)
        )

        # we handle configurations using the AtomicData class
        batch = data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.radial_cutoff)

        if self.local_rout is not None:
            batch.local_weight = torch.tensor(local_weights(struct, self.local_rin, self.local_rout), dtype=torch.get_default_dtype())
            for k, v in local_weight_graph(struct).items():  # pbc_zfs_local_dw: for the differentiable weights in the model
                setattr(batch, k, v)

        target = self.targets[idx]

        return batch, target

class EvaluationDataset(Dataset):

    def __init__(self,
                 structures,
                 radial_cutoff,
                 device
                 ):
        self.radial_cutoff = radial_cutoff
        self.structures = structures
        self.device = device

        z_table = set()
        for s in structures:
            s_z_table = s.get_atomic_numbers()
            z_table.update(s_z_table)
        self.z_table = tools.AtomicNumberTable(sorted(z_table))  # sorted: same one-hot order for every dataset

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        struct = self.structures[idx]

        config = data.Configuration(
            atomic_numbers=struct.numbers,
            positions=struct.positions,
            properties={'positions': 'positions'},
            property_weights={'positions': 1},
            cell=struct.cell.array if struct.pbc.any() else None,  # periodic supercells: neighbours across the cell
            pbc=tuple(bool(x) for x in struct.pbc)
        )

        # we handle configurations using the AtomicData class
        batch = data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.radial_cutoff)

        return batch

def collate_fn_eval(batch_):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list

    return {'batches': list(batch_)}
