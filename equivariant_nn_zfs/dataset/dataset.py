import torch
from torch.utils.data import Dataset
from mace import data, tools
from equivariant_nn_zfs.tools.convert_matrix import cartesian_to_spherical_irreps


class MinimalDataset(Dataset):

    def __init__(self,
                 structures,
                 irreps_out,
                 radial_cutoff,
                 device,
                 target_key='target_L2',
                 target_scale=1.0
                 ):
        self.radial_cutoff = radial_cutoff
        self.structures = structures
        self.irreps_out = irreps_out
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
