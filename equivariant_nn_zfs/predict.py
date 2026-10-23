"""
Predict the 3x3 (traceless) tensor with a trained TensorRegressor for every structure of an extxyz file.

Usage: python -m equivariant_nn_zfs.predict --model checkpoint_model.pth --data frames.xyz --rcut 6.0 \
           [--target_scale 1000] [--target_key target_L2] [--out pred.npz] [--use_cuda]
Output (npz): tensor (n, 3, 3) in the units of the training targets (target_scale undone), the scalar D = 3/2 D_zz and
E = (D_xx - D_yy)/2 in the principal frame (|D_zz| >= |D_xx| >= |D_yy|), the principal axes, and the reference tensor
(when --target_key is present in the file).
"""
import argparse

import numpy as np
import torch
from ase.io import read
from torch.utils.data import DataLoader

from equivariant_nn_zfs.dataset.dataset import EvaluationDataset, collate_fn_eval
from equivariant_nn_zfs.tools.convert_matrix import spherical_irreps_to_cartesian


def d_e(t):
    w, v = np.linalg.eigh(0.5 * (t + t.T))
    o = np.argsort(np.abs(w))  # |Dy| <= |Dx| <= |Dz|
    dy, dx, dz = w[o]
    return 1.5 * dz, 0.5 * (dx - dy), v[:, o[::-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='checkpoint_model.pth'); ap.add_argument('--data', required=True)
    ap.add_argument('--rcut', type=float, required=True); ap.add_argument('--target_scale', type=float, default=1.0)
    ap.add_argument('--target_key', default='target_L2'); ap.add_argument('--out', default='pred.npz')
    ap.add_argument('--use_cuda', action='store_true')
    a = ap.parse_args()
    device = torch.device('cuda' if a.use_cuda and torch.cuda.is_available() else 'cpu')
    model = torch.load(a.model, weights_only=False, map_location=device); model.device = device; model.eval()
    db = read(a.data, ':')
    ds = EvaluationDataset(db, radial_cutoff=a.rcut, device=device)
    ds.z_table = model_ztable(model, ds)
    out = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=1, collate_fn=collate_fn_eval):
            y = model([g.to(device) for g in b['batches']])  # (1, 5) l=2 components (e3nn basis)
            out.append(spherical_irreps_to_cartesian(y[0]).cpu().numpy() * a.target_scale)
    T = np.array(out)
    res = {'tensor': T}
    de = [d_e(t) for t in T]
    res['D'] = np.array([x[0] for x in de]); res['E'] = np.array([x[1] for x in de]); res['axes'] = np.array([x[2] for x in de])
    if all(a.target_key in s.info for s in db):
        R = np.array([s.info[a.target_key].reshape(3, 3) for s in db])
        R = R - np.trace(R, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)  # traceless part (what the model learns)
        res['reference'] = R
        print(f"tensor RMSE (traceless, per component): {np.sqrt(((T - R) ** 2).mean()):.4g}")
    np.savez(a.out, **res)
    print(f"{len(T)} structures -> {a.out}; D mean {res['D'].mean():.4g}, E mean {res['E'].mean():.4g}")


def model_ztable(model, ds):
    """The node attributes must use the training z_table; EvaluationDataset builds one from the new data only."""
    zt = getattr(model, 'z_table', None)
    return zt if zt is not None else ds.z_table


if __name__ == '__main__':
    main()
