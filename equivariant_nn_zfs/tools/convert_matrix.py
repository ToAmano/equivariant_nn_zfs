import torch
from e3nn.io import CartesianTensor
from e3nn.o3 import Irreps

# pbc_zfs: Cartesian <-> spherical conversion in e3nn's own basis. The original hand-written matrices use the standard
# real-spherical-harmonic order (xy, yz, 3z^2-r^2, xz, x^2-y^2), which is not e3nn's '2e' basis, so targets and model
# outputs transformed differently under rotation.
_SYM = CartesianTensor("ij=ji")  # irreps 1x0e + 1x2e


def cartesian_to_spherical_irreps(cartmat: torch.Tensor,
                                  irreps: Irreps,
                                  ) -> torch.Tensor:
    """3x3 symmetric matrix -> components in e3nn's basis for irreps '2e' (traceless part) or '0e+2e'."""
    assert cartmat.shape == (3, 3)
    sph = _SYM.from_cartesian(0.5 * (cartmat + cartmat.T))
    if str(irreps) == '1x2e':
        return sph[1:]
    if str(irreps) == '1x0e+1x2e':
        return sph
    raise ValueError(f"irreps {irreps} not supported (only symmetric tensors: 2e or 0e+2e)")


def spherical_irreps_to_cartesian(sphmat: torch.Tensor) -> torch.Tensor:
    """Components (..., 5) for '2e' or (..., 6) for '0e+2e' in e3nn's basis -> (..., 3, 3) symmetric matrix."""
    if sphmat.shape[-1] == 5:
        sphmat = torch.cat([torch.zeros(*sphmat.shape[:-1], 1, dtype=sphmat.dtype, device=sphmat.device), sphmat], dim=-1)
    return _SYM.to_cartesian(sphmat)
