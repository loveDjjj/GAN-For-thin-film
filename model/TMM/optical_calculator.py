import numpy as np
import torch

from model.TMM.TMM import TMM_solver
from data.myindex import index


def cache_master_compatible_metal_refractive_indices(params, device):
    """Cache master-compatible metal indices once and reuse them on GPU."""
    metal_refractive_indices = getattr(params, "metal_refractive_indices", None)
    if metal_refractive_indices is None:
        wavelengths_um = np.asarray(2 * np.pi / params.k.detach().cpu().numpy(), dtype=np.float64)
        metal_interp = index(params.metal_name)
        metal_values = np.asarray(metal_interp(wavelengths_um), dtype=np.complex128).reshape(-1)
        metal_refractive_indices = torch.as_tensor(
            metal_values,
            dtype=torch.complex128,
            device=device,
        )
        params.metal_refractive_indices = metal_refractive_indices
        params.metal_n_database = metal_refractive_indices.real.detach().cpu()
        params.metal_k_database = metal_refractive_indices.imag.detach().cpu()
    return metal_refractive_indices.to(device=device, dtype=torch.complex128)


def calculate_reflection(thicknesses, refractive_indices, params, device):
    """Calculate reflectance with master-compatible metal interpolation and GPU caching."""
    batch_size = thicknesses.size(0)
    k = params.k.to(device)
    complex_dtype = torch.complex128

    air_refractive = torch.ones((batch_size, 1, len(k)), dtype=complex_dtype, device=device)
    metal_refractive = cache_master_compatible_metal_refractive_indices(params, device)
    metal_refractive = metal_refractive.view(1, 1, -1).expand(batch_size, -1, -1)

    refractive_indices = refractive_indices.to(complex_dtype)
    refractive_indices = torch.cat((air_refractive, refractive_indices), dim=1)
    refractive_indices = torch.cat((refractive_indices, metal_refractive), dim=1)
    refractive_indices = torch.cat((refractive_indices, air_refractive), dim=1)

    air_length = torch.full((batch_size, 1), float("inf"), device=device, dtype=thicknesses.dtype)
    metal_length = torch.full((batch_size, 1), 0.2, device=device, dtype=thicknesses.dtype)
    thicknesses = torch.cat((air_length, thicknesses, metal_length, air_length), dim=1)

    thicknesses = thicknesses.to(complex_dtype)
    thicknesses = thicknesses.unsqueeze(-1).expand(-1, -1, len(k))
    thicknesses = torch.permute(thicknesses, (0, 2, 1))

    reflection, _ = TMM_solver(thicknesses, refractive_indices, k, params.theta, params.pol)
    return reflection
