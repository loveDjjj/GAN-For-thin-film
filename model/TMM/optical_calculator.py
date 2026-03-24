import torch

from model.TMM.TMM import TMM_solver
from data.myindex import MatDatabase


def _get_metal_refractive_indices(params, device, dtype):
    """Return cached metal refractive indices on the requested device/dtype."""
    metal_refractive_indices = getattr(params, "metal_refractive_indices", None)
    if metal_refractive_indices is None:
        metal_database = MatDatabase([params.metal_name])
        metal_n_database, metal_k_database = metal_database.interp_wv(
            2 * torch.pi / params.k,
            [params.metal_name],
            False,
        )
        params.metal_n_database = metal_n_database.squeeze(0)
        params.metal_k_database = metal_k_database.squeeze(0)
        params.metal_refractive_indices = (
            params.metal_n_database.to(device=device) + 1j * params.metal_k_database.to(device=device)
        )
        metal_refractive_indices = params.metal_refractive_indices
    return metal_refractive_indices.to(device=device, dtype=dtype)


def calculate_reflection(thicknesses, refractive_indices, params, device):
    """Calculate reflectance for multilayer films with cached metal optical constants."""
    batch_size = thicknesses.size(0)
    k = params.k.to(device)
    complex_dtype = refractive_indices.dtype if torch.is_complex(refractive_indices) else torch.complex64

    air_refractive = torch.ones((batch_size, 1, len(k)), dtype=complex_dtype, device=device)
    metal_refractive = _get_metal_refractive_indices(params, device, complex_dtype)
    metal_refractive = metal_refractive.view(1, 1, -1).expand(batch_size, -1, -1)

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
