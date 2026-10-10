"""Published homogeneous bare-slab two-group equations as a code benchmark.

The continuous reference follows the sinusoidal fundamental-mode derivation
in R. E. Pevey, *Benchmarking Report for WIGGLE*, DOE/OSTI 6398033 (1990).
Its coefficients here are this repository's teaching defaults, not measured
material data. This is solution verification, not experimental validation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvals

from solver import DEFAULTS, solve_two_group


TEACHING_CASES = {
    "default": {"length": 200.0, "sections": {}},
    "more_absorption": {"length": 160.0, "sections": {"Sa2": 0.10}},
    "more_scattering": {
        "length": 240.0,
        "sections": {"D1": 1.5, "D2": 0.5, "Ss12": 0.030, "nu_Sf2": 0.12},
    },
}


@dataclass(frozen=True)
class TwoGroupSlabResult:
    n_nodes: int
    k_numeric: float
    k_discrete: float
    k_continuous: float
    relative_error: float
    flux_shape_l2_error: float
    k_independent: float


def independent_matrix_eigenvalue(n_nodes: int, length: float, sections: dict) -> float:
    """Independently assemble F v = k A v and solve its generalized spectrum."""
    h = length / (n_nodes + 1)
    laplacian = np.diag(np.full(n_nodes, 2.0))
    laplacian += np.diag(np.full(n_nodes - 1, -1.0), 1)
    laplacian += np.diag(np.full(n_nodes - 1, -1.0), -1)
    identity = np.eye(n_nodes)
    zeros = np.zeros((n_nodes, n_nodes))
    a = np.block([
        [sections["D1"] * laplacian / h**2
         + (sections["Sa1"] + sections["Ss12"]) * identity, zeros],
        [-sections["Ss12"] * identity,
         sections["D2"] * laplacian / h**2 + sections["Sa2"] * identity],
    ])
    f = np.block([
        [sections["nu_Sf1"] * identity, sections["nu_Sf2"] * identity],
        [zeros, zeros],
    ])
    spectrum = eigvals(f, a)
    physical = spectrum[(abs(spectrum.imag) < 1e-8) & (spectrum.real > 0)]
    if not len(physical):
        raise RuntimeError("No positive real eigenvalue in independent matrix solve.")
    return float(max(physical.real))


def run_two_group_slab_benchmark(
    n_nodes: int, length: float = 200.0, sections: dict | None = None,
) -> TwoGroupSlabResult:
    """Compare the 1D two-group solver with continuous and discrete modes."""
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer.")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")

    p = {**DEFAULTS, **(sections or {})}
    result = solve_two_group(
        L=length, N=n_nodes, sections=p, max_iter=1000, tol=1e-11,
        raise_on_nonconvergence=True,
    )

    def eigenvalue(buckling_squared):
        fast_loss = p['D1'] * buckling_squared + p['Sa1'] + p['Ss12']
        thermal_loss = p['D2'] * buckling_squared + p['Sa2']
        return (p['nu_Sf1'] + p['nu_Sf2'] * p['Ss12'] / thermal_loss) / fast_loss

    h = length / (n_nodes + 1)
    continuous_b2 = (np.pi / length) ** 2
    discrete_b2 = 4 * np.sin(np.pi / (2 * (n_nodes + 1))) ** 2 / h**2
    k_continuous = eigenvalue(continuous_b2)
    k_discrete = eigenvalue(discrete_b2)

    # Shape alone is compared: eigenvector amplitude is arbitrary.
    thermal_ratio = p['Ss12'] / (p['D2'] * discrete_b2 + p['Sa2'])
    sine = np.sin(np.pi * result['x'] / length)
    reference_flux = np.concatenate((sine, thermal_ratio * sine))
    reference_flux /= np.max(reference_flux)
    flux_shape_error = np.linalg.norm(result['phi'] - reference_flux) / np.linalg.norm(reference_flux)

    return TwoGroupSlabResult(
        n_nodes=n_nodes,
        k_numeric=float(result['k_eff']),
        k_discrete=float(k_discrete),
        k_continuous=float(k_continuous),
        relative_error=float(abs(result['k_eff'] - k_continuous) / k_continuous),
        flux_shape_l2_error=float(flux_shape_error),
        k_independent=independent_matrix_eigenvalue(n_nodes, length, p),
    )


def convergence_study(n_nodes_values=(20, 40, 80, 160), length=200.0, sections=None):
    """Run a reproducible spatial refinement sequence."""
    return [run_two_group_slab_benchmark(n, length, sections) for n in n_nodes_values]


if __name__ == '__main__':
    print('N  k_numeric     k_discrete    k_continuous  rel_k_error  flux_shape_L2')
    for case in convergence_study():
        print(f'{case.n_nodes:3d} {case.k_numeric:.9f}  {case.k_discrete:.9f}  '
              f'{case.k_continuous:.9f}  {case.relative_error:.3e}   '
              f'{case.flux_shape_l2_error:.3e}')
