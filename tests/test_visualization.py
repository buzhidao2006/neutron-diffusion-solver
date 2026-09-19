import numpy as np

from visualization import build_convergence_figure, final_convergence_rate


def test_convergence_figure_has_eigenvalue_and_residual_axes():
    figure = build_convergence_figure({
        "k_history": [0.9, 1.0, 1.02],
        "residual": [1e-1, 1e-3, 1e-8],
    })

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "k_eff 收敛历史"
    assert figure.axes[1].get_yscale() == "log"


def test_convergence_figure_handles_empty_histories():
    figure = build_convergence_figure({})

    assert len(figure.axes) == 2


def test_final_convergence_rate_uses_last_two_changes():
    result = {"k_history": [1.0, 1.2, 1.25, 1.26]}

    assert np.isclose(final_convergence_rate(result), 0.2)


def test_final_convergence_rate_requires_three_values():
    assert final_convergence_rate({"k_history": [1.0, 1.1]}) is None
