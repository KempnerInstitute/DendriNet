"""Dense-reference tests for the matrix-free block singular solver."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.utils.state_transition import (
    BlockSubspaceSingularValueEstimate,
    leading_matrix_free_singular_value_block_subspace,
    leading_state_transition_singular_value,
)


def _orthogonal(rows: int, columns: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(rows, columns, dtype=torch.float64, generator=generator)
    result, _ = torch.linalg.qr(matrix, mode="reduced")
    return result


def _matrix_with_spectrum(
    singular_values: list[float],
    *,
    output_dimension: int | None = None,
) -> torch.Tensor:
    input_dimension = len(singular_values)
    output_dimension = output_dimension or input_dimension
    if output_dimension < input_dimension:
        raise ValueError("test matrix requires output_dimension >= input_dimension")
    left = _orthogonal(output_dimension, input_dimension, seed=113)
    right = _orthogonal(input_dimension, input_dimension, seed=127)
    diagonal = torch.diag(torch.tensor(singular_values, dtype=torch.float64))
    return left @ diagonal @ right.mT


def _solve(
    matrix: torch.Tensor,
    **kwargs: object,
) -> tuple[BlockSubspaceSingularValueEstimate, dict[str, int]]:
    calls = {"forward": 0, "adjoint": 0}

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        calls["forward"] += 1
        return matrix @ vector

    def rmatvec(vector: torch.Tensor) -> torch.Tensor:
        calls["adjoint"] += 1
        return matrix.mT @ vector

    result = leading_matrix_free_singular_value_block_subspace(
        input_dimension=matrix.shape[1],
        output_dimension=matrix.shape[0],
        device=matrix.device,
        dtype=matrix.dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        **kwargs,
    )
    return result, calls


class _DenseLinearization:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        self.dimension = matrix.shape[1]
        self.device = matrix.device
        self.dtype = matrix.dtype

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matrix @ vector

    def vjp(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matrix.mT @ vector


def test_block_solver_matches_rectangular_dense_svd_and_residual() -> None:
    matrix = _matrix_with_spectrum(
        [9.0, 5.0, 3.0, 2.0, 1.0, 0.7, 0.3, 0.2, 0.1, 0.08, 0.05, 0.01],
        output_dimension=15,
    )
    result, calls = _solve(
        matrix,
        block_size=3,
        max_iterations=4,
        relative_residual_tolerance=1e-11,
        absolute_residual_tolerance=1e-13,
        seed=19,
    )

    expected = float(torch.linalg.svdvals(matrix)[0])
    assert result.converged
    assert result.singular_value == pytest.approx(expected, rel=1e-11, abs=1e-12)
    torch.testing.assert_close(
        matrix @ result.right_vector,
        result.singular_value * result.left_vector,
        rtol=1e-11,
        atol=1e-12,
    )
    assert result.relative_residual_norm <= 1e-11 + 1e-13 / expected
    assert calls == {
        "forward": result.block_size * result.iterations,
        "adjoint": result.block_size * result.iterations,
    }
    assert result.metadata()["operator_calls"]["total"] == sum(calls.values())


def test_block_solver_resolves_near_degenerate_leading_cluster() -> None:
    # A one-vector method converges according to 4.999 / 5.0 here.  Retaining
    # the four-vector cluster makes convergence depend on 1.0 / 4.997 instead.
    matrix = _matrix_with_spectrum(
        [
            5.0,
            4.999,
            4.998,
            4.997,
            1.0,
            0.98,
            0.96,
            0.94,
            *[0.9 - 0.02 * index for index in range(32)],
        ]
    )
    result, _ = _solve(
        matrix,
        block_size=4,
        max_iterations=10,
        relative_residual_tolerance=1e-10,
        absolute_residual_tolerance=1e-12,
        seed=23,
    )

    assert result.converged
    assert result.iterations <= 10
    assert result.singular_value == pytest.approx(5.0, rel=1e-10, abs=1e-11)
    assert result.relative_residual_norm <= 1.1e-10


@pytest.mark.parametrize(
    "singular_values,block_size",
    [
        ([4.0] * 12, 4),
        ([6.0, 6.0, 6.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05], 3),
    ],
)
def test_block_solver_handles_exact_leading_degeneracy(
    singular_values: list[float],
    block_size: int,
) -> None:
    matrix = _matrix_with_spectrum(singular_values)
    result, _ = _solve(
        matrix,
        block_size=block_size,
        max_iterations=3 if block_size == 4 else 4,
        relative_residual_tolerance=1e-11,
        absolute_residual_tolerance=1e-13,
        seed=29,
    )

    assert result.converged
    assert result.singular_value == pytest.approx(singular_values[0], rel=1e-11)
    assert result.relative_residual_norm <= 1.1e-11
    if len(set(singular_values)) == 1:
        assert result.iterations == 1


def test_block_solver_fails_closed_at_exact_operator_call_budget() -> None:
    matrix = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    initial = torch.ones(2, 1, dtype=torch.float64)
    result, calls = _solve(
        matrix,
        block_size=1,
        max_iterations=1,
        relative_residual_tolerance=0.0,
        absolute_residual_tolerance=0.0,
        seed=31,
        initial_block=initial,
    )

    assert not result.converged
    assert result.iterations == 1
    assert calls == {"forward": 1, "adjoint": 1}
    assert result.matvec_calls == result.maximum_matvec_calls == 1
    assert result.rmatvec_calls == result.maximum_rmatvec_calls == 1
    assert result.residual_norm > 0.0


def test_block_solver_probes_beyond_an_explicit_invariant_start() -> None:
    matrix = torch.diag(torch.tensor([1.0, 10.0], dtype=torch.float64))
    initial = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    result, calls = _solve(
        matrix,
        block_size=1,
        max_iterations=2,
        relative_residual_tolerance=1e-12,
        absolute_residual_tolerance=1e-14,
        seed=33,
        initial_block=initial,
    )

    assert result.converged
    assert result.iterations == 2
    assert result.singular_value == pytest.approx(10.0, rel=1e-12)
    assert calls == {"forward": 2, "adjoint": 2}
    metadata = result.metadata()
    assert metadata["full_input_space_coverage"] is True
    assert metadata["global_leading_certificate"] is True
    assert metadata["residual_certificate_scope"] == (
        "extracted_ritz_singular_triplet_only"
    )


def test_block_solver_adapts_omitted_defaults_to_small_dimension() -> None:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64))
    result, calls = _solve(matrix)

    assert result.converged
    assert result.singular_value == pytest.approx(4.0)
    assert result.block_size == 4
    assert result.maximum_subspace_dimension == 4
    assert result.maximum_matvec_calls == 4
    assert result.maximum_rmatvec_calls == 4
    assert calls == {"forward": 4, "adjoint": 4}


def test_block_solver_resolves_each_individually_omitted_budget() -> None:
    matrix = torch.eye(12, dtype=torch.float64)

    block_only, _ = _solve(matrix, block_size=5)
    iterations_only, _ = _solve(matrix, max_iterations=3)

    assert block_only.block_size == 5
    assert block_only.maximum_subspace_dimension == 10
    assert iterations_only.block_size == 4
    assert iterations_only.maximum_subspace_dimension == 12


def test_block_solver_preserves_explicit_overbudget_rejection() -> None:
    identity = torch.eye(4, dtype=torch.float64)

    with pytest.raises(ValueError, match=r"block_size \* max_iterations"):
        leading_matrix_free_singular_value_block_subspace(
            input_dimension=4,
            output_dimension=4,
            device=identity.device,
            dtype=identity.dtype,
            matvec=lambda vector: identity @ vector,
            rmatvec=lambda vector: identity @ vector,
            block_size=2,
            max_iterations=3,
        )


def test_block_solver_escapes_an_initial_nullspace_block() -> None:
    matrix = torch.diag(torch.tensor([0.0, 0.0, 5.0, 3.0], dtype=torch.float64))
    initial = torch.eye(4, dtype=torch.float64)[:, :2]
    result, calls = _solve(
        matrix,
        block_size=2,
        max_iterations=2,
        relative_residual_tolerance=1e-12,
        absolute_residual_tolerance=1e-14,
        seed=37,
        initial_block=initial,
    )

    assert result.converged
    assert result.iterations == 2
    assert result.subspace_dimension == 4
    assert result.singular_value == pytest.approx(5.0, rel=1e-12)
    assert calls == {"forward": 4, "adjoint": 4}


def test_block_solver_does_not_certify_zero_from_a_partial_nullspace() -> None:
    matrix = torch.diag(torch.tensor([0.0, 0.0, 5.0, 3.0], dtype=torch.float64))
    initial = torch.eye(4, dtype=torch.float64)[:, :2]
    result, calls = _solve(
        matrix,
        block_size=2,
        max_iterations=1,
        relative_residual_tolerance=1e-12,
        absolute_residual_tolerance=1e-14,
        seed=41,
        initial_block=initial,
    )

    assert result.singular_value == 0.0
    assert not result.converged
    assert result.subspace_dimension == 2
    assert result.maximum_subspace_dimension == 2
    assert calls == {"forward": 2, "adjoint": 2}
    assert "zero_requires_full_input_space_coverage" in str(
        result.metadata()["convergence_criterion"]
    )


def test_block_solver_certifies_a_true_zero_only_after_full_space_coverage() -> None:
    matrix = torch.zeros(4, 4, dtype=torch.float64)
    initial = torch.eye(4, dtype=torch.float64)[:, :2]
    result, calls = _solve(
        matrix,
        block_size=2,
        max_iterations=2,
        relative_residual_tolerance=0.0,
        absolute_residual_tolerance=0.0,
        seed=43,
        initial_block=initial,
    )

    assert result.converged
    assert result.singular_value == 0.0
    assert result.iterations == 2
    assert result.subspace_dimension == 4
    assert calls == {"forward": 4, "adjoint": 4}


def test_scalar_power_iteration_restarts_from_an_explicit_null_direction() -> None:
    matrix = torch.diag(torch.tensor([0.0, 5.0], dtype=torch.float64))
    result = leading_state_transition_singular_value(
        _DenseLinearization(matrix),  # type: ignore[arg-type]
        max_iterations=12,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        seed=47,
        initial_vector=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )

    assert result.converged
    assert result.iterations > 1
    assert result.singular_value == pytest.approx(5.0, rel=1e-12)


@pytest.mark.parametrize("diagonal", ([1.0, 10.0], [10.0, 1.0]))
def test_scalar_power_iteration_probes_beyond_an_explicit_invariant_start(
    diagonal: list[float],
) -> None:
    matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    result = leading_state_transition_singular_value(
        _DenseLinearization(matrix),  # type: ignore[arg-type]
        max_iterations=20,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        seed=49,
        initial_vector=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )

    assert result.converged
    assert result.iterations > 2
    assert result.singular_value == pytest.approx(10.0, rel=1e-12)
    metadata = result.metadata()
    assert metadata["global_leading_certificate"] is False
    assert metadata["estimate_scope"] == ("randomized_leading_singular_value_estimate")


def test_scalar_power_iteration_null_direction_fails_closed_at_budget() -> None:
    matrix = torch.diag(torch.tensor([0.0, 5.0], dtype=torch.float64))
    result = leading_state_transition_singular_value(
        _DenseLinearization(matrix),  # type: ignore[arg-type]
        max_iterations=1,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        seed=53,
        initial_vector=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )

    assert result.singular_value == 0.0
    assert not result.converged
    assert result.iterations == 1


def test_scalar_power_iteration_does_not_certify_multidimensional_zero() -> None:
    matrix = torch.zeros(2, 2, dtype=torch.float64)
    result = leading_state_transition_singular_value(
        _DenseLinearization(matrix),  # type: ignore[arg-type]
        max_iterations=3,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        seed=59,
        initial_vector=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )

    assert result.singular_value == 0.0
    assert not result.converged
    assert result.iterations == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_block_solver_rejects_dtypes_unsupported_by_block_linalg(
    dtype: torch.dtype,
) -> None:
    identity = torch.eye(2, dtype=dtype)

    with pytest.raises(TypeError, match=r"torch\.float32 or torch\.float64"):
        leading_matrix_free_singular_value_block_subspace(
            input_dimension=2,
            output_dimension=2,
            device=identity.device,
            dtype=dtype,
            matvec=lambda vector: identity @ vector,
            rmatvec=lambda vector: identity @ vector,
            block_size=1,
            max_iterations=2,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scalar_power_iteration_does_not_require_block_linalg_dtype(
    dtype: torch.dtype,
) -> None:
    identity = torch.eye(2, dtype=dtype)

    result = leading_state_transition_singular_value(
        _DenseLinearization(identity),  # type: ignore[arg-type]
        max_iterations=2,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-3,
        seed=61,
    )

    assert result.converged
    assert result.singular_value == 1.0


def test_block_solver_rejects_rank_deficient_initial_block() -> None:
    identity = torch.eye(4, dtype=torch.float64)
    initial = torch.ones(4, 2, dtype=torch.float64)

    def operator(vector: torch.Tensor) -> torch.Tensor:
        return identity @ vector

    with pytest.raises(ValueError, match="full column rank"):
        leading_matrix_free_singular_value_block_subspace(
            input_dimension=4,
            output_dimension=4,
            device=identity.device,
            dtype=identity.dtype,
            matvec=operator,
            rmatvec=operator,
            block_size=2,
            max_iterations=2,
            initial_block=initial,
        )


def test_valid_initial_block_can_require_multiple_krylov_expansions() -> None:
    matrix = _matrix_with_spectrum([7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    initial = _orthogonal(8, 2, seed=211)
    result, _ = _solve(
        matrix,
        block_size=2,
        max_iterations=4,
        relative_residual_tolerance=1e-12,
        absolute_residual_tolerance=1e-14,
        seed=43,
        initial_block=initial,
    )

    assert result.converged
    assert result.iterations > 1
    assert result.singular_value == pytest.approx(7.0, rel=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_dimension", 4.0),
        ("output_dimension", 4.0),
        ("block_size", 2.0),
        ("max_iterations", 2.0),
        ("input_dimension", True),
        ("block_size", False),
    ],
)
def test_block_solver_rejects_non_integral_dimension_and_budget_types(
    field: str,
    value: object,
) -> None:
    identity = torch.eye(4, dtype=torch.float64)
    kwargs: dict[str, object] = {
        "input_dimension": 4,
        "output_dimension": 4,
        "device": identity.device,
        "dtype": identity.dtype,
        "matvec": lambda vector: identity @ vector,
        "rmatvec": lambda vector: identity @ vector,
        "block_size": 2,
        "max_iterations": 2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=r"integer|block_size"):
        leading_matrix_free_singular_value_block_subspace(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    (
        ("relative_residual_tolerance", True, TypeError, "real number"),
        ("absolute_residual_tolerance", "0.1", TypeError, "real number"),
        ("relative_residual_tolerance", float("nan"), ValueError, "finite"),
        ("absolute_residual_tolerance", float("inf"), ValueError, "finite"),
        ("relative_residual_tolerance", -1.0, ValueError, "non-negative"),
    ),
)
def test_block_solver_rejects_invalid_residual_tolerances(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    identity = torch.eye(4, dtype=torch.float64)
    kwargs: dict[str, object] = {
        "input_dimension": 4,
        "output_dimension": 4,
        "device": identity.device,
        "dtype": identity.dtype,
        "matvec": lambda vector: identity @ vector,
        "rmatvec": lambda vector: identity @ vector,
        "block_size": 2,
        "max_iterations": 2,
    }
    kwargs[field] = value

    with pytest.raises(error, match=message):
        leading_matrix_free_singular_value_block_subspace(**kwargs)
