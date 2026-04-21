"""
Unit tests for gpu_aware_mpi lifecycle in a singleton world.

Contract under test: docs/gpu-aware-mpi.md §3.1 — `ensure_mpi_initialized`
is lazy and idempotent; `get_rank` / `get_size` / `barrier` work without
`mpirun` (singleton MPI world of size 1).

Notes for maintainers:
- These tests call into the C++ extension exactly once to init MPI. Once
  MPI is initialized in a Python process, most MPI implementations do not
  allow re-initialization after MPI_Finalize — so we deliberately do NOT
  call finalize_mpi from these tests.
- The tests must be colocated with other tests that are safe to run
  post-MPI-init. Anything that expected a pristine MPI state should run
  in its own process (pytest --forked handles this).
"""

import pytest

pytest.importorskip("gpu_aware_mpi")


def test_get_rank_is_zero_in_singleton_world():
    from gpu_aware_mpi import get_rank

    # Without mpirun, OpenMPI initializes a singleton world where the sole
    # process is rank 0.
    assert get_rank() == 0


def test_get_size_is_one_in_singleton_world():
    from gpu_aware_mpi import get_size

    assert get_size() == 1


def test_get_rank_is_idempotent():
    from gpu_aware_mpi import get_rank

    first = get_rank()
    second = get_rank()

    assert first == second == 0


def test_barrier_returns_without_hanging():
    # In singleton world, MPI_Barrier returns immediately.
    from gpu_aware_mpi import barrier

    # No assertion on return value — the contract is "doesn't hang, doesn't
    # raise". A unit-test-level timeout from pytest-timeout would catch a
    # regression that made it block.
    barrier()
