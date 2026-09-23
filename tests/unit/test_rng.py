"""确定性随机数测试。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from virtual_world.core.rng import DeterministicRNG, derive_seed


def test_same_seed_gives_same_sequence() -> None:
    a = DeterministicRNG(42)
    b = DeterministicRNG(42)
    assert np.array_equal(a.normal(size=10), b.normal(size=10))
    assert not np.array_equal(
        DeterministicRNG(42).normal(size=10), DeterministicRNG(43).normal(size=10)
    )


def test_derive_seed_is_hash_based() -> None:
    expected = int.from_bytes(hashlib.blake2b(b"42:terrain", digest_size=8).digest(), "big")
    assert derive_seed(42, "terrain") == expected
    assert derive_seed(42, "terrain") != derive_seed(42, "climate")


def test_spawn_is_order_independent() -> None:
    terrain = DeterministicRNG(7).spawn("terrain")
    climate = DeterministicRNG(7).spawn("climate")
    assert terrain.seed != climate.seed
    assert DeterministicRNG(7).spawn("climate").seed == climate.seed
    assert np.array_equal(
        terrain.normal(size=5), DeterministicRNG(7).spawn("terrain").normal(size=5)
    )


def test_state_roundtrip() -> None:
    source = DeterministicRNG(123)
    source.normal(size=5)
    state = source.get_state()
    expected = source.normal(size=5)

    restored = DeterministicRNG(0)
    restored.set_state(state)
    assert restored.seed == 123
    assert np.array_equal(restored.normal(size=5), expected)


def test_requires_integer_seed() -> None:
    with pytest.raises(TypeError):
        DeterministicRNG(1.5)  # type: ignore[arg-type]


def test_distribution_helpers() -> None:
    rng = DeterministicRNG(11)
    assert rng.random(4).shape == (4,)
    assert np.all((rng.uniform(0, 1, size=100) >= 0) & (rng.uniform(0, 1, size=100) < 1))
    assert rng.integers(0, 10, size=5).shape == (5,)
    assert rng.choice(np.array([1, 2, 3]), size=3).shape == (3,)
