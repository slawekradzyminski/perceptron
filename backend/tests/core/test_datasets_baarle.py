"""Tests for Baarle-Hertog dataset (Chapter 4)."""

from backend.core.datasets import make_baarle_hertog_dataset


class TestBaarleHertogDataset:
    def test_creates_samples(self) -> None:
        samples = make_baarle_hertog_dataset(n_samples=100, seed=42)
        assert len(samples) > 0
        assert len(samples) <= 100

    def test_sample_structure(self) -> None:
        samples = make_baarle_hertog_dataset(n_samples=50, seed=42)
        for sample in samples:
            assert "x" in sample
            assert "y" in sample
            assert len(sample["x"]) == 2
            assert sample["y"] in (0, 1)

    def test_coordinates_normalized(self) -> None:
        samples = make_baarle_hertog_dataset(n_samples=100, seed=42)
        for sample in samples:
            x1, x2 = sample["x"]
            assert -1 <= x1 <= 1
            assert -1 <= x2 <= 1

    def test_both_classes_present(self) -> None:
        samples = make_baarle_hertog_dataset(n_samples=100, seed=42)
        labels = {s["y"] for s in samples}
        assert 0 in labels
        assert 1 in labels

    def test_deterministic_with_seed(self) -> None:
        samples1 = make_baarle_hertog_dataset(n_samples=50, seed=42)
        samples2 = make_baarle_hertog_dataset(n_samples=50, seed=42)
        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1["x"] == s2["x"]
            assert s1["y"] == s2["y"]

    def test_different_seeds_different_samples(self) -> None:
        samples1 = make_baarle_hertog_dataset(n_samples=50, seed=42)
        samples2 = make_baarle_hertog_dataset(n_samples=50, seed=123)
        # At least some samples should differ
        different_count = sum(1 for s1, s2 in zip(samples1, samples2) if s1["x"] != s2["x"])
        assert different_count > 0

