"""
Tests for trial randomization (PRD §19).
"""
import pytest
from collections import Counter
from backend.protocol.randomization import (
    generate_block_randomized_sequence,
    generate_fully_randomized_sequence,
)


class TestRandomization:

    def test_block_randomized_balanced(self):
        seq = generate_block_randomized_sequence(
            classes=["YES", "NO", "HELP"],
            trials_per_class=10,
            seed=42,
        )
        counts = Counter(seq)
        assert counts["YES"] == 10
        assert counts["NO"] == 10
        assert counts["HELP"] == 10

    def test_block_randomized_total_length(self):
        seq = generate_block_randomized_sequence(
            classes=["YES", "NO", "HELP"],
            trials_per_class=20,
            seed=42,
        )
        assert len(seq) == 60

    def test_not_all_same_class_first(self):
        seq = generate_block_randomized_sequence(
            classes=["YES", "NO", "HELP"],
            trials_per_class=10,
            seed=42,
        )
        # Should not be all YES in the first 10 positions
        first_10 = seq[:10]
        assert not all(s == "YES" for s in first_10)

    def test_reproducible_with_same_seed(self):
        seq1 = generate_block_randomized_sequence(["YES", "NO", "HELP"], 10, seed=42)
        seq2 = generate_block_randomized_sequence(["YES", "NO", "HELP"], 10, seed=42)
        assert seq1 == seq2

    def test_different_seed_gives_different_order(self):
        seq1 = generate_block_randomized_sequence(["YES", "NO", "HELP"], 30, seed=1)
        seq2 = generate_block_randomized_sequence(["YES", "NO", "HELP"], 30, seed=2)
        assert seq1 != seq2

    def test_fully_randomized_balanced(self):
        seq = generate_fully_randomized_sequence(["YES", "NO", "HELP"], 10, seed=42)
        counts = Counter(seq)
        assert counts["YES"] == 10
        assert counts["NO"] == 10
        assert counts["HELP"] == 10

    def test_empty_classes_raises(self):
        with pytest.raises(ValueError):
            generate_block_randomized_sequence([], 10, seed=42)

    def test_zero_trials_raises(self):
        with pytest.raises(ValueError):
            generate_block_randomized_sequence(["YES", "NO"], 0, seed=42)
