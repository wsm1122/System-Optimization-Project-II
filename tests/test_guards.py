import unittest

import torch

from src.retrieval_guards import validate_neighbors


class NeighborGuardTests(unittest.TestCase):
    def test_valid_ids_are_accepted(self):
        distances = torch.tensor([[0.0, 1.0]])
        ids = torch.tensor([[0, 2]])
        checked_distances, checked_ids = validate_neighbors(distances, ids, 3)
        self.assertIs(checked_distances, distances)
        self.assertIs(checked_ids, ids)

    def test_negative_and_upper_bound_ids_are_rejected(self):
        distances = torch.tensor([[0.0]])
        for neighbor_id in (-1, 3):
            with self.subTest(neighbor_id=neighbor_id):
                with self.assertRaises(ValueError):
                    validate_neighbors(distances, torch.tensor([[neighbor_id]]), 3)

    def test_nonfinite_distance_is_rejected(self):
        for distance in (float("nan"), float("inf")):
            with self.subTest(distance=distance):
                with self.assertRaises(ValueError):
                    validate_neighbors(torch.tensor([[distance]]), torch.tensor([[0]]), 1)


if __name__ == "__main__":
    unittest.main()
