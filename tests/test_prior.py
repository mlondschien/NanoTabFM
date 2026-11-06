import unittest

import torch

from prior import make_batch


class PriorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_and_binary_labels(self):
        for num_features in (1, 4):
            context_features, context_labels, query_features, query_labels = make_batch(
                batch_size=3, num_context=7, num_queries=5, num_features=num_features
            )
            self.assertEqual(context_features.shape, (3, 7, num_features))
            self.assertEqual(query_features.shape, (3, 5, num_features))
            self.assertEqual(context_labels.shape, (3, 7))
            self.assertEqual(query_labels.shape, (3, 5))
            labels = torch.cat([context_labels, query_labels], dim=1)
            self.assertTrue(((labels == 0) | (labels == 1)).all())
            self.assertTrue(torch.isfinite(context_features).all())
            self.assertTrue(torch.isfinite(query_features).all())

    def test_between_5_and_95_percent_of_rows_are_positive(self):
        _, context_labels, _, query_labels = make_batch(
            batch_size=64, num_context=90, num_queries=10, num_features=3
        )
        positives = torch.cat([context_labels, query_labels], dim=1).sum(dim=1)
        self.assertTrue(((positives >= 5) & (positives <= 95)).all())
        self.assertGreater(positives.float().std(), 10)

    def test_columns_are_continuous_or_integer_codes(self):
        context_features, _, _, _ = make_batch(
            batch_size=16, num_context=200, num_queries=10, num_features=10
        )
        columns = context_features.transpose(1, 2).flatten(0, 1)
        is_code = (columns == columns.round()).all(dim=1) & (columns >= 0).all(dim=1)
        is_code &= (columns <= 9).all(dim=1)
        self.assertTrue(is_code.any() and not is_code.all())


if __name__ == "__main__":
    unittest.main()
