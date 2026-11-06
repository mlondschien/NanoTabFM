import unittest

import torch

from model import NanoTabFM
from prior import make_batch


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = NanoTabFM().eval()
        self.context_features, self.context_labels, self.query_features, _ = make_batch(
            batch_size=2, num_context=12, num_queries=4, num_features=5
        )

    def test_context_row_order_does_not_change_predictions(self):
        permutation = torch.randperm(self.context_features.shape[1])
        with torch.no_grad():
            expected = self.model(self.context_features, self.context_labels, self.query_features)
            actual = self.model(
                self.context_features[:, permutation],
                self.context_labels[:, permutation],
                self.query_features,
            )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_queries_are_independent(self):
        context = (self.context_features, self.context_labels)
        with torch.no_grad():
            expected = self.model(*context, self.query_features)
            individually = torch.cat(
                [self.model(*context, query[:, None]) for query in self.query_features.unbind(1)],
                dim=1,
            )
            extra_queries = 100 * torch.randn(2, 3, 5)
            together = self.model(*context, torch.cat([self.query_features, extra_queries], dim=1))
        torch.testing.assert_close(individually, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(together[:, :4], expected, atol=1e-5, rtol=1e-5)

    def test_constant_features_and_single_context_row_are_finite(self):
        with torch.no_grad():
            logits = self.model(
                torch.ones(2, 1, 5),
                torch.zeros(2, 1, dtype=torch.long),
                torch.ones(2, 3, 5),
            )
        self.assertTrue(torch.isfinite(logits).all())

    def test_all_parameters_receive_finite_gradients(self):
        self.model.train()
        context_features, context_labels, query_features, query_labels = make_batch(
            batch_size=2, num_context=12, num_queries=4
        )
        logits = self.model(context_features, context_labels, query_features)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), query_labels.flatten())
        loss.backward()
        self.assertEqual(logits.shape, (2, 4, 2))
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
