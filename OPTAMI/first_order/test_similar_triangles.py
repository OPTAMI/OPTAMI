import unittest
import unittest.mock
import torch
from .similar_triangles import SimilarTriangles


class TestSimilarTriangles(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 1)
        self.criterion = torch.nn.MSELoss()
        self.x = torch.Tensor([2, 1])
        self.y = torch.Tensor([3])

    def test_invalid_L(self):
        with self.assertRaises(ValueError):
            SimilarTriangles(self.model.parameters(), L=-1)

    def test_not_adaptive(self):
        with unittest.mock.patch.object(SimilarTriangles, '_check_relaxation', return_value=None) as mock:
            self.optimizer = SimilarTriangles(
                self.model.parameters(), is_adaptive=False)
            self.closure()
            self.optimizer.step(self.closure)

        with self.assertRaises(AssertionError):
            mock.assert_called()

    def test_not_adaptive_oracle_calls(self):
        self.zero_order_calls = 0

        self.optimizer = SimilarTriangles(
            self.model.parameters(), is_adaptive=False)
        self.counting_closure()
        self.optimizer.step(self.counting_closure)

        self.assertEqual(self.zero_order_calls, 2)

        self.zero_order_calls = 0

    def test_adaptive_oracle_calls(self):
        self.zero_order_calls = 0
        self.optimizer = SimilarTriangles(
            self.model.parameters(), is_adaptive=True, verbose=False)
        self.counting_closure()
        self.optimizer.step(self.counting_closure)
        self.assertEqual(self.zero_order_calls, 3)
        self.zero_order_calls = 0
    
    def test_adaptive_iters(self):
        self.optimizer = SimilarTriangles(
            self.model.parameters(), L=1e+2, is_adaptive=True, verbose=False)
        self.optimizer._check_relaxation = unittest.mock.MagicMock(side_effect=self.optimizer._check_relaxation)
        self.closure()
        self.optimizer.step(self.closure)

        self.assertEqual(self.optimizer._check_relaxation.call_count, 1)

        self.optimizer = SimilarTriangles(
            self.model.parameters(), L=1e+0, is_adaptive=True, verbose=False)
        self.optimizer._check_relaxation = unittest.mock.MagicMock(side_effect=self.optimizer._check_relaxation)
        self.closure()
        self.optimizer.step(self.closure)

        self.assertEqual(self.optimizer._check_relaxation.call_count, 5)

        self.optimizer = SimilarTriangles(
            self.model.parameters(), L=1e-1, is_adaptive=True, verbose=False)
        self.optimizer._check_relaxation = unittest.mock.MagicMock(side_effect=self.optimizer._check_relaxation)
        self.closure()
        self.optimizer.step(self.closure)

        self.assertEqual(self.optimizer._check_relaxation.call_count, 8)

    def closure(self):
        loss = self.criterion(self.model(self.x), self.y)
        self.optimizer.zero_grad()
        return loss

    def counting_closure(self):
        global zero_order_calls

        loss = self.criterion(self.model(self.x), self.y)
        self.optimizer.zero_grad()
        self.zero_order_calls += 1

        return loss


if __name__ == '__main__':
    unittest.main()
