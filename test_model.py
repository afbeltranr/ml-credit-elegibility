import unittest
from train_logistic_model import logreg

class TestLogisticModel(unittest.TestCase):
    def test_model_exists(self):
        """Test if the logistic regression model has been created successfully."""
        self.assertIsNotNone(logreg, "The logistic regression model should not be None.")

    def test_model_accuracy(self):
        """Test if the model accuracy is within a reasonable range (example: > 0.5)."""
        from train_logistic_model import accuracy
        self.assertGreaterEqual(accuracy, 0.5, "The model accuracy should be greater than or equal to 0.5.")

if __name__ == '__main__':
    unittest.main()
