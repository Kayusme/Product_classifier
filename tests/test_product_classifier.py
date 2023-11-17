import unittest

import numpy as np
import pandas as pd
from product_classifier import ProductClassifier


class TestProductClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ProductClassifier()
        self.data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 0]
        })

    def test_handle_missing_values(self):
        self.classifier.handle_missing_values(self.data)
        self.assertFalse(self.data.isnull().any().any())

    def test_visualize_data(self):
        self.classifier.visualize_data(self.data)
        fig = plt.gcf()
        self.assertIsNotNone(fig)

    def test_evaluate_model(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.train_model(X_train, y_train)
        accuracy = self.classifier.evaluate_model(X_test, y_test)
        self.assertGreaterEqual(accuracy, 0.5)

    def tearDown(self):
        del self.classifier
        del self.data

if __name__ == '__main__':
    unittest.main()
