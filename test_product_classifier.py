import unittest

import pandas as pd
from product_classifier import ProductClassifier
from sklearn.ensemble import RandomForestClassifier


class TestProductClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ProductClassifier()
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': ['A', 'B', 'A', 'B', 'A']
        })

    def test_load_data(self):
        data = self.classifier.load_data('test.csv')
        self.assertTrue(isinstance(data, pd.DataFrame))

    def test_preprocess_data(self):
        processed_data = self.classifier.preprocess_data(self.data)
        self.assertEqual(processed_data.isnull().sum().sum(), 0)

    def test_evaluate_model(self):
        self.classifier.model = RandomForestClassifier()
        self.classifier.model.fit(self.data.drop('target', axis=1), self.data['target'])
        X_test = self.data.drop('target', axis=1)
        y_test = self.data['target']
        self.classifier.evaluate_model(X_test, y_test)
        self.assertTrue(True)

    def test_train_model(self):
        X_test, y_test = self.classifier.train_model(self.data)
        self.assertTrue(isinstance(X_test, pd.DataFrame))
        self.assertTrue(isinstance(y_test, pd.Series))

if __name__ == '__main__':
    unittest.main()
