import unittest

import numpy as np
import pandas as pd
from product_classifier import ProductClassifier


class TestProductClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ProductClassifier()
        self.data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": ["A", "B", "C", "D", "E"],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_handle_missing_values(self):
        data_with_missing_values = self.data.copy()
        data_with_missing_values.iloc[0, 0] = np.nan
        handled_data = self.classifier.handle_missing_values(data_with_missing_values)
        self.assertFalse(handled_data.isnull().any().any())

    def test_visualize_data(self):
        self.classifier.visualize_data(self.data)
        fig = plt.gcf()
        self.assertIsNotNone(fig)

    def test_evaluate_model(self):
        # Create a mock model and a mock dataset
        mock_model = RandomForestClassifier()
        mock_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [6, 7, 8, 9, 10],
                "target": [0, 1, 0, 1, 0],
            }
        )

        # Split the mock data into training and testing sets
        X = mock_data.drop("target", axis=1)
        y = mock_data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the mock model
        mock_model.fit(X_train, y_train)

        # Pass the mock model and the mock dataset to the method
        accuracy = self.classifier.evaluate_model(mock_model, X_test, y_test)

        # Assert that the method returns the expected evaluation metrics
        self.assertGreaterEqual(accuracy, 0.5)

    def tearDown(self):
        del self.classifier
        del self.data


if __name__ == "__main__":
    unittest.main()
