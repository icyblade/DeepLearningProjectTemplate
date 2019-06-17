"""Test for main module."""
from deep_learning.main import MeanRegressionModel


def test_mean_regression_model():
    """Test MeanRegressionModel."""
    features = [[1, 2, 3], [4, 5, 6]]
    labels = [2, 5]

    model = MeanRegressionModel().fit(features, labels)
    assert (model.predict(features) == labels).all()
