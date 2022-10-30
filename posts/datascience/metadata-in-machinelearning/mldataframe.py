"""Provides MLDataFrame class holding machine learning data with metadata."""
import pandas as pd

class MLDataFrame(pd.DataFrame):
    """Data frame holding multidimensional data with corresponding metadata.

    Args:
        data: any data input acceptable by pd.DataFrame
        x_data (np.ndarray): input data array, samples indexed along first dim.
        y_data (np.ndarray): target data array, indexed along first dimension.
        kwargs: additional keyword arguments accepted by pd.DataFrame.

    Attributes:
        x_data: machine learning inputs.
        y_data: machine learning targets.

    """
    _metadata = ["_x_data", "_y_data"]

    @property
    def _constructor(self):
        return MLDataFrame

    def __init__(self, data, x_data=None, y_data=None, **kwargs):
        if kwargs.pop("index", None):
            raise ValueError("Cannot set a custom index on MLDataFrame.")
        super().__init__(data, **kwargs)

        self._x_data = x_data
        self._y_data = y_data

    @property
    def x_data(self):
        """Machine learning inputs."""
        if self._x_data is None:
            raise ValueError("Cannot access unspecified x_data")
        return self._x_data[self.index]

    @property
    def y_data(self):
        """Machine Learning targets."""
        if self._y_data is None:
            raise ValueError("Cannot access unspecified y_data")
        return self._y_data[self.index]
