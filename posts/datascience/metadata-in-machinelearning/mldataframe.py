import pandas as pd

class MLDataFrame(pd.DataFrame):
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
        if self._x_data is None:
            raise ValueError("Cannot access unspecified x_data")
        return self._x_data[self.index]

    @property
    def y_data(self):
        if self._y_data is None:
            raise ValueError("Cannot access unspecified y_data")
        return self._y_data[self.index]

