"""Example code snippets from the following post.

https://www.samproell.io/posts/datascience/metadata-in-machinelearning

The article highlights how it can be cumbersome to deal with additional
information (metadata) about machine learning data. Typically, ML problems
rarely come as uniform packs of data in `x_data`+`y_data` form.
In the constructed example below, we have a release date and design season for
each piece of clothing in the Fashion-MNIST dataset.

Taking such information into account can improve model performance and avoid
unintended bias.
"""
# pylint: disable=wrong-import-position
# %% [markdown]
# ## Examples from tutorials and guides
# %%
# https://keras.io/api/datasets/fashion_mnist/
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,) and y_test.shape == (10000,)

# https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset
import sklearn.datasets

data = sklearn.datasets.fetch_olivetti_faces()
x_data, y_data = data["images"], data["target"]
assert x_data.shape == (400, 64, 64) and y_data.shape == (400,)

# %% [markdown]
# ## Create random metadata to illustrate real-world scenario
# %%
# x_train/test, y_train/test loaded as before through Keras datasets.
import numpy as np

year_train = np.random.choice(np.arange(2015, 2023, dtype=int), len(y_train))
year_test = np.random.choice(np.arange(2015, 2023, dtype=int), len(y_test))

seasons = "Spring Summer Autumn Winter".split()
season_train = np.random.choice(seasons, len(y_train))
season_test = np.random.choice(seasons, len(y_test))

# %% [markdown]
# ## Implement and fit a very basic Keras model
# %%
import tensorflow as tf

def get_compiled_model():
    """Get simple Multilayer Perceptron network, compiled with Adam and MSE."""
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    new_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return new_model

model = get_compiled_model()

# %%
# First, one-hot encode targets.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train_after_2020 = x_train[year_train > 2020]
y_train_after_2020 = y_train[year_train > 2020]

# define and compile your Keras model
# ...

model.fit(x_train_after_2020, y_train_after_2020)

# %% [markdown]
# ### Evaluate on filtered testing data.
# %%
x_test_after_2020 = x_test[year_test > 2020]
y_test_after_2020 = y_test[year_test > 2020]

print(model.metrics_names)
print(model.evaluate(x_test_after_2020, y_test_after_2020))

# %% [markdown]
# ### Evaluate on grouped testing data
# %%
data_filter = year_test > 2020
for season in seasons:
    season_filter = data_filter & (season_test == season)
    print(season, model.evaluate(x_test[season_filter], y_test[season_filter]))

# %% [markdown]
# # Potential solutions for better metadata handling
# ## Pandas data frame with metadata and exlicit indexing
# Create a data frame holding metadata, use the default numeric index to subset
# x and y data.
# %%
import pandas as pd
data_train = pd.DataFrame({
    "year": year_train, "season": season_train,
    "target": np.argmax(y_train, axis=1)
})
data_test = pd.DataFrame({
    "year": year_test, "season": season_test,
    "target": np.argmax(y_test, axis=1)
})

assert np.array_equal(x_train, x_train[data_train.index])

data_after2020 = data_train.query("year > 2020")
model.fit(x_train[data_after2020.index], y_train[data_after2020.index])

# %%
for season, df in data_test.query("year > 2020").groupby("season"):
    print(season, model.evaluate(x_test[df.index], y_test[df.index]))

# %% [markdown]
# ## Subclassing pandas dataframes: MLDataFrame
# %%
from mldataframe import MLDataFrame

mldf = MLDataFrame(
    {"year": year_train, "season": season_train,
     "target": np.argmax(y_train, axis=1)},
    x_data=x_train,
    y_data=y_train,
)

assert np.array_equal(mldf.query("year > 2020").x_data, x_train_after_2020)
assert np.array_equal(mldf.query("year > 2020").y_data, y_train_after_2020)

winter_df = mldf.query("season == 'Winter'")
model = get_compiled_model()  # a new untrained model for demonstration
model.fit(winter_df.x_data, winter_df.y_data)

# %%
mldf_test = MLDataFrame(
    {"year": year_test, "season": season_test,
     "target": np.argmax(y_test, axis=1)},
    x_data=x_test,
    y_data=y_test,
)

for season, season_df in mldf_test.query("year > 2020").groupby("season"):
    print(season, model.evaluate(season_df.x_data, season_df.y_data))

# %%  MLDataFrame also works with train_test_split
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(mldf, test_size=0.3)
print(f"Training split: {train_df.x_data.shape} -> {train_df.y_data.shape}")
print(f"Validation split: {val_df.x_data.shape} -> {val_df.y_data.shape}")

# %%  MLDataFrame holds a reference to the original data
# _x_data references corresponding array
assert mldf._x_data is x_train  # pylint: disable=protected-access
assert mldf._x_data is winter_df._x_data  # pylint: disable=protected-access

# but: x_data returns a copy
assert mldf.x_data is not x_train
assert winter_df.x_data is not x_train  # obviously - not even the same shape
