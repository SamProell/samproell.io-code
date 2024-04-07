# %%
import os
import pathlib

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold
import sklearn.metrics
import tensorflow as tf
import tqdm
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer
from mediapipe_model_maker.python.vision import gesture_recognizer

import utils

tf.config.set_visible_devices([], "GPU")
# %%
data_root = pathlib.Path(os.environ.get("DATA_ROOT", "./data"))
dataset_train = data_root / "SigNN_train100"
dataset_test = data_root / "SigNN_test"
print(dataset_train.is_dir(), dataset_train)

labels = [p.name for p in dataset_train.iterdir()]
print(len(labels), labels)

trainfiles = utils.find_images(dataset_train)
testfiles = utils.find_images(dataset_test)
print(len(trainfiles), len(testfiles), trainfiles[:3])
# %%
sample_files = np.random.choice(np.asarray(trainfiles), 10)
fig, axarr = utils.plot_image_files(sample_files, ncols=5)
plt.show()
fig.savefig("training-examples.jpg", dpi=150, bbox_inches="tight")
# %%


def load_cached_dataset(path, data_root, params=None, labelfile=None):
    labelfile = pathlib.Path(labelfile or pathlib.Path(path) / "labels.txt")
    try:
        dataset = tf.data.Dataset.load(path)
        labels = labelfile.read_text().split()
        data = gesture_recognizer.Dataset(
            dataset, labels, size=int(dataset.cardinality().numpy())
        )
    except tf.errors.NotFoundError:
        print("Dataset not found. Creating new dataset.")
        data = gesture_recognizer.Dataset.from_folder(str(data_root), params)
        data.gen_tf_dataset().unbatch().save(path)
        labelfile.write_text("\n".join(data.label_names))
    return data


# %%
handparams = gesture_recognizer.HandDataPreprocessingParams(
    min_detection_confidence=0.5
)
data = load_cached_dataset(
    "data/processed/" + dataset_train.name, dataset_train, handparams
)
# The next call takes a long time with the full dataset (10+ minutes for me)
# data = gesture_recognizer.Dataset.from_folder(str(dataset_train), handparams)
# data.gen_tf_dataset().unbatch().save("dataset_train")
# pathlib.Path("dataset_train_labels.txt").write_text("\n".join(data.label_names))
# data = gesture_recognizer.Dataset(
#     tf.data.Dataset.load("dataset_train"),
#     pathlib.Path("dataset_train_labels.txt").read_text().split(),
#     size=100,
# )
# %%
test_data = load_cached_dataset(
    "data/processed/" + dataset_test.name, dataset_test, handparams
)
# test_data = gesture_recognizer.Dataset.from_folder(
#     str(dataset_test), handparams
# )
# %%
train_data, validation_data = data.split(0.8)
# %%
hparams = gesture_recognizer.HParams(
    export_dir="exported_model",
    batch_size=32,
    epochs=30,
    shuffle=True,
    learning_rate=0.005,
)
moptions = gesture_recognizer.ModelOptions(dropout_rate=0.05)  # , layer_widths=[32])
options = gesture_recognizer.GestureRecognizerOptions(
    hparams=hparams, model_options=moptions
)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data, validation_data=validation_data, options=options
)
# %%
loss, acc = model.evaluate(test_data, batch_size=32)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.2%}")
# %%
model.export_model("asl_recognizer.task")
# %%
base_options = mp.tasks.BaseOptions(
    model_asset_path=hparams.export_dir + "/asl_recognizer.task"
)
options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE
)
# %%
test_samples = np.random.choice(np.asarray(testfiles), 10)
with GestureRecognizer.create_from_options(options) as recognizer:
    fig, axarr = utils.plot_recognizer_predictions(test_samples, recognizer, 5)
    fig.savefig("example-output.jpg", bbox_inches="tight", dpi=150)
# %%
test_results = []
with GestureRecognizer.create_from_options(options) as recognizer:
    for filename in tqdm.tqdm(testfiles):
        mp_image = mp.Image.create_from_file(str(filename))
        result = recognizer.recognize(mp_image)
        if len(result.gestures) > 0:
            pred = result.gestures[0][0].category_name or "n/a"
        else:
            pred = "empty"
        test_results.append((filename, filename.parent.name, pred))
# %%
results_df = pd.DataFrame(test_results, columns=["filename", "label", "pred"])
# %%
classes = sorted(test_data.label_names + ["n/a", "empty"])
cm = sklearn.metrics.confusion_matrix(
    results_df["label"], results_df["pred"], labels=classes, normalize="true"
)
# cm[cm < 0.5] = np.nan
sklearn.metrics.ConfusionMatrixDisplay(cm * 100, display_labels=classes).plot(
    include_values=False
)
plt.savefig("cm-test.png", bbox_inches="tight", dpi=150)
# %%

results_df["result"] = np.where(
    results_df.pred == results_df.label,
    "correct",
    np.where(results_df.pred.isin(["n/a", "empty"]), "not found", "incorrect"),
)
sns.histplot(
    data=results_df,
    x="label",
    hue="result",
    multiple="stack",
    stat="count",
    hue_order=["not found", "incorrect", "correct"],
)
plt.savefig("accuracy-per-class.png", dpi=150, bbox_inches="tight")
# %%
results_df.result.value_counts(normalize=True)
results_df["incorrect"] = results_df.result == "incorrect"
print(results_df.groupby("label").incorrect.mean().sort_values(ascending=False))
print(
    results_df.query("result == 'incorrect'")
    .groupby("label")
    .pred.value_counts()
    .sort_values(ascending=False)
)
# %%
samples = np.random.choice(
    np.asarray([f for f in testfiles if f.parent.name == "P"]), 10
)
with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
    utils.plot_recognizer_predictions(samples, recognizer, 5)
plt.show()
# %%
assert train_data.size is not None
train_ds = train_data.gen_tf_dataset(batch_size=train_data.size)
xy = train_ds.take(1).get_single_element()
# %%
embeddings, classes_onehot = xy[0].numpy(), xy[1].numpy()  # type: ignore
class_indices = np.argmax(classes_onehot, axis=1)
print(embeddings.shape, class_indices.shape)
# %%
tsne = sklearn.manifold.TSNE()
emb = tsne.fit_transform(embeddings)
# %%
embdf = pd.DataFrame(emb, columns=["X1", "X2"]).assign(label=class_indices)
sns.scatterplot(
    data=embdf, x="X1", y="X2", hue="label", palette="Spectral", legend=False
)
plt.axis("off")
for i, c in enumerate(train_data.label_names):
    if np.all(class_indices != i):
        continue
    center = emb[class_indices == i].mean(axis=0)
    bbox = {"boxstyle": "round", "fc": "#CCCCCC44", "ec": "r", "lw": 1}
    plt.annotate(
        c, center, center - 6, bbox=bbox, fontweight="semibold", color="#323232"
    )
plt.savefig("gesture-embeddings.png", dpi=120, bbox_inches="tight")
