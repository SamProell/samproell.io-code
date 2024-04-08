"""Some utility functions for working with the ASL detector dataset/model."""

import pathlib

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


def find_images(root: str | pathlib.Path) -> list[pathlib.Path]:
    """Find all JPG and PNG images under the given root directory."""
    return list(pathlib.Path(root).glob("**/*.[jpJP][npNP][egEG]*"))


def _read_img(filename, resize=(224, 224)):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, resize)


def plot_image_files(filenames, ncols=5, resize=(224, 224)):
    img_arrays = [_read_img(str(f), resize) for f in filenames]
    fig, axarr = _plot_image_array(img_arrays, ncols)
    for i, filename in enumerate(filenames):
        axarr[i].set_title(pathlib.Path(filename).parent.name, size="smaller")
    return fig, axarr


def plot_recognizer_predictions(
    filenames, recognizer, ncols=5, landmarks=True, resize=(224, 224)
):
    img_array = [_read_img(str(f), resize) for f in filenames]

    preds = []
    for arr in img_array:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(arr))
        result = recognizer.recognize(img)
        if len(result.gestures) > 0 and len(result.gestures[0]) > 0:
            preds.append(result.gestures[0][0].category_name or "N/A")
            if landmarks:
                draw_hand_landmarks(arr, result.hand_landmarks[0])
        else:
            preds.append("empty")

    fig, axarr = _plot_image_array(img_array, ncols)
    for i, (fname, pred) in enumerate(zip(filenames, preds)):
        axarr[i].set_title(
            f"{pred} (True: {pathlib.Path(fname).parent.name})", size="smaller"
        )
    return fig, axarr


def _plot_image_array(arrays, ncols):
    nrows = int(np.ceil(len(arrays) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axarr = np.reshape(axarr, (-1,))
    for ax, img in zip(axarr, arrays):
        ax.imshow(img)
        ax.axis("off")
    return fig, axarr


def draw_hand_landmarks(img, landmarks):
    # slightly modified from here:
    # https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend(  # type: ignore
        [
            landmark_pb2.NormalizedLandmark(  # type: ignore
                x=landmark.x, y=landmark.y, z=landmark.z
            )
            for landmark in landmarks
        ]
    )
    connections = mp.solutions.hands.HAND_CONNECTIONS  # type: ignore
    lm_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()  # type: ignore
    c_style = mp.solutions.drawing_styles.get_default_hand_connections_style()  # type: ignore
    mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
        img, proto, connections, lm_style, c_style
    )
