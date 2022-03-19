# camera.py

import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class Camera(QThread):
    """Wraps cv2.VideoCapture and emits Qt signal with frames in RGB format.

    The `run` function launches a loop that waits for new frames in the
    VideoCapture and emits them with a `frame_received` signal.  Calling
    `stop` stops the loop and releases the camera.
    """
    frame_received = pyqtSignal(np.ndarray)
    """PyQt Signal emitting new frames read from the camera.
    """

    def __init__(self, video=0, parent=None):
        """Initialize Camera instance.

        Args:
            video (int or string): ID of camera or video filename
            parent (QObject): parent object in Qt context
        """
        super().__init__(parent=parent)

        self._cap = cv2.VideoCapture(video)
        self._running = False
        # self.img = cv2.imread("face_example1.jpg")

    def run(self):
        """Start loop in thread capturing incoming frames.
        """
        self._running = True
        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                self._running = False
                raise RuntimeError("No frame received")
            # self.frame_received.emit(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            self.frame_received.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self):
        """Stop loop and release camera.
        """
        self._running = False
        time.sleep(0.1)
        self._cap.release()
