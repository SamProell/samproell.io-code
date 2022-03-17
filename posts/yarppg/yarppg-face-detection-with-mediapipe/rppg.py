# rppg.py

from collections import namedtuple
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
import mediapipe as mp

from camera import Camera

RppgResults = namedtuple("RppgResults", ["rawimg", "landmarks"])

class RPPG(QObject):

    rppg_updated = pyqtSignal(RppgResults)

    def __init__(self, parent=None, video=0):
        """rPPG model processing incoming frames and emitting calculation
        outputs.

        The signal RPPG.updated provides a named tuple RppgResults containing
          - rawimg: the raw frame from camera
          - landmarks: multiface_landmarks object returned by FaceMesh
        """
        super().__init__(parent=parent)

        self._cam = Camera(video=video, parent=parent)
        self._cam.frame_received.connect(self.on_frame_received)

        self.detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def on_frame_received(self, frame):
        """Process new frame - find face mesh and emit outputs.
        """
        rawimg = frame.copy()
        results = self.detector.process(frame)

        self.rppg_updated.emit(RppgResults(rawimg, results))

    def start(self):
        """Launch the camera thread.
        """
        self._cam.start()

    def stop(self):
        """Stop the camera thread and clean up the detector.
        """
        self._cam.stop()
        self.detector.close()
