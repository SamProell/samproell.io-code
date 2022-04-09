# rppg.py

from collections import namedtuple

import cv2
import scipy.signal
from PyQt5.QtCore import QObject, pyqtSignal

from camera import Camera
from detector import ROIDetector
from digitalfilter import LiveLFilter, LiveSosFilter

RppgResults = namedtuple("RppgResults", ["rawimg",
                                         "roimask",
                                         "landmarks",
                                         "signal",
                                         ])


class RPPG(QObject):

    rppg_updated = pyqtSignal(RppgResults)

    def __init__(self, parent=None, video=0, filter_function=None):
        """rPPG model processing incoming frames and emitting calculation
        outputs.

        The signal RPPG.updated provides a named tuple RppgResults containing
          - rawimg: the raw frame from camera
          - roimask: binary mask filled inside the region of interest
          - landmarks: multiface_landmarks object returned by FaceMesh
          - signal: reference to a list containing the signal
        """
        super().__init__(parent=parent)

        self._cam = Camera(video=video, parent=parent)
        self._cam.frame_received.connect(self.on_frame_received)

        self.detector = ROIDetector()

        if filter_function is None:
            self.filter_function = lambda x: x  # pass values unfiltered
        else:
            self.filter_function = filter_function

        self.signal = []

    def on_frame_received(self, frame):
        """Process new frame - find face mesh and extract pulse signal.
        """
        rawimg = frame.copy()
        roimask, results = self.detector.process(frame)

        r, g, b, a = cv2.mean(rawimg, mask=roimask)
        self.signal.append(self.filter_function(g))

        self.rppg_updated.emit(RppgResults(rawimg=rawimg,
                                           roimask=roimask,
                                           landmarks=results,
                                           signal=self.signal))

    def start(self):
        """Launch the camera thread.
        """
        self._cam.start()

    def stop(self):
        """Stop the camera thread and clean up the detector.
        """
        self._cam.stop()
        self.detector.close()


def get_heartbeat_filter(order=4, cutoff=[0.5, 2.5], btype="bandpass", fs=30,
                         output="ba"):
    """Create live filter with lfilter or sosfilter implmementation.
    """
    coeffs = scipy.signal.iirfilter(order, Wn=cutoff, fs=fs, btype=btype,
                                    ftype="butter", output=output)

    if output == "ba":
        return LiveLFilter(*coeffs)
    elif output == "sos":
        return LiveSosFilter(coeffs)
    raise NotImplementedError(f"Unknown output {output!r}")
