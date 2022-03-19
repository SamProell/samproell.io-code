# detector.py

import cv2
import mediapipe as mp
import numpy as np

def get_facemesh_coords(landmark_list, img):
    """Extract FaceMesh landmark coordinates into 468x3 NumPy array.
    """
    h, w = img.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]

    return np.multiply(xyz, [w, h, w]).astype(int)

def fill_roimask(point_list, img):
    """Create binary mask, filled inside contour given by list of points.
    """
    mask = np.zeros(img.shape[:2], dtype="uint8")
    if len(point_list) > 2:
        contours = np.reshape(point_list, (1, -1, 1, 2))  # expected by OpenCV
        cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)
    return mask

class ROIDetector:
    """Identifies lower face as region of interest.
    """
    _lower_face = [200, 431, 411, 340, 349, 120, 111, 187, 211]  # mesh indices

    def __init__(self):
        """Initialize detector (Mediapipe FaceMesh).
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        """Find single face in frame and extract lower half of the face.
        """
        results = self.face_mesh.process(frame)

        point_list = []
        if results.multi_face_landmarks is not None:
            coords = get_facemesh_coords(results.multi_face_landmarks[0], frame)
            point_list = coords[self._lower_face, :2]  # :2 -> only x and y
        roimask = fill_roimask(point_list, frame)

        return roimask, results

    def close(self):
        """Finish up (close Face Mesh instance).
        """
        self.face_mesh.close()
