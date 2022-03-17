# mainwindow.py

from PyQt5.QtWidgets import QMainWindow
import pyqtgraph as pg
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class MainWindow(QMainWindow):
    def __init__(self, rppg):
        """MainWindow visualizing the output of the RPPG model.
        """
        super().__init__()

        rppg.rppg_updated.connect(self.on_rppg_updated)
        self.init_ui()

    def on_rppg_updated(self, output):
        """Update UI based on RppgResults.
        """
        img = output.rawimg.copy()
        draw_facemesh(img, output.landmarks, tesselate=True, contour=True)
        self.img.setImage(img)


    def init_ui(self):
        """Initialize window with pyqtgraph image view box in the center.
        """
        self.setWindowTitle("FaceMesh detection in PyQt")

        layout = pg.GraphicsLayoutWidget()
        self.img = pg.ImageItem(axisOrder="row-major")
        vb = layout.addViewBox(invertX=True, invertY=True, lockAspect=True)
        vb.addItem(self.img)

        self.setCentralWidget(layout)


def draw_facemesh(img, results, tesselate=False,
                  contour=False, irises=False):
    """Draw all facemesh landmarks found in an image.

    Irises are only drawn if the corresponding landmarks are present,
    which requires FaceMesh to be initialized with refine=True.
    """
    if results is None or results.multi_face_landmarks is None:
        return

    for face_landmarks in results.multi_face_landmarks:
        if tesselate:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        if contour:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
        if irises and len(face_landmarks) > 468:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
