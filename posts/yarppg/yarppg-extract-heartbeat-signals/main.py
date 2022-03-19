# main.py
from PyQt5.QtWidgets import QApplication

from mainwindow import MainWindow
from rppg import RPPG

if __name__ == "__main__":
    app = QApplication([])
    rppg = RPPG(video=0, parent=app)
    win = MainWindow(rppg=rppg)
    win.show()

    rppg.start()
    app.exec_()
    rppg.stop()
