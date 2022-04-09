# main.py
from PyQt5.QtWidgets import QApplication

from mainwindow import MainWindow
from rppg import RPPG, get_heartbeat_filter

if __name__ == "__main__":
    app = QApplication([])
    live_bandpass = get_heartbeat_filter(order=4, cutoff=[0.5, 2.5], fs=30,
                                         btype="bandpass", output="ba")
    rppg = RPPG(video=0, parent=app, filter_function=live_bandpass)
    win = MainWindow(rppg=rppg)
    win.show()

    rppg.start()
    app.exec_()
    rppg.stop()
