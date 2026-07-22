"""Application entry point for the Institut Fresnel event-DSI microscope control GUI.

Run with:  python main.py   (from inside the dsi_microscope/ package directory)
"""

import sys

from PyQt6.QtWidgets import QApplication

from config import STYLESHEET
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    # Maximized rather than true full screen: the window keeps its title bar,
    # so it can still be moved and closed.
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
