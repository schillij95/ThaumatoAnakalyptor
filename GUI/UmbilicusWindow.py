### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QMainWindow, QAction, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QHBoxLayout,
                             QLineEdit, QGraphicsView, QGraphicsScene, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QPen, QBrush
from PIL import Image
import os

class GraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)

    def wheelEvent(self, event):
        factor = 1.1  # Zoom factor
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

class UmbilicusWindow(QMainWindow):
    def __init__(self, imagePath, parent=None):
        super().__init__(parent)
        self.imagePath = imagePath
        self.currentIndex = 0
        self.incrementing=True
        self.images = sorted([f for f in os.listdir(self.imagePath) if f.endswith('.tif')])
        self.points = {}  # Dictionary to store points as {index: (x, y)}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Generate Umbilicus")

        # Menu Bar
        menubar = self.menuBar()
        helpMenu = QAction('Help', self)
        helpMenu.triggered.connect(self.showHelp)  # Connect the triggered signal to showHelp method
        menubar.addAction(helpMenu)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Graphics View for Image Display
        self.scene = QGraphicsScene(self)
        self.view = GraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        # Set the focus policy to accept key events
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.setMouseTracking(True)  # Enable mouse tracking if needed

        self.view.mousePressEvent = self.viewMousePressEvent
        layout.addWidget(self.view)

        # Step Size and Index Input Area
        stepSizeLayout = QHBoxLayout()
        self.fileNameLabel = QLabel("File: " + self.images[self.currentIndex])
        stepSizeLabel = QLabel("Step Size:")
        self.stepSizeBox = QLineEdit("333")  # Set default value
        self.stepSizeBox.setFixedWidth(100)  # Adjust width as appropriate
        self.stepSizeBox.returnPressed.connect(lambda: self.view.setFocus())  # Unfocus on Enter

        indexLabel = QLabel("Index:")
        self.indexBox = QLineEdit()
        self.indexBox.setFixedWidth(100)  # Adjust width as appropriate
        self.indexBox.returnPressed.connect(self.jumpToIndex)  # Jump to index on Enter
        self.indexBox.setText(str(self.currentIndex))

        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.loadPoints)
        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.savePoints)

        stepSizeLayout.addWidget(self.fileNameLabel)
        stepSizeLayout.addWidget(stepSizeLabel)
        stepSizeLayout.addWidget(self.stepSizeBox)
        stepSizeLayout.addWidget(indexLabel)
        stepSizeLayout.addWidget(self.indexBox)
        stepSizeLayout.addWidget(loadButton)
        stepSizeLayout.addWidget(saveButton)
        layout.addLayout(stepSizeLayout)

        # Load the first image
        self.loadImage(self.currentIndex)

        # Set focus to the QGraphicsView
        self.view.setFocus()

        # Load the first image
        self.loadImage(self.currentIndex)

    def showHelp(self):
        helpText = "ThaumatoAnakalyptor Help\n\n" \
                   "If you already have an umbilicus generated, load it with 'Load'. \n" \
                   "Place the umbilicus points in the center of the scroll and when done press 'Save' before closing the window.\n\n" \
                   "There are the following shortcuts:\n\n" \
                   "- Use Mouse Wheel to zoom in and out.\n" \
                   "- Use 'A' and 'D' to switch between TIFF layers.\n" \
                   "- Use 'Ctrl + A' and 'Ctrl + D' to switch between TIFF layers with umbilicus points.\n" \
                   "- Click on the TIFF to place a point.\n" \
                   "- Use Ctrl + Click to automatically switch to the next TIFF.\n"
        QMessageBox.information(self, "Help", helpText)

    def loadImage(self, index):
        if 0 <= index < len(self.images):
            imagePath = os.path.join(self.imagePath, self.images[index])
            self.fileNameLabel.setText("File: " + self.images[index])
            image = Image.open(imagePath)

            self.image_width = image.size[0]
            self.image_height = image.size[1]

            # Convert to 8-bit grayscale
            image8bit = image.point(lambda i: i * (1./256)).convert('L')
            qimage = QImage(image8bit.tobytes(), image8bit.size[0], image8bit.size[1], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)

            # Clear the previous items in the scene
            self.scene.clear()

            # Add new pixmap item to the scene
            self.scene.addPixmap(pixmap)
            # self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            # Set the index box text
            self.indexBox.setText(str(index))

            # Draw a red point if it exists for this image
            if index in self.points:
                x, y = self.points[index]
                # Convert to scene coordinates
                sceneX = x / self.image_width * self.image_width
                sceneY = y / self.image_height * self.image_height
                # Size of point unchanged in display no matter the zoom
                size_display = 10
                # Size of point in image coordinates
                size_image = size_display / self.view.transform().m11()
                self.scene.addEllipse(sceneX, sceneY, size_image, size_image, QPen(Qt.red), QBrush(Qt.red))

    def jumpToIndex(self):
        index = int(self.indexBox.text())
        if 0 <= index < len(self.images):
            self.currentIndex = index
            self.loadImage(self.currentIndex)
        # Unfocus the index box
        self.view.setFocus()

    def incrementIndex(self):
        self.incrementing = True
        step_size = int(self.stepSizeBox.text())
        self.currentIndex = min((self.currentIndex + step_size), len(self.images) - 1)

    def decrementIndex(self):
        self.incrementing = False
        step_size = int(self.stepSizeBox.text())
        self.currentIndex = max((self.currentIndex - step_size), 0)

    def keyPressEvent(self, event: QKeyEvent):
        # "A" "D" keys for navigation
        if event.key() == Qt.Key_D:
            if event.modifiers() == Qt.ControlModifier:
                # find next image with umbilicus
                next_index = None
                for key in self.points.keys():
                    if key > self.currentIndex and (next_index is None or key < next_index):
                        next_index = key
                if next_index is not None:
                    self.currentIndex = next_index
            else:
                self.incrementIndex()
        elif event.key() == Qt.Key_A:
            if event.modifiers() == Qt.ControlModifier:
                # find previous image with umbilicus
                prev_index = None
                for key in self.points.keys():
                    if key < self.currentIndex and (prev_index is None or key > prev_index):
                        prev_index = key
                if prev_index is not None:
                    self.currentIndex = prev_index
            else:
                self.decrementIndex()
        self.loadImage(self.currentIndex)

    def viewMousePressEvent(self, event):
        if self.scene.itemsBoundingRect().width() > 0:
            scenePos = self.view.mapToScene(event.pos())
            imgRect = self.scene.itemsBoundingRect()
            # Calculate the proportional coordinates
            originalX = int((scenePos.x() / imgRect.width()) * self.image_width)
            originalY = int((scenePos.y() / imgRect.height()) * self.image_height)
            self.points[self.currentIndex] = (originalX, originalY)

            if event.modifiers() == Qt.ControlModifier:
                # Go to next image
                if self.incrementing:
                    self.incrementIndex()
                else:
                    self.decrementIndex()
            self.loadImage(self.currentIndex)

    def loadPoints(self):
        self.points = {}
        umbilicus_name = "umbilicus.txt"
        umbilicus_path = os.path.join(self.imagePath, umbilicus_name)
        if os.path.exists(umbilicus_path):
            with open(umbilicus_path, "r") as file:
                for line in file:
                    y, index, x = map(int, line.strip().split(', '))
                    self.points[index-500] = (x-500, y-500)
            self.loadImage(self.currentIndex)
            print("Points loaded from umbilicus.txt")


    def savePoints(self):
        umbilicus_name = "umbilicus.txt"
        try:
            umbilicus_path = os.path.join(self.imagePath, umbilicus_name)
            print(umbilicus_path)
            with open(umbilicus_path, "w") as file:
                for index, (x, y) in self.points.items():
                    file.write(f"{y + 500}, {index + 500}, {x + 500}\n")

            umbilicus_path = os.path.join(self.imagePath + "_grids", umbilicus_name)
            print(umbilicus_path)
            with open(umbilicus_path, "w") as file:
                for index, (x, y) in self.points.items():
                    file.write(f"{y + 500}, {index + 500}, {x + 500}\n")
            print("Points saved to umbilicus.txt")
        except Exception as e:
            print(e)
            # Error popup
            QMessageBox.critical(self, "Error", "Could not save points to umbilicus.txt.")

    def resizeEvent(self, event):
        if self.scene.itemsBoundingRect().width() > 0:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
