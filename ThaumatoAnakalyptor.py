### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QSplitter, QVBoxLayout, 
                             QGroupBox, QWidget, QPushButton, QLabel, QToolButton, QHBoxLayout, QFrame, QTextEdit,
                             QFileDialog, QLineEdit, QCheckBox, QDialog, QSpinBox, QGraphicsView, QGraphicsScene, QMessageBox, QStyle)
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QPen, QBrush
from PIL import Image
import os

from tqdm import tqdm
import time

def showHelp(self):
        helpText = "ThaumatoAnakalyptor Help\n\n" \
                   "There are the following shortcuts:\n\n" \
                   "- Use 'A' and 'D' to switch between TIFF layers.\n" \
                   "- Click on the TIFF to place a point.\n" 
        QMessageBox.information(self, "Help", helpText)

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
        helpMenu.triggered.connect(lambda: showHelp(self))  # Connect the triggered signal to showHelp method
        menubar.addAction(helpMenu)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Graphics View for Image Display
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
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
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            # Set the index box text
            self.indexBox.setText(str(index))

            # Draw a red point if it exists for this image
            if index in self.points:
                x, y = self.points[index]
                # Convert to scene coordinates
                sceneX = x / self.image_width * self.image_width
                sceneY = y / self.image_height * self.image_height
                self.scene.addEllipse(sceneX, sceneY, 10, 10, QPen(Qt.red), QBrush(Qt.red))

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
            self.incrementIndex()
        elif event.key() == Qt.Key_A:
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
            umbilicus_path_grids = os.path.join(self.imagePath + "_grids", umbilicus_name)
            print(umbilicus_path_grids)
            with open(umbilicus_path_grids, "w") as file:
                for index, (x, y) in self.points.items():
                    file.write(f"{y + 500}, {index + 500}, {x + 500}\n")

            umbilicus_path = os.path.join(self.imagePath, umbilicus_name)
            print(umbilicus_path)
            with open(umbilicus_path, "w") as file:
                for index, (x, y) in self.points.items():
                    file.write(f"{y + 500}, {index + 500}, {x + 500}\n")
            print("Points saved to umbilicus.txt")
        except Exception as e:
            print(e)
            # Error popup
            QMessageBox.critical(self, "Error", "Could not save points to umbilicus.txt. \n\nMake sure to have the 8um 2D TIFF folder <name> and the 8um 3D Grid Cells folder <name>_grids in the same directory.")

    def resizeEvent(self, event):
        if self.scene.itemsBoundingRect().width() > 0:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

class CollapsibleBox(QGroupBox):
    def __init__(self, title):
        super().__init__()
        self.initUI(title)

    def initUI(self, title):
        self.setFlat(True)  # Remove the box border

        # Toggle Button
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; text-align: left; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.clicked.connect(self.on_toggled)
        self.toggle_button.setCheckable(True)

        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setAlignment(Qt.AlignTop)
        
        # Main layout with button
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toggle_button)
        main_layout.addLayout(self.content_layout)

    def on_toggled(self):
        if self.toggle_button.isChecked():
            self.toggle_button.setArrowType(Qt.UpArrow)
        else:
            self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_content_visibility(self.toggle_button.isChecked())

    def toggle_content_visibility(self, visible):
        for i in range(self.content_layout.count()):
            widget = self.content_layout.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(visible)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
        widget.setVisible(False)

class ThaumatoAnakalyptor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ThaumatoAnakalyptor')

        # Menu Bar
        menubar = self.menuBar()
        settings = QAction('Settings', self)
        helpMenu = QAction('Help', self)
        helpMenu.triggered.connect(lambda: showHelp(self))  # Connect the triggered signal to showHelp method
        menubar.addAction(settings)
        menubar.addAction(helpMenu)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left Panel for Image
        left_panel = QLabel()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        pixmap = QPixmap(400, 400)
        pixmap.fill(Qt.white)  # White background
        left_panel.setPixmap(pixmap)

        # Right Panel with sections
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)  # Align sections at the top

        # Collapsible Sections
        self.addVolumeProcessing(right_layout)
        self.addCollapsibleSection(right_layout, "Mesh Generation")
        self.addCollapsibleSection(right_layout, "Rendering")

        right_panel.setLayout(right_layout)

        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

    def addCollapsibleSection(self, layout, title):
        box = CollapsibleBox(title)
        button1 = QPushButton('Button 1')
        button1.clicked.connect(self.printHi)  # Connect the button click to printHi method

        button2 = QPushButton('Button 2')
        button2.clicked.connect(self.doTQDM)

        button3 = QPushButton('Button 3')

        box.add_widget(button1)
        box.add_widget(button2)
        box.add_widget(button3)

        layout.addWidget(box)

    def printHi(self):
        print("Hi")

    def doTQDM(self):
        for i in tqdm(range(10)):
            time.sleep(1)

    def addVolumeProcessing(self, layout):
        # Main Collapsible Box for Volume Preprocessing
        volumeBox = CollapsibleBox("Volume Preprocessing")

        # Generate Grid Cells Area
        generateGridBox = CollapsibleBox("Generate Grid Cells")
        self.addGenerateGridCellsArea(generateGridBox)
        volumeBox.add_widget(generateGridBox)

        # Umbilicus Area
        umbilicusBox = CollapsibleBox("Umbilicus")
        self.addUmbilicusArea(umbilicusBox)
        volumeBox.add_widget(umbilicusBox)

        # Pointcloud Area
        pointcloudBox = CollapsibleBox("Pointcloud")
        self.addPointcloudArea(pointcloudBox)
        volumeBox.add_widget(pointcloudBox)

        layout.addWidget(volumeBox)

    def addUmbilicusArea(self, box):
        label = QLabel("Umbilicus")
        # Create an info button
        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(self.showUmbilicusInfo)        
        self.tiffVolumePath = QLineEdit()
        self.tiffVolumePath.setPlaceholderText("Input Path")
        # Set standard path for development
        # self.tiffVolumePath.setText("/media/julian/FastSSD/PHerc0051Cr04Fr08.volpkg/volumes/20231121152933") # Development
        self.tiffVolumePath.mousePressEvent = lambda event: self.selectPath(self.tiffVolumePath)

        button = QPushButton("Generate")
        button.clicked.connect(self.openUmbilicusWindow)
        box.add_widget(label)
        box.add_widget(infoButton)
        box.add_widget(self.tiffVolumePath)
        box.add_widget(button)

    def showUmbilicusInfo(self):
        QMessageBox.information(self, "Umbilicus Information",
                                "Select the 8um downsampled 2D TIFF files to construct the umbilicus. \n\nMake sure to have the 8um 2D TIFF folder <name> and the 8um 3D Grid Cells folder <name>_grids in the same directory.")

    def openUmbilicusWindow(self):
        if os.path.exists(self.tiffVolumePath.text()):
            self.umbilicusWindow = UmbilicusWindow(self.tiffVolumePath.text())
            self.umbilicusWindow.show()

    def addGenerateGridCellsArea(self, box):
        label = QLabel("Generate Grid Cells")
        scaleFactor = QLineEdit()
        scaleFactor.setPlaceholderText("Scale Factor")

        inputPath = QLineEdit()
        inputPath.setPlaceholderText("Input Path")
        inputPath.mousePressEvent = lambda event: self.selectPath(inputPath)

        outputPath = QLineEdit()
        outputPath.setPlaceholderText("Output Path")
        outputPath.mousePressEvent = lambda event: self.selectPath(outputPath)

        computeButton = QPushButton("Compute")

        box.add_widget(label)
        box.add_widget(scaleFactor)
        box.add_widget(inputPath)
        box.add_widget(outputPath)
        box.add_widget(computeButton)

    def addPointcloudArea(self, box):
        label = QLabel("Pointcloud")
        basePath = QLineEdit()
        basePath.setPlaceholderText("Base Path")
        basePath.mousePressEvent = lambda event: self.selectPath(basePath)

        recomputeCheckbox = QCheckBox("Recompute")
        computeButton = QPushButton("Compute")

        box.add_widget(label)
        box.add_widget(basePath)
        box.add_widget(recomputeCheckbox)
        box.add_widget(computeButton)

    def selectPath(self, lineEdit):
        # Placeholder function for path selection
        # Implement as required
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        lineEdit.setText(path)

def main():
    app = QApplication(sys.argv)
    ex = ThaumatoAnakalyptor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
