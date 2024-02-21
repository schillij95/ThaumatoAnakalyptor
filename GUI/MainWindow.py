### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QMainWindow, QAction, QSplitter, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QFrame,
                             QFileDialog, QLineEdit, QCheckBox, QMessageBox, QStyle, QVBoxLayout, QScrollArea, QHBoxLayout, QGraphicsScene, QGraphicsView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QKeyEvent, QIcon

import tifffile

# Custom GUI Elements
from .CollapsibleBox import CollapsibleBox
from .UmbilicusWindow import UmbilicusWindow
from .ConfigWindow import ConfigWindow

import os

from tqdm import tqdm
import time
import json
import threading
import multiprocessing
# multiprocessing.set_start_method('spawn')
import subprocess

import signal
import numpy as np

from ThaumatoAnakalyptor.sheet_to_mesh import umbilicus_xy_at_z

# import computation functions
from ThaumatoAnakalyptor.generate_half_sized_grid import compute as compute_grid_cells
from ThaumatoAnakalyptor.grid_to_pointcloud import compute as compute_pointcloud
from ThaumatoAnakalyptor.Random_Walks import compute as compute_stitch_sheet

class GraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)

    def wheelEvent(self, event):
        if not event.modifiers() == Qt.ControlModifier:
            # send to parent if Ctrl is not pressed
            super().wheelEvent(event)
            return
        # Ctrl + Wheel to zoom
        factor = 1.1  # Zoom factor
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)


class ThaumatoAnakalyptor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.process = None
        self.isSelectingStartingPoint = False
        self.points = []
        self.loadConfig()
        self.initUI()
        # set icon
        icon = QIcon("GUI/ThaumatoAnakalyptor.png")
        self.setWindowIcon(icon)
        # Set minimum window size
        self.setMinimumSize(800, 600)

    def initUI(self):
        self.setWindowTitle('ThaumatoAnakalyptor')

        # Menu Bar
        menubar = self.menuBar()
        config = QAction('Config', self)
        helpMenu = QAction('Help', self)
        helpMenu.triggered.connect(self.showHelp)  # Connect the triggered signal to showHelp method
        menubar.addAction(config)
        menubar.addAction(helpMenu)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left Panel for Image
        left_panel = QLabel()
        left_panel.setFrameStyle(QFrame.StyledPanel)

        # Check and load TIFF files
        self.tifImages = self.loadTifImages(self.Config.get("downsampled_2d_tiffs", ""))
        self.currentTifIndex = 0

        # Setup left panel with QGraphicsView
        self.tifScene = QGraphicsScene(self)
        self.tifView = GraphicsView(self.tifScene)
        self.tifView.setRenderHint(QPainter.Antialiasing)
        self.tifView.mousePressEvent = self.onTifMousePress

        left_panel_layout = QVBoxLayout()
        left_panel_layout.addWidget(self.tifView)
        self.setupTifNavigation(left_panel_layout)

        left_panel.setLayout(left_panel_layout)

        # Load the first TIFF image
        self.loadTifImage(self.currentTifIndex)
        self.tifView.fitInView(self.tifScene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # Right Panel setup with a Scroll Area
        right_panel_scroll_area = QScrollArea()  # Create a QScrollArea
        right_panel_scroll_area.setWidgetResizable(True)  # Make the scroll area resizable
        right_panel = QWidget()  # This will be the scrollable content
        right_layout = QVBoxLayout(right_panel)  # Use right_panel as the parent for the layout
        right_layout.setAlignment(Qt.AlignTop)  # Align sections at the top
        right_panel_scroll_area.setWidget(right_panel)  # Set the widget you want to scroll

        # Collapsible Sections
        self.addVolumeProcessing(right_layout)
        self.addMeshGeneration(right_layout)
        self.addRendering(right_layout)
        self.addInkDetection(right_layout)

        right_panel.setLayout(right_layout)

        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel_scroll_area)
        # Set splitter ratio
        splitter.setSizes([400, 200])

        # Trigger click on Config
        config.triggered.connect(self.openConfigWindow)


    def loadTifImages(self, path):
        if path and os.path.exists(path):
            return sorted([f for f in os.listdir(path) if f.endswith('.tif')])
        return []

    def setupTifNavigation(self, layout):
        navigationLayout = QHBoxLayout()

        # Add stretch to align content to the right
        navigationLayout.addStretch()

        # Step Size
        navigationLayout.addWidget(QLabel("Step Size:"))
        self.stepSizeBox = QLineEdit("100")
        self.stepSizeBox.setFixedWidth(50)  # Adjust width as needed
        navigationLayout.addWidget(self.stepSizeBox)

        # Index
        navigationLayout.addWidget(QLabel("Index:"))
        self.indexBox = QLineEdit("0")
        self.indexBox.setFixedWidth(50)  # Adjust width as needed
        self.indexBox.returnPressed.connect(self.jumpToTifIndex)
        navigationLayout.addWidget(self.indexBox)

        # Add the navigation layout to the main layout
        layout.addLayout(navigationLayout)

    def loadTifImage(self, index):
        umbilicus_path = os.path.join(self.Config.get("downsampled_2d_tiffs", ""), "umbilicus.txt")
        if os.path.exists(umbilicus_path):
            self.points = []
            if os.path.exists(umbilicus_path):
                with open(umbilicus_path, "r") as file:
                    for line in file:
                        y, z, x = map(int, line.strip().split(', '))
                        self.points.append((x-500, y-500, z-500))

                self.points = np.array(self.points)

        if 0 <= index < len(self.tifImages):
            imagePath = os.path.join(self.Config.get("downsampled_2d_tiffs", ""), self.tifImages[index])

            # Use tifffile to read the TIFF image
            with tifffile.TiffFile(imagePath) as tif:
                image_array = tif.asarray()

            # print(f"Loaded image {imagePath} at index {index} with shape {image_array.shape} and dtype {image_array.dtype}")

            if image_array.dtype == np.uint16:
                image_array = (image_array / 256).astype(np.uint8)

            # Assuming the image is grayscale, prepare it for display
            image_height, image_width = image_array.shape

            self.image_width = image_width
            self.image_height = image_height

            qimage = QImage(image_array.data, image_width, image_height, image_width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)

            # Clear the previous items in the scene
            self.tifScene.clear()

            # Add new pixmap item to the scene
            self.tifScene.addPixmap(pixmap)
            # self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            # Set the index box text
            self.indexBox.setText(str(index))

            # Draw a red point if it exists for this image
            if len(self.points) > 0:
                point_np = umbilicus_xy_at_z(self.points, np.array([index]))
                x, y = point_np[0][0], point_np[0][1]
                # Convert to scene coordinates
                sceneX = x / self.image_width * self.image_width
                sceneY = y / self.image_height * self.image_height
                # Size of point unchanged in display no matter the zoom
                size_display = 10
                # Size of point in image coordinates
                size_image = size_display / self.tifView.transform().m11()
                sceneX -= size_image / 2
                sceneY -= size_image / 2
                self.tifScene.addEllipse(sceneX, sceneY, size_image, size_image, QPen(Qt.red), QBrush(Qt.red))

                # Draw a cone from the point to the bottom of the image with cone angle 45 degrees
                # calculate bottom x point (just bottom of the image)
                cone_line_left_y = self.image_height
                cone_line_right_y = self.image_height
                # calculate 45 degree angle line between scene xy and cone line x to get cone line y
                cone_line_left_x = sceneX + (cone_line_left_y - sceneY) * np.tan(np.deg2rad(45))
                cone_line_right_x = sceneX - (cone_line_right_y - sceneY) * np.tan(np.deg2rad(45))
                # Draw the lines
                self.tifScene.addLine(sceneX, sceneY, cone_line_left_x, cone_line_left_y, QPen(Qt.green))
                self.tifScene.addLine(sceneX, sceneY, cone_line_right_x, cone_line_right_y, QPen(Qt.green))
                self.tifScene.addLine(cone_line_left_x, cone_line_left_y, cone_line_right_x, cone_line_right_y, QPen(Qt.green))

            # Draw green starting point on the z index image
            if hasattr(self, "xField") and hasattr(self, "yField") and hasattr(self, "zField") and self.zField.text() == str(index):
                x = int(self.xField.text())
                y = int(self.yField.text())
                # Convert to scene coordinates
                sceneX = x / self.image_width * self.image_width
                sceneY = y / self.image_height * self.image_height
                # Size of point unchanged in display no matter the zoom
                size_display = 10
                # Size of point in image coordinates
                size_image = size_display / self.tifView.transform().m11()
                sceneX -= size_image / 2
                sceneY -= size_image / 2
                self.tifScene.addEllipse(sceneX, sceneY, size_image, size_image, QPen(Qt.green), QBrush(Qt.green))
            
        else:
            print(f"Index {index} out of range")
            # white placeholder image
            pixmap = QPixmap(400, 400)
            pixmap.fill(Qt.white)  # White background
            self.tifScene.clear()
            self.tifScene.addPixmap(pixmap)


    def jumpToTifIndex(self):
        index = int(self.indexBox.text())
        self.loadTifImage(index)

    def onTifMousePress(self, event):
        # print(f"Clicked position in image coordinates: ({int(scenePos.x())}, {int(scenePos.y())}, {self.currentTifIndex})")
        # Set starting point if in selection mode
        if self.isSelectingStartingPoint:
            scenePos = self.tifView.mapToScene(event.pos())
            
            self.xField.setText(str(int(scenePos.x())))
            self.yField.setText(str(int(scenePos.y())))
            self.zField.setText(str(self.currentTifIndex))

            self.loadTifImage(self.currentTifIndex)

    def incrementIndex(self):
        self.incrementing = True
        step_size = int(self.stepSizeBox.text())
        self.currentTifIndex = min((self.currentTifIndex + step_size), len(self.tifImages) - 1)

    def decrementIndex(self):
        self.incrementing = False
        step_size = int(self.stepSizeBox.text())
        self.currentTifIndex = max((self.currentTifIndex - step_size), 0)

    def keyPressEvent(self, event: QKeyEvent):
        # "A" "D" keys for navigation
        if event.key() == Qt.Key_D:
            self.incrementIndex()
        elif event.key() == Qt.Key_A:
            self.decrementIndex()
        self.loadTifImage(self.currentTifIndex)

    def openConfigWindow(self):
        self.configWindow = ConfigWindow(self)
        self.configWindow.show()

    def loadConfig(self):
        try:
            with open("config.json", "r") as file:
                self.Config = json.load(file)
                # Restore the state of UI components using self.Config
        except FileNotFoundError:
            self.Config = {}

    def showHelp(self):
        helpText = "ThaumatoAnakalyptor Help\n\n" \
                   "There are the following shortcuts:\n\n" \
                   "- Use 'A' and 'D' to switch between TIFF layers.\n" \
                   "- Click on the TIFF to place a point.\n" \
                   "If you run into out-of-memory issues on your GPU, try to reduce the number of threads and the batch size." 
        QMessageBox.information(self, "Help", helpText)

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
        print(self.Config)

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

        # Instances Area
        instancesBox = CollapsibleBox("Instances")
        self.addInstancesArea(instancesBox)
        volumeBox.add_widget(instancesBox)


        layout.addWidget(volumeBox)

    def addUmbilicusArea(self, box):
        label = QLabel("Umbilicus")
        # Create an info button
        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(self.showUmbilicusInfo)        

        button = QPushButton("Generate")
        button.clicked.connect(self.openUmbilicusWindow)
        box.add_widget(label)
        box.add_widget(infoButton)
        box.add_widget(button)

    def showUmbilicusInfo(self):
        QMessageBox.information(self, "Umbilicus Information",
                                "If you already have an umbilicus generated, load it with 'Load'. Place the umbilicus points in the center of the scroll. Save your work with 'Save' before closing. Make sure to have the appropriate paths set in the Config.")

    def openUmbilicusWindow(self):
        if self.Config.get("downsampled_2d_tiffs", None) and os.path.exists(self.Config["downsampled_2d_tiffs"]):
            self.umbilicusWindow = UmbilicusWindow(self.Config["downsampled_2d_tiffs"])
            self.umbilicusWindow.resize(self.width(), self.height())  # Adjust width and height as needed
            self.umbilicusWindow.show()

    def addGenerateGridCellsArea(self, box):
        label = QLabel("Generate Grid Cells")

        self.computeGridCellsButton = QPushButton("Compute")
        self.stopGridCellsButton = QPushButton("Stop")
        self.stopGridCellsButton.setEnabled(False)

        self.computeGridCellsButton.clicked.connect(self.computeGridCells)
        self.stopGridCellsButton.clicked.connect(self.stopGridCells)

        box.add_widget(label)
        box.add_widget(self.computeGridCellsButton)
        box.add_widget(self.stopGridCellsButton)

    def gridCellsComputation(self, config):
        try:
            compute_grid_cells(input_directory=config["original_2d_tiffs"], 
                               output_directory=config["downsampled_2d_tiffs"], 
                               downsample_factor=abs(config["downsample_factor"]))

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            print(f"Error in computation: {e}")

    def postComputation(self):
        # Note: This is executed in the child process
        print("Computation completed.")

    def computeGridCells(self):
        command = [
                "python3", "-m", "ThaumatoAnakalyptor.generate_half_sized_grid", 
                "--input_directory", str(self.Config["original_2d_tiffs"]), 
                "--output_directory", str(self.Config["downsampled_2d_tiffs"]), 
                "--downsample_factor", str(abs(self.Config["downsample_factor"]))
            ]

        self.process = subprocess.Popen(command)

        # self.process = multiprocessing.Process(target=self.gridCellsComputation, args=(self.Config,))
        # self.process.start()
        self.computeGridCellsButton.setEnabled(False)
        self.stopGridCellsButton.setEnabled(True)

    def stopGridCells(self):
        if self.process and self.process.poll() is None:  # Check if process is running
            self.process.terminate()  # or self.process.kill() for a more forceful termination
            self.process = None

        # if self.process and self.process.is_alive():
        #     os.kill(self.process.pid, signal.SIGTERM)
        #     self.process.join()
        self.computeGridCellsButton.setEnabled(True)
        self.stopGridCellsButton.setEnabled(False)

        # Clean up computation after completion
        self.postComputation()
        print("Computation process stopped.")

    def addPointcloudArea(self, box):
        label = QLabel("Pointcloud")

        self.recomputeCheckbox = QCheckBox("Recompute")
        self.computePointcloudButton = QPushButton("Compute")
        self.stopPointcloudButton = QPushButton("Stop")
        self.stopPointcloudButton.setEnabled(False)

        self.computePointcloudButton.clicked.connect(self.computePointcloud)
        self.stopPointcloudButton.clicked.connect(self.stopPointcloud)

        box.add_widget(label)
        box.add_widget(self.recomputeCheckbox)
        box.add_widget(self.computePointcloudButton)
        box.add_widget(self.stopPointcloudButton)

    def computePointcloud(self):
        try:
            command = [
                "python3", "-m", "ThaumatoAnakalyptor.grid_to_pointcloud", 
                "--base_path", "", 
                "--volume_subpath", self.Config["downsampled_3d_grids"], 
                "--disk_load_save", "", "", 
                "--pointcloud_subpath", os.path.join(self.Config["surface_points_path"], "point_cloud"), 
                "--num_threads", str(self.Config["num_threads"]), 
                "--gpus", str(self.Config["gpus"])
            ]
            
            if self.recomputeCheckbox.isChecked():
                command += ["--recompute"]
            self.process = subprocess.Popen(command)
            self.computePointcloudButton.setEnabled(False)
            self.stopPointcloudButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeGridCellsButton.setEnabled(True)
            self.stopPointcloudButton.setEnabled(False)

    def stopPointcloud(self):
        if self.process and self.process.poll() is None:  # Check if process is running
            self.process.terminate()  # or self.process.kill() for a more forceful termination
            self.process = None
        self.computePointcloudButton.setEnabled(True)
        self.stopPointcloudButton.setEnabled(False)
        print("Computation process stopped.")

    def addInstancesArea(self, box):
        label = QLabel("Instances")
        # self.addFieldWithLabel(box, "Batch Size:", "Enter Batch Size", "batchSizeField", value=4)
        self.computeInstancesButton = QPushButton("Compute")
        self.stopInstancesButton = QPushButton("Stop")
        self.stopInstancesButton.setEnabled(False)

        self.computeInstancesButton.clicked.connect(self.computeInstances)
        self.stopInstancesButton.clicked.connect(self.stopInstances)

        box.add_widget(label)
        box.add_widget(self.computeInstancesButton)
        box.add_widget(self.stopInstancesButton)

    def computeInstances(self):
        try:
            batch_size = int(self.Config["batch_size"])

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.pointcloud_to_instances", 
                "--path", self.Config["surface_points_path"], 
                "--dest", self.Config["surface_points_path"], 
                "--umbilicus_path", self.Config["umbilicus_path"], 
                "--main_drive", "", "--alternative_ply_drives", "", "", 
                "--max_umbilicus_dist", "-1",
                "--batch_size", str(batch_size),
            ]

            self.process = subprocess.Popen(command)
            self.computeInstancesButton.setEnabled(False)
            self.stopInstancesButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeInstancesButton.setEnabled(True)
            self.stopInstancesButton.setEnabled(False)

    def stopInstances(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeInstancesButton.setEnabled(True)
        self.stopInstancesButton.setEnabled(False)
        print("Computation process stopped.")

    def addMeshGeneration(self, layout):
        # Main Collapsible Box for Mesh Generation
        volumeBox = CollapsibleBox("Mesh Generation")

        # Generate Grid Cells Area
        stitchSheetBox = CollapsibleBox("Stitch Sheet")
        self.addStitchSheetArea(stitchSheetBox)
        volumeBox.add_widget(stitchSheetBox)

        # Meshing Area
        meshingBox = CollapsibleBox("Meshing")
        self.addMeshingArea(meshingBox)
        volumeBox.add_widget(meshingBox)

        # Flattening Area
        flatteningBox = CollapsibleBox("Flattening")
        self.addFlatteningArea(flatteningBox)
        volumeBox.add_widget(flatteningBox)

        # Slim UV Area
        slimUVBox = CollapsibleBox("Slim UV")
        self.addSlimArea(slimUVBox)
        volumeBox.add_widget(slimUVBox)

        # Finalize Area
        finalizeBox = CollapsibleBox("Finalize")
        self.addFinalizeArea(finalizeBox)
        volumeBox.add_widget(finalizeBox)

        # Swap Volume Area
        swapVolumeBox = CollapsibleBox("Swap Volume")
        self.addSwapVolumeArea(swapVolumeBox)
        volumeBox.add_widget(swapVolumeBox)

        layout.addWidget(volumeBox)

    def addFieldWithLabel(self, box, labelText, placeholderText, fieldAttribute, value=None):
        # Create a widget to hold the label and field, ensuring proper layout within the collapsible box
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Create and add the label
        label = QLabel(labelText)
        layout.addWidget(label)
        
        # Create, configure, and add the field
        field = QLineEdit()
        field.setPlaceholderText(placeholderText)
        if value:
            field.setText(str(value))
        layout.addWidget(field)
        
        # Add the composite widget to the box
        box.add_widget(widget)
        
        # Store a reference to the field using setattr for future access
        setattr(self, fieldAttribute, field)

    def toggleStartingPointSelection(self):
        self.isSelectingStartingPoint = not self.isSelectingStartingPoint
        if self.isSelectingStartingPoint:
            self.setCursor(Qt.CrossCursor)  # Change cursor to crosshair
        else:
            self.setCursor(Qt.ArrowCursor)  # Reset cursor to default

    def addStartingPointArea(self, box):
        # Starting Point button
        self.startingPointField = QPushButton("Select Starting Point")
        self.startingPointField.clicked.connect(self.toggleStartingPointSelection)
        box.add_widget(self.startingPointField)

        # Title for the starting point section
        title_label = QLabel("Starting Point:")
        box.add_widget(title_label)


        # Horizontal layout to hold the x, y, and z fields
        starting_point_layout = QHBoxLayout()

        # X coordinate field
        self.xField = QLineEdit()
        self.xField.setPlaceholderText("x")
        starting_point_layout.addWidget(QLabel("x:"))
        starting_point_layout.addWidget(self.xField)

        # Y coordinate field
        self.yField = QLineEdit()
        self.yField.setPlaceholderText("y")
        starting_point_layout.addWidget(QLabel("y:"))
        starting_point_layout.addWidget(self.yField)

        # Z coordinate field
        self.zField = QLineEdit()
        self.zField.setPlaceholderText("z")
        starting_point_layout.addWidget(QLabel("z:"))
        starting_point_layout.addWidget(self.zField)

        # Create a container widget to add the layout to
        starting_point_widget = QWidget()
        starting_point_widget.setLayout(starting_point_layout)

        # Add the starting point widget to the box
        box.add_widget(starting_point_widget)

    def setStitchSheetDefaultValues(self):
        self.sheetKRangeStartField.setText("-1")
        self.sheetKRangeEndField.setText("1")
        self.sheetZRangeStartField.setText("0")
        self.sheetZRangeEndField.setText("40000")
        self.minStepsField.setText("16")
        self.minEndStepsField.setText("4")
        self.maxNrWalksField.setText("30000")
        self.walkAggregationThresholdField.setText("5")

    def addStitchSheetArea(self, box):
        label = QLabel("Stitch Sheet")
        box.add_widget(label)

        self.addStartingPointArea(box)

        # Sheet K range fields
        self.addFieldWithLabel(box, "Sheet K Range Start:", "Enter start for K range", "sheetKRangeStartField")
        self.addFieldWithLabel(box, "Sheet K Range End:", "Enter end for K range", "sheetKRangeEndField")

        # Sheet Z range fields
        self.addFieldWithLabel(box, "Sheet Z Range Start:", "Enter start for Z range", "sheetZRangeStartField")
        self.addFieldWithLabel(box, "Sheet Z Range End:", "Enter end for Z range", "sheetZRangeEndField")

        # Other parameter fields
        self.addFieldWithLabel(box, "Min Steps:", "Enter minimum steps", "minStepsField")
        self.addFieldWithLabel(box, "Min End Steps:", "Enter minimum end steps", "minEndStepsField")
        self.addFieldWithLabel(box, "Max Nr Walks:", "Enter maximum number of walks", "maxNrWalksField")
        self.addFieldWithLabel(box, "Walk Aggregation Threshold:", "Enter walk aggregation threshold", "walkAggregationThresholdField")


        self.recomputeStitchSheetCheckbox = QCheckBox("Recompute")
        self.continueSegmentationCheckbox = QCheckBox("Continue Segmentation")
        self.computeStitchSheetButton = QPushButton("Compute")
        self.stopStitchSheetButton = QPushButton("Stop")
        self.stopStitchSheetButton.setEnabled(False)

        self.defaultValuesButton = QPushButton("Default Values")
        self.defaultValuesButton.clicked.connect(self.setStitchSheetDefaultValues)

        self.computeStitchSheetButton.clicked.connect(self.computeStitchSheet)
        self.stopStitchSheetButton.clicked.connect(self.stopStitchSheet)

        box.add_widget(self.recomputeStitchSheetCheckbox)
        box.add_widget(self.continueSegmentationCheckbox)
        box.add_widget(self.computeStitchSheetButton)
        box.add_widget(self.stopStitchSheetButton)
        box.add_widget(self.defaultValuesButton)

    # def computeStitchSheet(self):
    #     try:
    #         # Fetching values from GUI fields
    #         path = os.path.join(self.Config["surface_points_path"], "point_cloud_colorized_verso_subvolume_blocks")
    #         starting_point = f"{self.xField.text()} {self.yField.text()} {self.zField.text()}"
    #         sheet_k_range = f"{self.sheetKRangeStartField.text()} {self.sheetKRangeEndField.text()}"
    #         sheet_z_range = f"{self.sheetZRangeStartField.text()} {self.sheetZRangeEndField.text()}"
    #         min_steps = self.minStepsField.text()
    #         min_end_steps = self.minEndStepsField.text()
    #         max_nr_walks = self.maxNrWalksField.text()
    #         continue_segmentation = '1' if self.continueSegmentationCheckbox.isChecked() else '0'
    #         recompute = '1' if self.recomputeStitchSheetCheckbox.isChecked() else '0'
    #         walk_aggregation_threshold = self.walkAggregationThresholdField.text()

    #         # Construct the command
    #         command = [
    #             "python3", "-m", "ThaumatoAnakalyptor.Random_Walks",
    #             "--path", path,
    #             "--starting_point", self.xField.text(), self.yField.text(), self.zField.text(),
    #             "--sheet_k_range", self.sheetKRangeStartField.text(), self.sheetKRangeEndField.text(),
    #             "--sheet_z_range", self.sheetZRangeStartField.text(), self.sheetZRangeEndField.text(),
    #             "--min_steps", min_steps,
    #             "--min_end_steps", min_end_steps,
    #             "--max_nr_walks", max_nr_walks,
    #             "--continue_segmentation", continue_segmentation,
    #             "--recompute", recompute,
    #             "--walk_aggregation_threshold", walk_aggregation_threshold
    #         ]

    #         print(f"Command: {command}")

    #         # Starting the process
    #         self.process = subprocess.Popen(command)
    #         self.computeStitchSheetButton.setEnabled(False)
    #         self.stopStitchSheetButton.setEnabled(True)

    #         # Create a thread to monitor the completion of the process
    #         # self.monitorThread = threading.Thread(target=self.monitorProcess)
    #         # self.monitorThread.start()

    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
    #         self.computeStitchSheetButton.setEnabled(True)
    #         self.stopStitchSheetButton.setEnabled(False)

    def stitchSheetComputation(self, overlapp_threshold, start_point, path, recompute, stop_event):
        try:
            # Compute
            compute_stitch_sheet(overlapp_threshold, start_point=start_point, path=path, recompute=recompute, stop_event=stop_event)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            print(f"Error in computation: {e}")

    def computeStitchSheet(self):
        try:
            # Fetching values from GUI fields
            path = os.path.join(self.Config["surface_points_path"], "point_cloud_colorized_verso_subvolume_blocks")
            start_point = [int(self.xField.text()), int(self.yField.text()), int(self.zField.text())]
            sheet_k_range = (int(self.sheetKRangeStartField.text()), int(self.sheetKRangeEndField.text()))
            sheet_z_range = (int(self.sheetZRangeStartField.text()), int(self.sheetZRangeEndField.text()))
            min_steps = int(self.minStepsField.text())
            min_end_steps = int(self.minEndStepsField.text())
            max_nr_walks = int(self.maxNrWalksField.text())
            continue_segmentation = self.continueSegmentationCheckbox.isChecked()
            recompute = self.recomputeStitchSheetCheckbox.isChecked()
            walk_aggregation_threshold = int(self.walkAggregationThresholdField.text())


            overlapp_threshold = {"sample_ratio_score": 0.03, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.0005, "score_threshold": 0.005, "fit_sheet": False, "cost_threshold": 17, "cost_percentile": 75, "cost_percentile_threshold": 14, 
                          "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.005,
                          "cost_threshold_prediction": 2.5, "min_prediction_threshold": 0.15, "nr_points_min": 200.0, "nr_points_max": 4000.0, "min_patch_points": 300.0, 
                          "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
                          "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
                          "min_points_winding_switch": 3800, "min_winding_switch_sheet_distance": 9, "max_winding_switch_sheet_distance": 20, "winding_switch_sheet_score_factor": 1.5, "winding_direction": -1.0, "enable_winding_switch": False, "enable_winding_switch_postprocessing": False,
                          "surrounding_patches_size": 3, "max_sheet_clip_distance": 60, "sheet_z_range": (-5000, 400000), "sheet_k_range": (-1, 2), "volume_min_certainty_total_percentage": 0.0, "max_umbilicus_difference": 30,
                          "walk_aggregation_threshold": 100, "walk_aggregation_max_current": -1
                          }

            # max_nr_walks = 10000
            max_steps = 101
            # min_steps = 16
            max_tries = 6
            # min_end_steps = 4
            max_unchanged_walks = 30 * max_nr_walks

            overlapp_threshold["sheet_z_range"] = [z_range_ /(200.0 / 50.0) for z_range_ in sheet_z_range]
            overlapp_threshold["sheet_k_range"] = sheet_k_range
            overlapp_threshold["walk_aggregation_threshold"] = walk_aggregation_threshold
            overlapp_threshold["max_nr_walks"] = max_nr_walks
            overlapp_threshold["max_unchanged_walks"] = max_unchanged_walks
            overlapp_threshold["continue_walks"] = continue_segmentation
            overlapp_threshold["max_steps"] = max_steps
            overlapp_threshold["max_tries"] = max_tries
            overlapp_threshold["min_steps"] = min_steps
            overlapp_threshold["min_end_steps"] = min_end_steps

            # Creating an Event object
            self.stop_event = threading.Event()

            # start self.stitchSheetComputation in a new thread
            self.process = threading.Thread(target=self.stitchSheetComputation, args=(overlapp_threshold, start_point, path, recompute, self.stop_event))
            self.process.start()

            # self.process = multiprocessing.Process(target=self.stitchSheetComputation, args=(overlapp_threshold, start_point, path, recompute,))
            # self.process.start()

            self.computeStitchSheetButton.setEnabled(False)
            self.stopStitchSheetButton.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeStitchSheetButton.setEnabled(True)
            self.stopStitchSheetButton.setEnabled(False)


    def stopStitchSheet(self):
        # if self.process and self.process.poll() is None:
        #     self.process.terminate()
        #     self.process = None

        # if self.process and self.process.is_alive():
        #     os.kill(self.process.pid, signal.SIGTERM)
        #     self.process.join()

        # Kill the thread
        self.stop_event.set()
        self.process.join()

        self.computeStitchSheetButton.setEnabled(True)
        self.stopStitchSheetButton.setEnabled(False)
        print("Computation process stopped.")

    def addMeshingArea(self, box):
        label = QLabel("Meshing")
        self.computeMeshingButton = QPushButton("Compute")
        self.stopMeshingButton = QPushButton("Stop")
        self.stopMeshingButton.setEnabled(False)

        self.computeMeshingButton.clicked.connect(self.computeMeshing)
        self.stopMeshingButton.clicked.connect(self.stopMeshing)

        box.add_widget(label)
        box.add_widget(self.computeMeshingButton)
        box.add_widget(self.stopMeshingButton)

    def computeMeshing(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            path_base = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/")
            print(f"path_base: {path_base}")

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.sheet_to_mesh",
                "--path_base", path_base, 
                "--path_ta", "point_cloud_colorized_verso_subvolume_main_sheet_RW.ta", 
                "--umbilicus_path", self.Config["umbilicus_path"]
            ]
            
            self.process = subprocess.Popen(command)
            self.computeMeshingButton.setEnabled(False)
            self.stopMeshingButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeMeshingButton.setEnabled(True)
            self.stopMeshingButton.setEnabled(False)

    def stopMeshing(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeMeshingButton.setEnabled(True)
        self.stopMeshingButton.setEnabled(False)
        print("Computation process stopped.")

    def addFlatteningArea(self, box):
        label = QLabel("Flattening")
        self.delaunyCheckbox = QCheckBox("Use Delaunay")
        self.delaunyCheckbox.setChecked(True)
        self.computeFlatteningButton = QPushButton("Compute")
        self.stopFlatteningButton = QPushButton("Stop")
        self.stopFlatteningButton.setEnabled(False)

        self.computeFlatteningButton.clicked.connect(self.computeFlattening)
        self.stopFlatteningButton.clicked.connect(self.stopFlattening)

        box.add_widget(label)
        box.add_widget(self.delaunyCheckbox)
        box.add_widget(self.computeFlatteningButton)
        box.add_widget(self.stopFlatteningButton)

    def computeFlattening(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            path = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}", "point_cloud_colorized_verso_subvolume_blocks.obj")

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.mesh_to_uv", 
                "--path", path, 
                "--umbilicus_path", self.Config["umbilicus_path"]
            ]

            if self.delaunyCheckbox.isChecked():
                command += ["--enable_delauny"]


            self.process = subprocess.Popen(command)
            self.computeFlatteningButton.setEnabled(False)
            self.stopFlatteningButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeFlatteningButton.setEnabled(True)
            self.stopFlatteningButton.setEnabled(False)

    def stopFlattening(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeFlatteningButton.setEnabled(True)
        self.stopFlatteningButton.setEnabled(False)
        print("Computation process stopped.")

    def addSlimArea(self, box):
        label = QLabel("Slim")
        self.computeSlimButton = QPushButton("Compute")
        self.stopSlimButton = QPushButton("Stop")
        self.stopSlimButton.setEnabled(False)

        self.computeSlimButton.clicked.connect(self.computeSlim)
        self.stopSlimButton.clicked.connect(self.stopSlim)

        box.add_widget(label)
        box.add_widget(self.computeSlimButton)
        box.add_widget(self.stopSlimButton)

    def computeSlim(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            path = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}", "point_cloud_colorized_verso_subvolume_blocks_uv.obj")

            command_slimFlattening = [
                "python3", "-m", "ThaumatoAnakalyptor.slim_uv",
                "--path", path,
                "--iter", "20"
            ]

            command = command_slimFlattening

            self.process = subprocess.Popen(command)
            self.computeSlimButton.setEnabled(False)
            self.stopSlimButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeSlimButton.setEnabled(True)
            self.stopSlimButton.setEnabled(False)

    def stopSlim(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeSlimButton.setEnabled(True)
        self.stopSlimButton.setEnabled(False)
        print("Computation process stopped.")

    def addFinalizeArea(self, box):
        label = QLabel("Finalize")
        self.computeFinalizeButton = QPushButton("Compute")
        self.stopFinalizeButton = QPushButton("Stop")
        self.stopFinalizeButton.setEnabled(False)

        self.computeFinalizeButton.clicked.connect(self.computeFinalize)
        self.stopFinalizeButton.clicked.connect(self.stopFinalize)

        box.add_widget(label)
        box.add_widget(self.computeFinalizeButton)
        box.add_widget(self.stopFinalizeButton)

    def computeFinalize(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            input_mesh = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}", "point_cloud_colorized_verso_subvolume_blocks_uv_flatboi.obj")
            command = [
                "python3", "-m", "ThaumatoAnakalyptor.finalize_mesh", 
                "--input_mesh", input_mesh, 
                "--cut_size", "40000", 
                "--scale_factor", f"{abs(self.Config['downsample_factor']):f}"
            ]

            self.process = subprocess.Popen(command)
            self.computeFinalizeButton.setEnabled(False)
            self.stopFinalizeButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeFinalizeButton.setEnabled(True)
            self.stopFinalizeButton.setEnabled(False)

    def stopFinalize(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeFinalizeButton.setEnabled(True)
        self.stopFinalizeButton.setEnabled(False)
        print("Computation process stopped.")

    def addSwapVolumeArea(self, box):
        label = QLabel("Swap Volume")
        self.targetVolumeIdField = QLineEdit()
        self.targetVolumeIdField.setPlaceholderText("Target Volume ID")
        self.computeSwapVolumeButton = QPushButton("Compute")
        self.stopSwapVolumeButton = QPushButton("Stop")
        self.stopSwapVolumeButton.setEnabled(False)

        self.computeSwapVolumeButton.clicked.connect(self.computeSwapVolume)
        self.stopSwapVolumeButton.clicked.connect(self.stopSwapVolume)

        box.add_widget(label)
        box.add_widget(self.targetVolumeIdField)
        box.add_widget(self.computeSwapVolumeButton)
        box.add_widget(self.stopSwapVolumeButton)

    def computeSwapVolume(self):
        try:
            target_volume_id = self.targetVolumeIdField.text()

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.mesh_transform", 
                "--transform_path", self.Config["surface_points_path"], 
                "--targed_volume_id", target_volume_id,
                "--base_path", self.Config["surface_points_path"]
            ]

            self.process = subprocess.Popen(command)
            self.computeSwapVolumeButton.setEnabled(False)
            self.stopSwapVolumeButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeSwapVolumeButton.setEnabled(True)
            self.stopSwapVolumeButton.setEnabled(False)

    def stopSwapVolume(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeSwapVolumeButton.setEnabled(True)
        self.stopSwapVolumeButton.setEnabled(False)
        print("Computation process stopped.")

    def addRendering(self, layout):
        # Main Collapsible Box for Rendering
        renderingBox = CollapsibleBox("Rendering")

        # PPM Area
        ppmBox = CollapsibleBox("PPM")
        self.addPpmArea(ppmBox)
        renderingBox.add_widget(ppmBox)

        # Texturing Area
        texturingBox = CollapsibleBox("Texturing")
        self.addTexturingArea(texturingBox)
        renderingBox.add_widget(texturingBox)

        layout.addWidget(renderingBox)

    def addPpmArea(self, box):
        label = QLabel("PPM")
        self.computePpmButton = QPushButton("Compute")
        self.stopPpmButton = QPushButton("Stop")
        self.stopPpmButton.setEnabled(False)

        self.computePpmButton.clicked.connect(self.computePPM)
        self.stopPpmButton.clicked.connect(self.stopPPM)

        box.add_widget(label)
        box.add_widget(self.computePpmButton)
        box.add_widget(self.stopPpmButton)

    def computePPM(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            path_base = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/")
            print(f"path_base: {path_base}")

            downsampled_2d_tiffs = self.Config.get("downsampled_2d_tiffs", None)
            if downsampled_2d_tiffs is None:
                QMessageBox.critical(self, "Error", f"Please specify the 2D Tiff files path")
                return
            # volpkg is downsampled_2d_tiffs without last two folders
            volpkg_path = os.path.dirname(os.path.dirname(downsampled_2d_tiffs)) + "/"
            volume = os.path.basename(downsampled_2d_tiffs)
            obj_path = self.Config.get("surface_points_path", None)
            if obj_path is None:
                QMessageBox.critical(self, "Error", f"Please specify the surface points path")
                return
            ppm_path = obj_path + f"/working/working_{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/thaumato.obj"
            obj_path = obj_path + f"/working/working_{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/point_cloud_colorized_verso_subvolume_blocks_uv_flatboi.obj"
            print("ppm paths:", volpkg_path, volume, obj_path, ppm_path)

            command = [
                "/volume-cartographer-papyrus/build/bin/vc_generate_ppm",
                "--input-mesh", obj_path,
                "--output-ppm", ppm_path,
                "--uv-reuse"
            ]
            
            self.process = subprocess.Popen(command)
            self.computePpmButton.setEnabled(False)
            self.stopPpmButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computePpmButton.setEnabled(True)
            self.stopPpmButton.setEnabled(False)

    def stopPPM(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computePpmButton.setEnabled(True)
        self.stopPpmButton.setEnabled(False)
        print("Computation process stopped.")

    def addTexturingArea(self, box):
        label = QLabel("Texturing")
        self.computeTexturingButton = QPushButton("Compute")
        self.stopTexturingButton = QPushButton("Stop")
        self.stopTexturingButton.setEnabled(False)

        self.computeTexturingButton.clicked.connect(self.computeTexturing)
        self.stopTexturingButton.clicked.connect(self.stopTexturing)

        box.add_widget(label)
        box.add_widget(self.computeTexturingButton)
        box.add_widget(self.stopTexturingButton)

    def computeTexturing(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            path_base = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/")
            print(f"path_base: {path_base}")

            downsampled_2d_tiffs = self.Config.get("downsampled_2d_tiffs", None)
            if downsampled_2d_tiffs is None:
                QMessageBox.critical(self, "Error", f"Please specify the 2D Tiff files path")
                return
            # volpkg is downsampled_2d_tiffs without last two folders
            volpkg_path = os.path.dirname(os.path.dirname(downsampled_2d_tiffs)) + "/"
            volume = os.path.basename(downsampled_2d_tiffs)
            obj_path = self.Config.get("surface_points_path", None)
            if obj_path is None:
                QMessageBox.critical(self, "Error", f"Please specify the surface points path")
                return
            ppm_path = obj_path + f"/working_{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/thaumato.obj"
            obj_path = obj_path + f"/working_{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/point_cloud_colorized_verso_subvolume_blocks_uv_flatboi.obj"
            print("ppm paths:", volpkg_path, volume, obj_path, ppm_path)

            command = [
                "/volume-cartographer-papyrus/build/bin/vc_generate_ppm",
                "--input-mesh", obj_path,
                "--output-ppm", ppm_path,
                "--uv-reuse"
            ]
            
            self.process = subprocess.Popen(command)
            self.computeTexturingButton.setEnabled(False)
            self.stopTexturingButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeTexturingButton.setEnabled(True)
            self.stopTexturingButton.setEnabled(False)

    def stopTexturing(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeTexturingButton.setEnabled(True)
        self.stopTexturingButton.setEnabled(False)
        print("Computation process stopped.")

    def addInkDetection(self, layout):
        # Main Collapsible Box for Ink Detection
        renderingBox = CollapsibleBox("Ink Detection")

        # Ink Detection Area
        timesformerBox = CollapsibleBox("TimeSformer Inference")
        self.addInkDetectionArea(timesformerBox)
        renderingBox.add_widget(timesformerBox)

        layout.addWidget(renderingBox)

    def addInkDetectionArea(self, box):
        label = QLabel("Ink Detection")
        self.computeInkDetectionButton = QPushButton("Compute")
        self.stopInkDetectionButton = QPushButton("Stop")
        self.stopInkDetectionButton.setEnabled(False)

        self.computeInkDetectionButton.clicked.connect(self.computeInkDetection)
        self.stopInkDetectionButton.clicked.connect(self.stopInkDetection)

        box.add_widget(label)
        box.add_widget(self.computeInkDetectionButton)
        box.add_widget(self.stopInkDetectionButton)

    def computeInkDetection(self):
        try:
            # Set the path to the virtual environment's Python executable
            python_executable = "/youssefGP/bin/python"  # Adjust this path as necessary

            # Prepare the environment variable
            env = os.environ.copy()
            env["WANDB_MODE"] = "dryrun"  # Set the WANDB_MODE environment variable

            # Assuming you have these values or similar ways to obtain them
            segment_id = "working_20230520191415"
            # Modify the segment_path as per your requirement or dynamically determine it
            segment_path = "scroll.volpkg"
            model_path = "Vesuvius-Grandprize-Winner/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt"
            out_path = "./"  # The current directory or specify as needed

            # Construct the command with the provided arguments
            command = [
                python_executable, "Vesuvius-Grandprize-Winner/inference_timesformer.py",
                "--segment_id", segment_id,
                "--segment_path", segment_path,
                "--model_path", model_path,
                "--out_path", out_path
            ]

            # Run the command with the specified environment variables
            self.process = subprocess.Popen(command, env=env)
            self.computeInkDetectionButton.setEnabled(False)
            self.stopInkDetectionButton.setEnabled(True)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeInkDetectionButton.setEnabled(True)
            self.stopInkDetectionButton.setEnabled(False)

    def stopInkDetection(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
        self.computeInkDetectionButton.setEnabled(True)
        self.stopInkDetectionButton.setEnabled(False)
        print("Computation process stopped.")