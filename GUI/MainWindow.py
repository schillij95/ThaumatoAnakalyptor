### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QMainWindow, QAction, QSplitter, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QFrame,
                             QFileDialog, QLineEdit, QCheckBox, QMessageBox, QStyle, QVBoxLayout, QScrollArea, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

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

# import computation functions
from ThaumatoAnakalyptor.generate_half_sized_grid import compute as compute_grid_cells
from ThaumatoAnakalyptor.grid_to_pointcloud import compute as compute_pointcloud


class ThaumatoAnakalyptor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadConfig()
        self.process = None
        self.isSelectingStartingPoint = False

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
        pixmap = QPixmap(400, 400)
        pixmap.fill(Qt.white)  # White background
        left_panel.setPixmap(pixmap)

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
        self.addCollapsibleSection(right_layout, "Rendering")

        right_panel.setLayout(right_layout)

        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel_scroll_area)

        # Trigger click on Config
        config.triggered.connect(self.openConfigWindow)

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
                   "- Click on the TIFF to place a point.\n" 
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
                               downsample_factor=config["downsample_factor"])

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
                "--downsample_factor", str(self.Config["downsample_factor"])
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
            command = [
                "python3", "-m", "ThaumatoAnakalyptor.pointcloud_to_instances", 
                "--path", self.Config["surface_points_path"], 
                "--dest", self.Config["surface_points_path"], 
                "--umbilicus_path", self.Config["umbilicus_path"], 
                "--main_drive", "", "--alternative_ply_drives", "", "", 
                "--max_umbilicus_dist", "-1"
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

        # Finalize Area
        finalizeBox = CollapsibleBox("Finalize")
        self.addFinalizeArea(finalizeBox)
        volumeBox.add_widget(finalizeBox)

        # Swap Volume Area
        swapVolumeBox = CollapsibleBox("Swap Volume")
        self.addSwapVolumeArea(swapVolumeBox)
        volumeBox.add_widget(swapVolumeBox)

        layout.addWidget(volumeBox)

    def addFieldWithLabel(self, box, labelText, placeholderText, fieldAttribute):
        # Create a widget to hold the label and field, ensuring proper layout within the collapsible box
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Create and add the label
        label = QLabel(labelText)
        layout.addWidget(label)
        
        # Create, configure, and add the field
        field = QLineEdit()
        field.setPlaceholderText(placeholderText)
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

        self.computeStitchSheetButton.clicked.connect(self.computeStitchSheet)
        self.stopStitchSheetButton.clicked.connect(self.stopStitchSheet)

        box.add_widget(self.recomputeStitchSheetCheckbox)
        box.add_widget(self.continueSegmentationCheckbox)
        box.add_widget(self.computeStitchSheetButton)
        box.add_widget(self.stopStitchSheetButton)

    def computeStitchSheet(self):
        try:
            # Fetching values from GUI fields
            path = os.path.join(self.Config["surface_points_path"], "point_cloud_colorized_verso_subvolume_blocks")
            starting_point = f"{self.xField.text()} {self.yField.text()} {self.zField.text()}"
            sheet_k_range = f"{self.sheetKRangeStartField.text()} {self.sheetKRangeEndField.text()}"
            sheet_z_range = f"{self.sheetZRangeStartField.text()} {self.sheetZRangeEndField.text()}"
            min_steps = self.minStepsField.text()
            min_end_steps = self.minEndStepsField.text()
            max_nr_walks = self.maxNrWalksField.text()
            continue_segmentation = '1' if self.continueSegmentationCheckbox.isChecked() else '0'
            recompute = '1' if self.recomputeCheckbox.isChecked() else '0'
            walk_aggregation_threshold = self.walkAggregationThresholdField.text()

            # Construct the command
            command = [
                "python3", "-m", "ThaumatoAnakalyptor.Random_Walks",
                "--path", path,
                "--starting_point", starting_point,
                "--sheet_k_range", sheet_k_range,
                "--sheet_z_range", sheet_z_range,
                "--min_steps", min_steps,
                "--min_end_steps", min_end_steps,
                "--max_nr_walks", max_nr_walks,
                "--continue_segmentation", continue_segmentation,
                "--recompute", recompute,
                "--walk_aggregation_threshold", walk_aggregation_threshold
            ]

            # Starting the process
            self.process = subprocess.Popen(command)
            self.computeStitchSheetButton.setEnabled(False)
            self.stopStitchSheetButton.setEnabled(True)

            # Create a thread to monitor the completion of the process
            self.monitorThread = threading.Thread(target=self.monitorProcess)
            self.monitorThread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeStitchSheetButton.setEnabled(True)
            self.stopStitchSheetButton.setEnabled(False)

    def stopStitchSheet(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
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
        self.computeFlatteningButton = QPushButton("Compute")
        self.stopFlatteningButton = QPushButton("Stop")
        self.stopFlatteningButton.setEnabled(False)

        self.computeFlatteningButton.clicked.connect(self.computeFlattening)
        self.stopFlatteningButton.clicked.connect(self.stopFlattening)

        box.add_widget(label)
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
            input_mesh = os.path.join(self.Config["surface_points_path"], f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}", "point_cloud_colorized_verso_subvolume_blocks_uv.obj")
            command = [
                "python3", "-m", "ThaumatoAnakalyptor.finalize_mesh", 
                "--input_mesh", input_mesh, 
                "--cut_size", "40000", 
                "--scale_factor", f"{self.Config['downsample_factor']:f}"
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