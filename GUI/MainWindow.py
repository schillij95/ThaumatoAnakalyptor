### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QMainWindow, QAction, QSplitter, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QFrame,
                             QFileDialog, QLineEdit, QCheckBox, QMessageBox, QStyle, QVBoxLayout)
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

        # Right Panel with sections
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)  # Align sections at the top

        # Collapsible Sections
        self.addVolumeProcessing(right_layout)
        self.addMeshGeneration(right_layout)
        self.addCollapsibleSection(right_layout, "Rendering")

        right_panel.setLayout(right_layout)

        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

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
                                "Make sure to have the appropriate paths set in the Config. Place the umbilicus in the center of the scroll.")

    def openUmbilicusWindow(self):
        if self.Config.get("downsampled_2d_tiffs", None) and os.path.exists(self.Config["downsampled_2d_tiffs"]):
            self.umbilicusWindow = UmbilicusWindow(self.Config["downsampled_2d_tiffs"])
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
        self.process = multiprocessing.Process(target=self.gridCellsComputation, args=(self.Config,))
        self.process.start()
        self.computeGridCellsButton.setEnabled(False)
        self.stopGridCellsButton.setEnabled(True)

    def stopGridCells(self):
        if self.process and self.process.is_alive():
            os.kill(self.process.pid, signal.SIGTERM)
            self.process.join()
        self.computeGridCellsButton.setEnabled(True)
        self.stopGridCellsButton.setEnabled(False)
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
            command = ["python3", "-m" "ThaumatoAnakalyptor.grid_to_pointcloud", "--base_path", "", "--volume_subpath", self.Config["downsampled_3d_grids"], "--disk_load_save", "", "", "--pointcloud_subpath", os.path.join(self.Config["pointcloud_subpath"], "point_cloud"), "--num_threads", str(self.Config["num_threads"]), "--gpus", str(self.Config["gpus"])]
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
            command = ["python3", "-m" "ThaumatoAnakalyptor.pointcloud_to_instances", "--path", self.Config["pointcloud_subpath"], "--dest", self.Config["pointcloud_subpath"], "--umbilicus_path", self.Config["umbilicus_path"], "--main_drive", "", "--alternative_ply_drives", "", "", "--max_umbilicus_dist", "-1"]
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

    def addStitchSheetArea(self, box):
        label = QLabel("Stitch Sheet")
        # Starting Point button
        self.startingPointField = QPushButton("Select Starting Point")
        self.startingPointField.clicked.connect(lambda: self.selectPath(self.startingPointField)) # TODO
        self.starting_point = [0, 0, 0]
        # Sheet k range fields
        self.sheetKRangeStartField = QLineEdit()
        self.sheetKRangeStartField.setPlaceholderText("Sheet K Range Start")
        self.sheetKRangeEndField = QLineEdit()
        self.sheetKRangeEndField.setPlaceholderText("Sheet K Range End")

        # Sheet z range fields
        self.sheetZRangeStartField = QLineEdit()
        self.sheetZRangeStartField.setPlaceholderText("Sheet Z Range Start")
        self.sheetZRangeEndField = QLineEdit()
        self.sheetZRangeEndField.setPlaceholderText("Sheet Z Range End")

        # Other parameter fields
        self.minStepsField = QLineEdit()
        self.minStepsField.setPlaceholderText("Min Steps")
        self.minEndStepsField = QLineEdit()
        self.minEndStepsField.setPlaceholderText("Min End Steps")
        self.maxNrWalksField = QLineEdit()
        self.maxNrWalksField.setPlaceholderText("Max Nr Walks")
        self.walkAggregationThresholdField = QLineEdit()
        self.walkAggregationThresholdField.setPlaceholderText("Walk Aggregation Threshold")

        self.recomputeStitchSheetCheckbox = QCheckBox("Recompute")
        self.continueSegmentationCheckbox = QCheckBox("Continue Segmentation")
        self.computeStitchSheetButton = QPushButton("Compute")
        self.stopStitchSheetButton = QPushButton("Stop")
        self.stopStitchSheetButton.setEnabled(False)

        self.computeStitchSheetButton.clicked.connect(self.computeStitchSheet)
        self.stopStitchSheetButton.clicked.connect(self.stopStitchSheet)

        box.add_widget(label)
        box.add_widget(self.startingPointField)
        box.add_widget(self.sheetKRangeStartField)
        box.add_widget(self.sheetKRangeEndField)
        box.add_widget(self.sheetZRangeStartField)
        box.add_widget(self.sheetZRangeEndField)
        box.add_widget(self.minStepsField)
        box.add_widget(self.minEndStepsField)
        box.add_widget(self.maxNrWalksField)
        box.add_widget(self.walkAggregationThresholdField)
        box.add_widget(self.recomputeStitchSheetCheckbox)
        box.add_widget(self.continueSegmentationCheckbox)
        box.add_widget(self.computeStitchSheetButton)
        box.add_widget(self.stopStitchSheetButton)

    def computeStitchSheet(self):
        try:
            # Fetching values from GUI fields
            path = os.path.join(self.Config["pointcloud_subpath"], "point_cloud_colorized_verso_subvolume_blocks")
            starting_point = f"{self.starting_point[0]} {self.starting_point[1]} {self.starting_point[2]}"
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
            command = ["python3", "-m" "ThaumatoAnakalyptor.meshing", "--path", self.Config["pointcloud_subpath"], "--dest", self.Config["pointcloud_subpath"], "--umbilicus_path", self.Config["umbilicus_path"], "--main_drive", "", "--alternative_ply_drives", "", "", "--max_umbilicus_dist", "-1"]
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
            command = ["python3", "-m" "ThaumatoAnakalyptor.flattening", "--path", self.Config["pointcloud_subpath"], "--dest", self.Config["pointcloud_subpath"], "--umbilicus_path", self.Config["umbilicus_path"], "--main_drive", "", "--alternative_ply_drives", "", "", "--max_umbilicus_dist", "-1"]
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
            command = ["python3", "-m" "ThaumatoAnakalyptor.finalize", "--path", self.Config["pointcloud_subpath"], "--dest", self.Config["pointcloud_subpath"], "--umbilicus_path", self.Config["umbilicus_path"], "--main_drive", "", "--alternative_ply_drives", "", "", "--max_umbilicus_dist", "-1"]
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
        self.computeSwapVolumeButton = QPushButton("Compute")
        self.stopSwapVolumeButton = QPushButton("Stop")
        self.stopSwapVolumeButton.setEnabled(False)

        self.computeSwapVolumeButton.clicked.connect(self.computeSwapVolume)
        self.stopSwapVolumeButton.clicked.connect(self.stopSwapVolume)

        box.add_widget(label)
        box.add_widget(self.computeSwapVolumeButton)
        box.add_widget(self.stopSwapVolumeButton)

    def computeSwapVolume(self):
        try:
            command = ["python3", "-m" "ThaumatoAnakalyptor.swap_volume", "--path", self.Config["pointcloud_subpath"], "--dest", self.Config["pointcloud_subpath"], "--umbilicus_path", self.Config["umbilicus_path"], "--main_drive", "", "--alternative_ply_drives", "", "", "--max_umbilicus_dist", "-1"]
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