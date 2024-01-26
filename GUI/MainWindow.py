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
import multiprocessing
# multiprocessing.set_start_method('spawn')

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
        self.addCollapsibleSection(right_layout, "Mesh Generation")
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
                                "Select the 8um downsampled 2D TIFF files to construct the umbilicus. \n\nMake sure to have the 8um 2D TIFF folder <name> and the 8um 3D Grid Cells folder <name>_grids in the same directory.")

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

        self.computePointcloudButton.clicked.connect(self.computePointcloud)
        self.stopPointcloudButton.clicked.connect(self.stopPointcloud)

        box.add_widget(label)
        box.add_widget(self.recomputeCheckbox)
        box.add_widget(self.computePointcloudButton)
        box.add_widget(self.stopPointcloudButton)

    def pointcloudComputation(self, config, recompute):
        try:
            # disk_load_save=["", ""], 
            # base_path="",
            # volume_subpath=config["downsampled_3d_grids"],
            # pointcloud_subpath=config["pointcloud_subpath"],
            # maximum_distance=-1,
            # recompute=recompute,
            # fix_umbilicus=False,
            # start_block=(500, 500, 500),
            # num_threads=config["num_threads"],
            # gpus=config["gpus"]
            args = {
                "disk_load_save": ["", ""],
                "base_path": "",
                "volume_subpath": config["downsampled_3d_grids"],
                "pointcloud_subpath": config["pointcloud_subpath"],
                "maximum_distance": -1,
                "recompute": recompute,
                "fix_umbilicus": False,
                "start_block": (500, 500, 500),
                "num_threads": config["num_threads"],
                "gpus": config["gpus"]
            }

            print("Computing grid cells...")
            compute_pointcloud(**args)

            # Clean up computation after completion
            self.postComputation()
        except Exception as e:
            print(f"Error in computation: {e}")

    def computePointcloud(self):
        self.process = multiprocessing.Process(target=self.pointcloudComputation, args=(self.Config, self.recomputeCheckbox.isChecked(),))
        self.process.start()
        self.computePointcloudButton.setEnabled(False)
        self.stopPointcloudButton.setEnabled(True)

    def stopPointcloud(self):
        if self.process and self.process.is_alive():
            os.kill(self.process.pid, signal.SIGTERM)
            self.process.join()
        self.computePointcloudButton.setEnabled(True)
        self.stopPointcloudButton.setEnabled(False)
        print("Computation process stopped.")
