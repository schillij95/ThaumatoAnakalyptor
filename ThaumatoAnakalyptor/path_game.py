import sys
import numpy as np
import vispy
from vispy import app, scene
# vispy.use("egl")
app.use_app("glfw")
import zarr
from vispy.visuals.transforms import STTransform
import time
import cv2
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtGui import QMouseEvent


# Load your Zarr volume
zarr_path = '/media/julian/1/PHerc0332.volpkg/volumes/20231117143551_standardized.zarr'
zarr_map = zarr.open(zarr_path, mode='r')
resolution = 1  # Use the desired resolution level
vol = zarr_map[resolution][:1000, :1000, :1000]
vol = np.array(vol)  # Convert Zarr data to a NumPy array

# Prepare canvas
canvas = scene.SceneCanvas(show=False)
view = canvas.central_widget.add_view()

# Create the plane visual
plane = scene.visuals.Volume(
    vol,
    parent=view.scene,
    raycasting_mode='plane',  # Use plane rendering mode
    method='mip',  # Maximum Intensity Projection
    plane_thickness=3.0,  # Initial plane thickness
    plane_position=(128, 128, 128),  # Initial position of the plane in the volume
    plane_normal=(1, 0, 0),  # Plane normal vector
)
# # It helps to visualize where the plane is cutting through.
# volume = scene.visuals.Volume(
#     vol,
#     parent=view.scene,
#     raycasting_mode='volume',
#     method='mip',
# )
# volume.set_gl_state('additive')
# volume.opacity = 0.2
axis = scene.visuals.XYZAxis(parent=view.scene)
# Increase the size of the axis by adjusting the scale
axis.transform.scale = (100, 100, 100)

# Use PanZoomCamera for a flat 2D orthographic projection
cam = scene.cameras.TurntableCamera(fov=0, parent=view.scene, elevation=0, azimuth=0, interactive=False)
view.camera = cam

# Adjust the camera to fit the plane in view
cam.set_range(x=(0, 256), y=(0, 256))  # Adjust based on your volume size

def update_camera_view():
    plane_pos = plane.plane_position
    plane_normal = plane.plane_normal / np.linalg.norm(plane.plane_normal)  # Normalize the plane normal

    # Elevation is the angle above or below the xy-plane
    elev = -np.degrees(np.arctan2(plane_normal[0], plane_normal[1]))
    # module to range -90 to 90
    elev = (elev + 90) % 180 - 90
    print("Elevation: ", elev)
    cam.elevation = elev
    yz = (plane_normal[0]**2 + plane_normal[1]**2)**0.5
    
    # Azimuth is the angle in the xy-plane
    azi = np.degrees(np.arctan2(yz, plane_normal[2])) - 90
    azi = (azi + 90) % 180 - 90
    print("Azimuth: ", azi)
    cam.azimuth = azi
    
    # Center the camera on the plane's position
    cam.center = plane_pos

    print(f"Normal: {plane_normal}, Position: {plane_pos} to Elevation: {cam.elevation}, Azimuth: {cam.azimuth}")

    # Adjust the camera's range to ensure the plane fits within the view
    cam.set_range(x=(0, vol.shape[1]), y=(0, vol.shape[2]))

# Function to return the updated 2D slice
def get_2d_slice():
    return canvas.render().astype(np.uint8)[:,:,:3]

# Initialize the previous slice for comparison
prev_slice = get_2d_slice()

def update_display():
    global prev_slice
    new_slice = get_2d_slice()
    
    # Compare current slice with the previous one to detect changes
    diff = np.abs(new_slice - prev_slice)
    
    # Update only if there are differences
    prev_slice = new_slice
    
    # Show the updated slice using OpenCV
    cv2.imshow('Updated Slice', new_slice.astype(np.uint8))
    cv2.waitKey(1)  # Add a small delay to refresh the window

# Function to capture the rendered image from Vispy
def capture_image():
    img_data = get_2d_slice()  # Get the rendered image as numpy array
    # img_data = np.ascontiguousarray(img_data)
    # print(f"shape: {img_data.shape}")
    # img_data = np.flipud(img_data)  # Flip image because of coordinate system difference
    print(f"shape: {img_data.shape}")

    # Show the updated slice using OpenCV
    # cv2.imshow('Updated Slice', img_data)
    # cv2.waitKey(1)

    height, width, channels = img_data.shape
    bytes_per_line = 3 * width  # 3 channels per pixel (RGB)
    
    # Create the QImage using the correct format
    return QImage(img_data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)

# PyQt Window to display the captured image
class ImageDisplayWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vispy Render in PyQt6 Window")
        self.setGeometry(100, 100, 600, 600)

        # Set up label to display image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout to contain the label
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Timer to update the image at 30 Hz
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(33)  # Approximately 30 Hz (33 ms intervals)

        # Timer to update plane position and normal at 30 Hz
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_plane_position_and_normal)
        self.update_timer.start(33)  # Update every 33 ms for 30 Hz

        # Store cursor position (initially center of the window)
        self.cursor_pos = None

    def update_image(self):
        # Capture image from Vispy
        self.image = capture_image()

        # Draw a crosshair on the image
        self.draw_crosshair()

        # Update the QLabel with the new image
        pixmap = QPixmap.fromImage(self.image)
        self.label.setPixmap(pixmap)

    def draw_crosshair(self):
        # Create a QPainter object to draw on the QImage
        painter = QPainter(self.image)

        # Set the pen color and width for the crosshair
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(2)
        painter.setPen(pen)

        # Calculate the center of the image
        width = self.image.width()
        height = self.image.height()
        center_x = width // 2
        center_y = height // 2

        # Draw the horizontal and vertical lines
        painter.drawLine(0, center_y, width, center_y)  # Horizontal line
        painter.drawLine(center_x, 0, center_x, height)  # Vertical line

        # End the painting operation
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent):
        # Update cursor position
        self.cursor_pos = event.pos()

    def update_plane_position_and_normal(self):
        if self.cursor_pos:
            # Get image center
            width = self.label.width()
            height = self.label.height()
            center_x = width // 2
            center_y = height // 2

            # Get cursor position relative to the center
            dx = self.cursor_pos.x() - center_x
            dy = self.cursor_pos.y() - center_y

            # Normalize the vector and update plane normal
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                normalized_vector = np.array([dx / length, dy / length, 0])
                plane.plane_normal = normalized_vector

            # Update camera view after modifying the plane normal
            update_camera_view()
        
# Main function to launch the PyQt window
def main():
    # Start the Vispy rendering (offscreen)
    canvas.render()
    update_camera_view()

    # Start the PyQt6 application
    app = QApplication(sys.argv)
    window = ImageDisplayWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()