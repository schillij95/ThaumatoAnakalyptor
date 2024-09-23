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
    return canvas.render()

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


# Keyboard interaction for changing plane properties
@canvas.events.key_press.connect
def on_key_press(event):
    if event.text == '1':
        methods = ['mip', 'average']
        method = methods[(methods.index(plane.method) + 1) % 2]
        print("Volume render method: %s" % method)
        plane.method = method
    elif event.text == '2':
        modes = ['volume', 'plane']
        plane.raycasting_mode = modes[1] if plane.raycasting_mode == modes[0] else modes[0]
        print("Switched to mode: %s" % plane.raycasting_mode)
    elif event.text in '{}':
        t = -1 if event.text == '{' else 1
        plane.plane_thickness += t
        print(f"Plane thickness: {plane.plane_thickness}")
    elif event.text in '[]':
        shift = plane.plane_normal / np.linalg.norm(plane.plane_normal)
        if event.text == '[':
            plane.plane_position -= 2 * shift
        elif event.text == ']':
            plane.plane_position += 2 * shift
        print(f"Plane position: {plane.plane_position}")
    elif event.text == 'z':
        plane.plane_normal = [0, 0, 1]
    elif event.text == 'y':
        plane.plane_normal = [0, 1, 0]
    elif event.text == 'x':
        plane.plane_normal = [1, 0, 0]
    elif event.text == 'o':
        plane.plane_normal = [1, 1, 1]
    elif event.text == 'i':
        plane.plane_normal = [0, 1, 1]
    elif event.text == 'u':
        plane.plane_normal = [2, 5, 1]
    plane.plane_normal / np.linalg.norm(plane.plane_normal)

    cam.elevation = 0
    cam.azimuth = 0
    cam.roll = 0

    update_camera_view()
    update_display()

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
