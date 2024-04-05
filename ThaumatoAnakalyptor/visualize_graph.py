import ezdxf
from ezdxf.math import Vec3

def create_dxf_with_colored_3d_polyline(filename, points, edges, color=7):
    # Create a new DXF document.
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Convert points to Vec3 objects
    vec_points = [Vec3(*point) for point in points]

    # Create polyline points by referencing the edges
    polyline_points = [vec_points[index] for edge in edges for index in edge]

    # Add the 3D polyline to the model space
    msp.add_polyline3d(polyline_points, dxfattribs={'color': color})

    # Save the DXF document
    doc.saveas(filename)

# Example usage
points = [(0, 0, 200), (300, 0, 0), (100, 100, 0), (0, 100, 50)]  # Define points as (x, y, z)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Define edges as pairs of point indices

create_dxf_with_colored_3d_polyline("example_colored_3d_polyline.dxf", points, edges, color=1)  # Color=1 is red
