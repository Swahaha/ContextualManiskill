import os
import sys
import xml.etree.ElementTree as ET
import tempfile
import atexit

# Global list to track temporary files
_temp_files = []

def create_scaled_urdf(original_urdf_path, link_name, scale, output_path=None):
    """
    Create a new URDF with scaled dimensions for a specific link.
    
    This revised version dynamically adjusts the connection joints so that
    the scaled link (panda_link5) remains properly attached to link4 and link6.
    It assumes that only the z-axis of link5 is being scaled (thus changing its length).
    
    Additionally, it adjusts joint5 to help move link5 upward relative to link4,
    reducing the overlap between the two.
    
    Args:
        original_urdf_path: Path to the original URDF file.
        link_name: Name of the link to scale (currently only "panda_link5" is supported).
        scale: [x, y, z] scaling factors (we assume only z is different from 1).
        output_path: Where to save the new URDF. If None, creates a temporary file.
    
    Returns:
        Path to the new URDF file.
    """
    # Parse the URDF
    tree = ET.parse(original_urdf_path)
    root = tree.getroot()
    
    # Get directory of original URDF to resolve relative paths
    urdf_dir = os.path.dirname(os.path.abspath(original_urdf_path))
    
    # FIRST: Remove collision elements from links 0-5 (if desired)
    # for i in range(6):  # Links 0 through 5
    target_link_name = f"panda_link{5}"
    for link in root.findall(".//link"):
        if link.get("name") == target_link_name:
            collisions = link.findall("collision")
            for collision in collisions:
                link.remove(collision)
            print(f"Removed all collision elements from {target_link_name}")
    
    # Convert all mesh paths to absolute paths
    for mesh_elem in root.findall(".//mesh"):
        if "filename" in mesh_elem.attrib:
            rel_path = mesh_elem.attrib["filename"]
            if not os.path.isabs(rel_path):
                abs_path = os.path.join(urdf_dir, rel_path)
                mesh_elem.set("filename", abs_path)
    
    if link_name == "panda_link5":
        # --- Determine original effective length of link5 ---
        default_length = 0.088  # default approximation in meters
        link5_length = default_length
        for joint in root.findall(".//joint"):
            if joint.get("name") == "panda_joint6":
                origin = joint.find("origin")
                if origin is not None and "xyz" in origin.attrib:
                    xyz = origin.get("xyz").split()
                    measured = abs(float(xyz[1]))  # assume local y holds the length
                    if measured > 1e-5:
                        link5_length = measured
                        print(f"Determined link5 length from joint6: {link5_length} m")
                    else:
                        print("Joint6 origin y is zero; using default link5 length.")
                break
        
        # --- Scale the visual and collision meshes for link5 ---
        for link in root.findall(".//link"):
            if link.get("name") == "panda_link5":
                print(f"Found {link_name}, applying scale: {scale}")
                for visual in link.findall(".//visual/geometry/mesh"):
                    visual.set("scale", f"{scale[0]} {scale[1]} {scale[2]}")
                    print("Scaled visual mesh for link5")
                for collision in link.findall(".//collision/geometry/mesh"):
                    collision.set("scale", f"{scale[0]} {scale[1]} {scale[2]}")
                    print("Scaled collision mesh for link5")
                break
        
        # --- Compute extra length added due to z-scaling ---
        extra_length = link5_length * (scale[2] - 1.0)
        print(f"Computed extra length for link5: {extra_length} m (original_length: {link5_length}, scale_z: {scale[2]})")
        
        # --- Adjust joint6 (connection between link5 and link6) ---
        # Place joint6 at the tip of the scaled link5.
        # new_joint6_y = link5_length * scale[2]
        new_joint6_y = -0.01 * (scale[2] - 1.0)
        for joint in root.findall(".//joint"):
            if joint.get("name") == "panda_joint6":
                origin = joint.find("origin")
                if origin is not None:
                    origin.set("xyz", f"0 {new_joint6_y} 0")
                    print(f"Adjusted joint6 origin to: 0 {new_joint6_y} 0")
                break
        
        # --- Adjust link5's internal geometry origins ---
        # To keep the connection at joint5 (between link4 and link5) fixed, we need to shift
        # the internal geometry downward in link5's local frame by the full extra length.
        print(f"Shifting link5 geometry origins by {extra_length} m in local y")
        for link in root.findall(".//link"):
            if link.get("name") == "panda_link5":
                for visual in link.findall("visual"):
                    origin = visual.find("origin")
                    if origin is not None and "xyz" in origin.attrib:
                        xyz_vals = [float(v) for v in origin.get("xyz").split()]
                        xyz_vals[1] -= extra_length
                        origin.set("xyz", " ".join(str(v) for v in xyz_vals))
                        print(f"Shifted visual origin for panda_link5 by -{extra_length} m in local y")
                for collision in link.findall("collision"):
                    origin = collision.find("origin")
                    if origin is not None and "xyz" in origin.attrib:
                        xyz_vals = [float(v) for v in origin.get("xyz").split()]
                        xyz_vals[1] -= extra_length
                        origin.set("xyz", " ".join(str(v) for v in xyz_vals))
                        print(f"Shifted collision origin for panda_link5 by -{extra_length} m in local y")
                break
        
        # --- Adjust joint5 (connection between link4 and link5) ---
        # Move joint5 upward relative to link4 to further alleviate overlap.
        # For instance, add a fixed offset (tunable) to joint5's y coordinate.
        extra_joint5_offset = 0.2*(scale[2]-1.0) # This value may need further tuning.
        for joint in root.findall(".//joint"):
            if joint.get("name") == "panda_joint5":
                origin = joint.find("origin")
                if origin is not None and "xyz" in origin.attrib:
                    xyz_vals = [float(v) for v in origin.get("xyz").split()]
                    xyz_vals[1] += extra_joint5_offset
                    origin.set("xyz", " ".join(str(v) for v in xyz_vals))
                    print(f"Adjusted joint5 origin by adding {extra_joint5_offset} m to local y")
                break
    else:
        print(f"Warning: Special handling is only implemented for panda_link5, not {link_name}")
    
    # Write to a temporary file if output_path is not provided
    temp_file = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
    temp_file.close()
    output_path = temp_file.name
    
    _temp_files.append(output_path)
    if len(_temp_files) == 1:
        atexit.register(lambda: [os.unlink(f) for f in _temp_files if os.path.exists(f)])
    
    tree.write(output_path)
    print(f"Wrote scaled URDF to temporary file: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_scaled_urdf.py <urdf_path> <link_name> <scale_x> [<scale_y>] [<scale_z>]")
        sys.exit(1)
        
    urdf_path = sys.argv[1]
    link_name = sys.argv[2]
    
    if len(sys.argv) >= 6:
        scale = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
    elif len(sys.argv) >= 5:
        scale = [float(sys.argv[3]), 1.0, float(sys.argv[4])]
    else:
        scale = [float(sys.argv[3]), 1.0, 1.0]
        
    create_scaled_urdf(urdf_path, link_name, scale)