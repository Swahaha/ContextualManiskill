import os
import sys
import xml.etree.ElementTree as ET
import tempfile
import atexit

# Global list to track temporary files
_temp_files = []

def create_scaled_urdf(original_urdf_path, link_name, scale, output_path=None):
    """
    Create a new URDF with scaled dimensions for a specific link
    
    Args:
        original_urdf_path: Path to the original URDF file
        link_name: Name of the link to scale
        scale: [x, y, z] scaling factors
        output_path: Where to save the new URDF. If None, creates a temporary file.
    
    Returns:
        Path to the new URDF file
    """
    # Parse the URDF
    tree = ET.parse(original_urdf_path)
    root = tree.getroot()
    
    # Get directory of original URDF to resolve relative paths
    urdf_dir = os.path.dirname(os.path.abspath(original_urdf_path))
    
    # Convert all mesh paths to absolute paths
    for mesh_elem in root.findall(".//mesh"):
        if "filename" in mesh_elem.attrib:
            rel_path = mesh_elem.attrib["filename"]
            if not os.path.isabs(rel_path):
                abs_path = os.path.join(urdf_dir, rel_path)
                mesh_elem.set("filename", abs_path)
    
    # Find link5
    if link_name == "panda_link5":
        # Scale the visual and collision meshes
        for link in root.findall(".//link"):
            if link.get("name") == "panda_link5":
                print(f"Found link5, applying scale: {scale}")
                
                # Scale visual meshes
                for visual in link.findall(".//visual/geometry/mesh"):
                    visual.set("scale", f"{scale[0]} {scale[1]} {scale[2]}")
                    print("Scaled visual mesh")
                
                # Scale collision meshes
                for collision in link.findall(".//collision/geometry/mesh"):
                    collision.set("scale", f"{scale[0]} {scale[1]} {scale[2]}")
                    print("Scaled collision mesh")
                
                # Update inertial properties - optional but helps with dynamics
                # inertial = link.find("inertial")
                # if inertial is not None:
                #     # Scale mass by volume increase
                #     mass_elem = inertial.find("mass")
                #     if mass_elem is not None:
                #         original_mass = float(mass_elem.get("value"))
                #         # Mass scales with volume
                #         new_mass = original_mass * scale[2]  # For z-only scaling
                #         mass_elem.set("value", str(new_mass))
                #         print(f"Updated mass from {original_mass} to {new_mass}")
                    
                    # Scale inertia tensor
                    # inertia = inertial.find("inertia")
                    # if inertia is not None:
                    #     # Simple scaling approximation for z-only scaling
                    #     for prop in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]:
                    #         if prop in inertia.attrib:
                    #             orig_val = float(inertia.get(prop))
                    #             # For z-only scaling, different components scale differently
                    #             if prop in ["ixx", "iyy"]:
                    #                 # These scale with z² and mass
                    #                 new_val = orig_val * scale[2]**3
                    #             elif prop == "izz":
                    #                 # This scales only with mass for z-direction
                    #                 new_val = orig_val * scale[2]
                    #             else:
                    #                 # Products of inertia - approximation
                    #                 new_val = orig_val * scale[2]**2
                                
                    #             inertia.set(prop, str(new_val))
                    #     print("Updated inertia tensor")
        
        # CRITICAL: Update joint6 position to match the scaled link5
        for joint in root.findall(".//joint"):
            if joint.get("name") == "panda_joint6":
                origin = joint.find("origin")
                if origin is not None:
                    # Extract current position
                    xyz_str = origin.get("xyz", "0 0 0")
                    xyz = xyz_str.split()
                    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                    
                    # Analyze joint orientations from URDF:
                    # - joint5 has -1.57 rad (90°) around x, which means:
                    #   * link5's +z becomes +y in global space
                    # - joint6 is at (0,0,0) relative to link5
                    
                    # When we scale link5 in z by 2, we need to:
                    # 1. Keep the joint at the end of the scaled link
                    # 2. Since z maps to y, we adjust y-coordinate (after rotations)
                    
                    # For a scaling factor of 2 in z, we need to add the following offset:
                    # The original link5 extends about 0.1m along its local z-axis
                    offset = 0.1 * (scale[2] - 1.0)  # Additional length
                    
                    # Apply offset - after rotation, link5's z becomes global y
                    new_y = y + offset
                    
                    # Set new position
                    origin.set("xyz", f"{x} {new_y} {z}")
                    print(f"Updated joint6 position to keep connection: {x} {new_y} {z}")
                    break
    else:
        print(f"Warning: Special handling is only implemented for panda_link5, not {link_name}")
    
    # Always use a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
    temp_file.close()  # Close it so we can write to it with ET
    output_path = temp_file.name
    
    # Register for cleanup at exit
    _temp_files.append(output_path)
    
    # Ensure cleanup is registered (only need to do this once)
    if len(_temp_files) == 1:
        atexit.register(lambda: [os.unlink(f) for f in _temp_files if os.path.exists(f)])
    
    # Write the modified URDF
    tree.write(output_path)
    print(f"Wrote scaled URDF to temporary file: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_scaled_urdf.py <urdf_path> <link_name> <scale_x> [<scale_y>] [<scale_z>]")
        sys.exit(1)
        
    urdf_path = sys.argv[1]
    link_name = sys.argv[2]
    
    # Parse scale arguments
    if len(sys.argv) >= 6:
        scale = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
    elif len(sys.argv) >= 5:
        scale = [float(sys.argv[3]), float(sys.argv[4]), 1.0]
    else:
        scale = [float(sys.argv[3]), 1.0, 1.0]
        
    create_scaled_urdf(urdf_path, link_name, scale)