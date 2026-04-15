import os
import subprocess
import csv
import glob
import re
import xml.etree.ElementTree as ET

def _detect_gl_render_path():
    """Auto-detect GLRender3d binary for the current platform."""
    import platform
    _sim_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    if platform.system() == "Windows":
        candidates = [
            os.path.join(_sim_root, "GLRender3d", "build_win3", "Release", "GLRender3d.exe"),
            os.path.join(_sim_root, "GLRender3d", "build", "GLRender3d.exe"),
        ]
    else:
        candidates = [
            os.path.join(_sim_root, "GLRender3d", "build", "GLRender3d"),
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback to relative path
    return "GLRender3d/build/GLRender3d"


class MPMEmulator:
    def __init__(self,
                 base_path='results',
                 GL_render_path=None,
                 xml_config_path="config/setting.xml"):
        """
        OBJ renderer with config-specific displacement value passing and XML camera config.
        
        Args:
            base_path: Root directory containing simulation results
            GL_render_path: Path to rendering executable
            xml_config_path: Path to the XML configuration file
        """
        self.base_path = base_path
        self.GL_render_path = GL_render_path or _detect_gl_render_path()
        self.xml_config_path = xml_config_path
        
        # Load camera configuration from XML
        self._load_camera_config()
    
    def _load_camera_config(self):
        """Parse camera settings from the XML configuration file"""
        if not os.path.exists(self.xml_config_path):
            print(f"Warning: Config file {self.xml_config_path} not found. Using defaults.")
            self.py_camera_position = "0.0,20.0,2.0"
            self.py_camera_quat = "1.0,0.0,0.0,0.0"
            self.py_camera_window = "960,540"
            self.py_camera_fov = "50.0"
            return

        try:
            tree = ET.parse(self.xml_config_path)
            root = tree.getroot()
            
            # Navigate to GLRender -> camera
            gl_render_node = root.find('GLRender')
            if gl_render_node is None:
                raise ValueError("No <GLRender> tag found in XML")
                
            camera_node = gl_render_node.find('camera')
            if camera_node is None:
                raise ValueError("No <camera> tag found under <GLRender>")
            
            # Extract attributes and convert space-separated values to comma-separated
            # XML format: "x y z" -> C++ format: "x,y,z"
            
            eyepos = camera_node.attrib['eyepos'].strip().replace(' ', ',')
            quat = camera_node.attrib['quat'].strip().replace(' ', ',')
            window_size = camera_node.attrib['window_size'].strip().replace(' ', ',')
            fov = camera_node.attrib['fov'].strip()

            self.py_camera_position = eyepos
            self.py_camera_quat = quat
            self.py_camera_window = window_size
            self.py_camera_fov = fov
            
            print(f"Loaded Camera Config from {self.xml_config_path}:")
            print(f"  Position: {self.py_camera_position}")
            print(f"  Quat:     {self.py_camera_quat}")
            print(f"  Window:   {self.py_camera_window}")
            print(f"  FOV:      {self.py_camera_fov}")
            
        except Exception as e:
            print(f"Error loading XML config: {e}. Using hardcoded defaults.")
            # Fallback defaults
            self.py_camera_position = "0.0,20.0,2.0"
            self.py_camera_quat = "1.0,0.0,0.0,0.0"
            self.py_camera_window = "960,540"
            self.py_camera_fov = "50.0"

    def _generate_render_command(self, obj_path, x_value):
        """Generate rendering command with specific displacement value"""
        out_png = obj_path.replace('.obj', '.png')
        
        # Construct the command line arguments
        # Note: We wrap paths in quotes to handle spaces
        cmd = (f'"{self.GL_render_path}" '
               f'-a "{out_png}" '       # Output file
               f'-b "{obj_path}" '      # Input OBJ
               f'-c {self.py_camera_position} '
               f'-d {self.py_camera_quat} '
               f'-e {self.py_camera_fov} '
               f'-f {self.py_camera_window} '
               f'-g "{x_value}"')       # Displacement
        return cmd
    
    def _get_csv_paths(self, run_dir=None):
        """Find simulation_results.csv files.

        If run_dir is given, only check that directory.
        Otherwise scan all run_* and validation_* directories under base_path.
        """
        if run_dir is not None:
            csv_path = os.path.join(run_dir, "simulation_results.csv")
            return [csv_path] if os.path.exists(csv_path) else []

        candidate_dirs = (glob.glob(os.path.join(self.base_path, 'run_*')) +
                          glob.glob(os.path.join(self.base_path, 'validation_*')))
        csv_paths = []
        for d in candidate_dirs:
            csv_path = os.path.join(d, "simulation_results.csv")
            if os.path.exists(csv_path):
                csv_paths.append(csv_path)
        return csv_paths
    
    def _get_displacement_value(self, obj_filename, displacements, width):
        """
        Get displacement value based on OBJ filename configuration
        
        Mapping:
        - config_00.obj -> 0.0 (Initial state)
        - config_01.obj -> x1 (displacement 1 + width)
        - ...
        - config_08.obj -> x8 (displacement 8 + width)
        """
        # Extract configuration number from filename (e.g., config_05.obj)
        match = re.search(r'config_(\d{2})', obj_filename)
        if match:
            config_num = match.group(1)
            
            # Frame 0 is the initial state
            if config_num == '00':
                return width
                
            # Frames 1-8 correspond to the displacement list indices 0-7
            # We add 'width' because the displacement is relative to the initial position
            elif config_num in ['01','02','03','04','05','06','07','08']:
                idx = int(config_num) - 1
                if idx < len(displacements):
                    try:
                        val = float(displacements[idx]) + float(width)
                        return f"{val:.4f}"
                    except ValueError:
                        return width
                        
        return width
    
    def _parse_csv_row(self, row):
        """Parse CSV row and extract parameter information"""
        try:
            # CSV structure from main.py:
            # [n, eta, sigma_y, width, height, x_01, ..., x_08]
            n = row[0]
            eta = row[1]
            sigma_y = row[2]
            width = row[3]
            height = row[4]
            displacements = row[5:13]  # x_01 to x_08
            
            # Reconstruct the sample directory name
            # Must match the format in main.py: "{n:.2f}_{eta:.2f}_{sigma_y:.2f}"
            sample_dir_name = f"{float(n):.2f}_{float(eta):.2f}_{float(sigma_y):.2f}"
            
            return sample_dir_name, displacements, width
        except Exception as e:
            print(f"Error parsing CSV row: {e}")
            return None, None, None
    
    def render_all(self, run_dir=None):
        """Render OBJ files with config-specific displacement values.

        Args:
            run_dir: If given, only render this specific directory.
                     If None, renders all run_* and validation_* directories.
        """
        # 1. Find all CSV files (one per run)
        csv_paths = self._get_csv_paths(run_dir=run_dir)
        if not csv_paths:
            print(f"No simulation_results.csv files found in {self.base_path}")
            return
        
        processed_count = 0
        
        # 2. Iterate through each run
        for csv_path in csv_paths:
            run_dir = os.path.dirname(csv_path)
            print(f"Processing run directory: {os.path.basename(run_dir)}")
            
            try:
                with open(csv_path, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    headers = next(csv_reader, None)  # Skip header row
                    
                    # 3. Iterate through each sample in the CSV
                    for row_idx, row in enumerate(csv_reader):
                        if not row: continue
                        
                        # Parse row data
                        sample_dir_name, displacements, width = self._parse_csv_row(row)
                        if not sample_dir_name:
                            continue
                            
                        # Locate the specific sample directory (e.g., results/run_.../0.65_2.71_8.40)
                        sample_dir = os.path.join(run_dir, sample_dir_name)
                        if not os.path.exists(sample_dir):
                            print(f"  [Skipped] Sample directory missing: {sample_dir_name}")
                            continue
                        
                        # Find all OBJ files for this sample
                        obj_files = glob.glob(os.path.join(sample_dir, "*.obj"))
                        if not obj_files:
                            print(f"  [Skipped] No OBJ files in: {sample_dir_name}")
                            continue
                        
                        print(f"  Rendering sample: {sample_dir_name} ({len(obj_files)} files)")
                        
                        # 4. Render each OBJ file
                        for obj_path in obj_files:
                            obj_filename = os.path.basename(obj_path)
                            
                            # Calculate the correct displacement for this specific frame
                            x_value = self._get_displacement_value(obj_filename, displacements, width)

                            # Generate and execute the render command
                            cmd = self._generate_render_command(obj_path, x_value)
                            
                            # print(f"    Exec: {cmd}") # Debug print
                            ret = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            if ret.returncode != 0:
                                print(f"    Error rendering {obj_filename}")
                            else:
                                processed_count += 1
                                
            except Exception as e:
                print(f"Error reading {csv_path}: {str(e)}")
        
        print(f"\nRendering workflow finished. Total images generated: {processed_count}")

if __name__ == '__main__':
    # This block allows you to run this script independently to re-render existing results
    print("Starting independent rendering process...")
    renderer = MPMEmulator()
    if os.path.exists(renderer.GL_render_path):
        renderer.render_all()
    else:
        print(f"Error: Renderer not found at {renderer.GL_render_path}")
        print("Please compile GLRender3d first.")