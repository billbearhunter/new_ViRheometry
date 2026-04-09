import os
import taichi as ti
import csv
import ctypes
import numpy as np


@ti.data_oriented
class FileOperations:
    def __init__(self):
        self.py_saved_iteration = 0
        self.py_filename = ''
        self.py_save_count = 1
        self.py_root_dir_path = 'data'
        self.py_file_processing = ''
    
    def save_data(self, data, output_dir, file_prefix, frame_count):
        csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
        dat_path = os.path.join(output_dir, f"{file_prefix}.dat")
        

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([f"{v:.16e}" for v in data])


        with open(dat_path, 'a') as dat_file:
            dat_file.write(' '.join([f"{v:.16e}" for v in data]) + '\n')

        return csv_path, dat_path
    

# @ti.data_oriented
# class FileOperations:
#     def __init__(self):
#         self.py_saved_iteration = 0
#         self.py_filename = ''
#         self.py_save_count = 1

#         self.py_root_dir_path = 'data'
#         self.py_file_processing = ''
    
    
#     def saveFile(self, agTaichiMPM, output_dir):
#         self.py_save_count  = agTaichiMPM.py_num_saved_frames
#         # print("output_dir: ", output_dir)
#         saveStateFilePath = os.path.join(output_dir, 'config_' + str(self.py_save_count).zfill(2) + ".dat")
#         saveStateIntermediateFilePath = os.path.join(output_dir, 'config_' + str(self.py_save_count).zfill(2) + "_phi" + ".dat")
#         outObjFilePath = os.path.join(output_dir, 'config_' + str(self.py_save_count).zfill(2) + ".obj")
#         particleSkinnerApp = 'ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py'
#         marching_cube_path = 'ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes'
 
#         for filepath in [saveStateFilePath, saveStateIntermediateFilePath, outObjFilePath]:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
        
#         # marching_cube_path = os.path.join('ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes')
#         # agTaichiMPM.particleSkinnerApp = os.path.join('ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py')

#         print('[AGTaichiMPM] saving state to ' + saveStateFilePath)
#         f = open(saveStateFilePath, 'wb')
#         particle_is_inner_of_box_id = np.where(agTaichiMPM.ti_particle_is_inner_of_box.to_numpy()[0:agTaichiMPM.ti_particle_count[None]].astype(np.int32) == 1)
#         f.write(ctypes.c_int32(agTaichiMPM.ti_particle_count[None] -  particle_is_inner_of_box_id[0].size))
#         #output x
#         p_x = agTaichiMPM.ti_particle_x.to_numpy()[0:agTaichiMPM.ti_particle_count[None]].astype(np.float32)
#         np.delete(p_x, particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
#         #output radius
#         np.delete((np.ones(agTaichiMPM.ti_particle_count[None], np.float32) * agTaichiMPM.py_particle_hl).astype(np.float32), particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
#         #output velocity
#         np.delete(agTaichiMPM.ti_particle_v.to_numpy()[0:agTaichiMPM.ti_particle_count[None]].astype(np.float32), particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
#         #output id
#         np.delete(np.ones(agTaichiMPM.ti_particle_count[None], ctypes.c_int32), particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
#         f.close()


@ti.data_oriented
class FileOperations:
    def __init__(self):
        self.py_saved_iteration = 0
        self.py_filename = ''
        self.py_save_count = 1
        self.py_root_dir_path = 'data'
        self.py_file_processing = ''
    
    def saveFile(self, agTaichiMPM, output_dir):
        """
        Save particle data to .dat file and prepare for skinning process
        
        Parameters:
        agTaichiMPM: MPM simulator object
        output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        self.py_save_count = agTaichiMPM.py_num_saved_frames
        
        # Create file paths
        saveStateFilePath = os.path.join(output_dir, f'config_{self.py_save_count:02d}.dat')
        saveStateIntermediateFilePath = os.path.join(output_dir, f'config_{self.py_save_count:02d}_phi.dat')
        outObjFilePath = os.path.join(output_dir, f'config_{self.py_save_count:02d}.obj')
        
        # Define skinning paths
        particleSkinnerApp = 'ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py'
        marching_cube_path = 'ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes'
 
        # Clean up existing files
        for filepath in [saveStateFilePath, saveStateIntermediateFilePath, outObjFilePath]:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Print status message
        print(f'[AGTaichiMPM] saving state to {saveStateFilePath}')
        
        # Generate particle data file
        self._generate_particle_dat_file(agTaichiMPM, saveStateFilePath)

        # Prepare for skinning process
        self.generate_obj_file(agTaichiMPM, outObjFilePath, saveStateFilePath, saveStateIntermediateFilePath, marching_cube_path)

        
        # Note: External skinning command remains commented out as in original
        # cmd = f'python3 "{particleSkinnerApp}" ...'
        # os.system(cmd)

    def _generate_particle_dat_file(self, agTaichiMPM, filepath):
        """
        Generate .dat file with particle data (internal helper function)
        
        Parameters:
        agTaichiMPM: MPM simulator object
        filepath: Full path to output file
        """
        # Get particle count
        particle_count = agTaichiMPM.ti_particle_count[None]
        
        # Identify valid particles (not inside static boxes)
        particle_is_inner_of_box = agTaichiMPM.ti_particle_is_inner_of_box.to_numpy()[:particle_count]
        valid_indices = np.where(particle_is_inner_of_box == 0)[0]
        valid_count = len(valid_indices)
        
        # Open file for binary writing
        with open(filepath, 'wb') as f:
            # Write number of valid particles (int32)
            f.write(ctypes.c_int32(valid_count))
            
            # Write particle positions (x,y,z)
            particle_positions = agTaichiMPM.ti_particle_x.to_numpy()[:particle_count]
            valid_positions = particle_positions[valid_indices].astype(np.float32).flatten()
            valid_positions.tofile(f)
            
            # Write particle radii
            radius_value = agTaichiMPM.py_particle_hl
            particle_radii = np.full(valid_count, radius_value, dtype=np.float32)
            particle_radii.tofile(f)
            
            # Write particle velocities (vx, vy, vz)
            particle_velocities = agTaichiMPM.ti_particle_v.to_numpy()[:particle_count]
            valid_velocities = particle_velocities[valid_indices].astype(np.float32).flatten()
            valid_velocities.tofile(f)
            
            # Write particle IDs (all set to 1)
            particle_ids = np.ones(valid_count, dtype=ctypes.c_int32)
            particle_ids.tofile(f)

    def generate_obj_file(self, agTaichiMPM, outObjFilePath,saveStateFilePath, saveStateIntermediateFilePath, marching_cube_path):
        """
        Generate .obj file from particle data
        Parameters:
        agTaichiMPM: MPM simulator object
        output_dir: Output directory path
        """

        # Generate the .obj file using the particle skinning process
        particleSkinnerApp = 'ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py'

        cmd = f'python3 "{particleSkinnerApp}" "{0.05}" "{saveStateFilePath}" "{saveStateIntermediateFilePath}" "{outObjFilePath}" "{marching_cube_path}"'
        print(cmd)

        print(f'[AGTaichiMPM] generating OBJ file: {outObjFilePath}')
        os.system(cmd)  