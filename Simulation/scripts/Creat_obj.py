import os 
import subprocess
import sys
import zipfile

zip_file_path = ''

base_path = 'data'
particleSkinnerApp = 'ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py'
marching_cube_path = 'ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes'

for root, dirs, files in os.walk(base_path):
    dat_files = [f for f in files if f.endswith('.dat') and not f.endswith('_phi.dat')]
    #print dat_files_name
    print(dat_files)

    for dat_file in dat_files:
        dat_file_path = os.path.join(root, dat_file)
        phi_dat_file_path = os.path.join(root, os.path.splitext(dat_file)[0] + '_phi.dat')
        obj_file_path = os.path.join(root, os.path.splitext(dat_file)[0] + '.obj')

        cmd = 'python3 "' + particleSkinnerApp + '" ' + str(0.063) + ' "' + dat_file_path + '" "' + phi_dat_file_path + '" "' + obj_file_path + '" "' + marching_cube_path + '"'
        print(cmd)
        
        subprocess.run(cmd, shell=True)

#if process end print generated obj files done
print('Generated obj files done')