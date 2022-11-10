#!/usr/bin/env python

import sys
import os
import subprocess

# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script
filter_script_inertia_mlx = """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Transform: Scale">
  <Param type="RichDynamicFloat" value="1000" min="0.1" name="axisX" max="10"/>
  <Param type="RichDynamicFloat" value="1000" min="0.1" name="axisY" max="10"/>
  <Param type="RichDynamicFloat" value="1000" min="0.1" name="axisZ" max="10"/>
  <Param type="RichBool" value="true" name="uniformFlag"/>
  <Param enum_val0="origin" enum_val1="barycenter" enum_cardinality="3" enum_val2="custom point" type="RichEnum" value="0" name="scaleCenter"/>
  <Param x="0" y="0" z="0" type="RichPoint3f" name="customCenter"/>
  <Param type="RichBool" value="false" name="unitFlag"/>
  <Param type="RichBool" value="true" name="Freeze"/>
  <Param type="RichBool" value="false" name="ToAll"/>
 </filter>
 <filter name="Compute Geometric Measures"/>
</FilterScript>
"""


def create_tmp_filter_file(filename='filter_file_tmp.mlx'):
    with open('/tmp/' + filename, 'w') as f:
        f.write(filter_script_mlx)
    return '/tmp/' + filename


def create_tmp_inertia_filter_file(filename='filter_file_inertia_tmp.mlx'):
    with open('/tmp/' + filename, 'w') as f:
        f.write(filter_script_inertia_mlx)
    return '/tmp/' + filename


def reduce_faces(in_file, out_file, filter_script_path=create_tmp_filter_file()):
    # Add input mesh
    command = "meshlabserver -i " + in_file
    # Add the filter script
    command += " -s " + filter_script_path
    # Add the output filename and output flags
    command += " -o " + out_file + " -om vn fn"
    # Execute command
    print "Going to execute: " + command
    output = subprocess.check_output(command, shell=True)
    last_line = output.splitlines()[-1]
    print
    print "Done:"
    print in_file + " > " + out_file + ": " + last_line


def compute_inertia(in_file,
                    out_folder,
                    filename,
                    filter_script_path=create_tmp_inertia_filter_file()):
    filename = filename.split('.')[0] + '.txt'
    out_full_path = out_folder + filename
    print "Output file: " + out_full_path
    # Add the input file
    command = "meshlabserver -i" + in_file
    # Add the filter script
    command += " -s " + filter_script_path

    print "Done reducing, find the file at: " + out_full_path


if __name__ == '__main__':
    in_folder = "/home/ffh/meshes_inertia/"
    out_folder = in_folder + "output/"
    os.mkdir(out_folder)
    for filename in os.listdir(in_folder):
        in_full_path = in_folder + filename
        print "Input mesh: " + in_full_path
        compute_inertia(in_full_path, out_folder, filename)