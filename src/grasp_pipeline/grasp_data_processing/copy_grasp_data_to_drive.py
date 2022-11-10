import os
from shutil import copyfile

if __name__ == '__main__':
    base_path = '/home/ffh/Documents'
    dst_base_path = '/media/vm/9CAF-2E7B'

    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)

        # copy the grasp_data.h5 file from documents folder to usb stick
        grasp_file_path = os.path.join(path, 'grasp_data.h5')
        dst_path = os.path.join(dst_base_path, folder)
        grasp_file_dst_path = os.path.join(dst_path, 'grasp_data.h5')
        os.mkdir(dst_path)
        copyfile(grasp_file_path, grasp_file_dst_path)

        # copy timing file
        grasp_timing_path = os.path.join(path, 'grasp_timing.txt')
        if os.path.exists(grasp_timing_path):
            grasp_timing_dst_path = os.path.join(dst_path, 'grasp_timing.txt')
            copyfile(grasp_timing_path, grasp_timing_dst_path)
