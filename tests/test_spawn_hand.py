import shutil

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler

gazebo_objects_path = '/home/vm/gazebo-objects/objects_gazebo'

tilt_objects = []

# shutil.rmtree('/home/vm/grasp_data')
grasp_client = GraspClient(is_rec_sess=True, grasp_data_recording_path='/tmp/')
metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)

pose = [0.5, 0.0, 0.2, 0, 0, 0]

while (True):

    object_metadata = metadata_handler.choose_next_grasp_object()

    grasp_client.update_object_metadata(object_metadata)

    grasp_client.spawn_hand(pose_type='init', pose_arr=pose)

    letter = raw_input("Append object to list? (Y/n): ")
    if letter == 'y':
        tilt_objects.append(object_metadata["name"])
    else:
        continue

    print(tilt_objects)
