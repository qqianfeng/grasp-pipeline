import os
import rospy
from grasp_pipeline.utils.object_names_in_datasets import *

KIT_SMALL_HEIGHT_Z = ["BathDetergent", "ChoppedTomatoes"]
YCB_PI_HALF_ROLL = ["008_pudding_box", "009_gelatin_box", "035_power_drill"]


class MetadataHandler():
    """ Simple class to help iterate through objects and 
    """
    def __init__(self, gazebo_objects_path='/home/vm/gazebo-objects/objects_gazebo'):
        #self.datasets = [BIGBIRD_OBJECTS]
        self.datasets = [KIT_OBJECTS, YCB_OBJECTS, BIGBIRD_OBJECTS]

        #self.datasets_name = ['bigbird']
        self.datasets_name = ['kit', 'ycb', 'bigbird']

        self.object_ix = -1
        self.dataset_ix = 0
        self.gazebo_objects_path = gazebo_objects_path

    def get_total_num_objects(self):
        return len(YCB_OBJECTS + BIGBIRD_OBJECTS + KIT_OBJECTS)

    def choose_next_grasp_object(self):
        """ Iterates through all objects in all datasets and returns object_metadata. Gives a new object each time it is called.
        """
        choose_success = False
        while (not choose_success):
            try:
                # When this is called a new object is requested
                self.object_ix += 1

                # Check if we are past the last object of the dataset. If so take next dataset
                if self.object_ix == len(self.datasets[self.dataset_ix]):
                    self.object_ix = 0
                    self.dataset_ix += 1
                    if self.dataset_ix == 3:
                        self.dataset_ix = 0

                # Set some relevant variables
                dataset_list = self.datasets[self.dataset_ix]
                dataset_name = self.datasets_name[self.dataset_ix]
                object_name = dataset_list[self.object_ix]

                object_metadata = self.get_object_metadata(dataset_name, object_name)

                rospy.loginfo('Trying to grasp object: %s' % object_metadata["name"])
                choose_success = True

                if dataset_name == 'kit' and object_metadata["name"] in KIT_OBJECTS_DATA_GENERATED:
                    choose_success = False
            except:
                self.object_ix += 1

        return object_metadata

    def get_object_metadata(self, dataset_name, object_name):
        object_path = os.path.join(self.gazebo_objects_path, dataset_name, object_name)
        files = os.listdir(object_path)
        collision_mesh = [s for s in files if "collision" in s][0]

        # Create the final metadata dict to return
        object_metadata = dict()
        object_metadata["name"] = object_name
        object_metadata["model_file"] = os.path.join(object_path, object_name + '.sdf')
        object_metadata["collision_mesh_path"] = os.path.join(object_path, collision_mesh)
        object_metadata["dataset"] = dataset_name
        object_metadata["name_rec_path"] = dataset_name + '_' + object_name
        object_metadata["mesh_frame_pose"] = None
        object_metadata["seg_pose"] = None
        object_metadata["aligned_pose"] = None
        object_metadata["seg_dim_whd"] = None
        object_metadata["aligned_dim_whd"] = None
        object_metadata["spawn_angle_roll"] = 0

        # Set the spawn height differently for the different datasets
        if dataset_name == 'kit' or dataset_name == 'ycb':
            object_metadata["spawn_height_z"] = 0.05
        elif dataset_name == 'bigbird':
            object_metadata["spawn_height_z"] = 0.02
        else:
            raise Exception("Bad dataset name.")

        if dataset_name == 'kit' or object_name in YCB_PI_HALF_ROLL:
            object_metadata["spawn_angle_roll"] = 1.57079632679

        return object_metadata
