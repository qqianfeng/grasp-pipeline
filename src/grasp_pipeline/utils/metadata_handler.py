import os
import rospy
from grasp_pipeline.utils.object_names_in_datasets import *

KIT_SMALL_HEIGHT_Z = ["BathDetergent", "ChoppedTomatoes"]
YCB_PI_HALF_ROLL = ["008_pudding_box", "009_gelatin_box", "035_power_drill"]


class MetadataHandler():
    """ Simple class to help iterate through objects and 
    """
    def __init__(self, gazebo_objects_path):

        # self.datasets = [BIGBIRD_OBJECTS]
        self.datasets = [BIGBIRD_OBJECTS, KIT_OBJECTS]#, YCB_OBJECTS]

        #self.datasets_name = ['bigbird']
        self.datasets_name = ['bigbird', 'kit']#, 'ycb']

        # self.datasets_name = ['bigbird']#, 'ycb']

        self.object_ix = -1
        self.dataset_ix = 0
        self.gazebo_objects_path = gazebo_objects_path

    def get_total_num_objects(self):
        # return len(YCB_OBJECTS + BIGBIRD_OBJECTS + KIT_OBJECTS)
        return len(BIGBIRD_OBJECTS + KIT_OBJECTS)
        # return len(BIGBIRD_OBJECTS)

    def choose_next_grasp_object(self,case=''):
        """ Iterates through all objects in all datasets and returns object_metadata. Gives a new object each time it is called.
        case should be either generation or postprocessing
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
                    if self.dataset_ix == 2:
                        self.dataset_ix = 0

                # Set some relevant variables
                dataset_list = self.datasets[self.dataset_ix]
                dset_name = self.datasets_name[self.dataset_ix]
                obj_name = dataset_list[self.object_ix]

                object_metadata = self.get_object_metadata(dset_name, obj_name)

                choose_success = True
                if case == 'generation':
                    if dset_name == 'bigbird' and object_metadata["name"] in BIGBIRD_OBJECTS_DATA_GENERATED:
                        choose_success = False
                    if dset_name == 'kit' and object_metadata["name"] in KIT_OBJECTS_DATA_GENERATED:
                        choose_success = False
                elif case == 'postprocessing' or case == '':
                    pass
                else:
                    raise ValueError('wrong case of', case)
                if choose_success:
                    rospy.loginfo('Trying to grasp object: %s in dataset: %s' % (object_metadata["name"], object_metadata['dataset']))

            except Exception as e:
                rospy.logerr("%s" % e)
                self.object_ix += 1

        return object_metadata

    def get_object_metadata(self, dset_name, obj_name):
        if dset_name is False:
            if obj_name in BIGBIRD_OBJECTS:
                dset_name = 'bigbird'
            elif obj_name in KIT_OBJECTS:
                dset_name = 'kit'
            else:
                raise ValueError('object name not found',obj_name)
        object_path = os.path.join(self.gazebo_objects_path, dset_name, obj_name)
        files = os.listdir(object_path)
        collision_mesh = [s for s in files if "collision" in s][0]

        # Create the final metadata dict to return
        object_metadata = dict()
        object_metadata["name"] = obj_name
        object_metadata["model_file"] = os.path.join(object_path, obj_name + '.sdf')
        object_metadata["collision_mesh_path"] = os.path.join(object_path, collision_mesh)
        object_metadata["dataset"] = dset_name
        object_metadata["name_rec_path"] = dset_name + '_' + obj_name
        object_metadata["mesh_frame_pose"] = None
        object_metadata["seg_pose"] = None
        object_metadata["aligned_pose"] = None
        object_metadata["seg_dim_whd"] = None
        object_metadata["aligned_dim_whd"] = None
        object_metadata["spawn_angle_roll"] = 0

        # Set the spawn height differently for the different datasets
        if dset_name == 'kit' or dset_name == 'ycb':
            object_metadata["spawn_height_z"] = 0.05
        elif dset_name == 'bigbird':
            object_metadata["spawn_height_z"] = 0.02
        else:
            raise Exception("Bad dataset name.")

        if dset_name == 'kit' or obj_name in YCB_PI_HALF_ROLL:
            object_metadata["spawn_angle_roll"] = 1.57079632679

        if obj_name in SPAWN_HIGH_Z:
            object_metadata["spawn_height_z"] += 0.1

        return object_metadata

    def split_full_name(self, obj_full):
        """Takes a full name in the format dataset_objectname and returns seperated dataset and objectname strings.
        """
        split_name = obj_full.split('_')
        obj_name = '_'.join(split_name[1:])
        dset_name = split_name[0]
        return dset_name, obj_name


class MetadataHandlerFinalDataGen(MetadataHandler):
    """ Simple class to help iterate through objects and 
    """
    def __init__(self, gazebo_objects_path):
        self.dset_obj_names = OBJECTS_TO_GENERATE_DATA_FOR_AFTER_15_04_Desktop

    def get_total_num_objects(self):
        return len(self.dset_obj_names)

    def choose_next_grasp_object(self):
        """ Iterates through all objects in all datasets and returns object_metadata. Gives a new object each time it is called.
        """
        choose_success = False
        while (not choose_success):
            try:
                # When this is called a new object is requested
                self.object_ix += 1

                # Set some relevant variables
                dset_obj_name = self.dset_obj_names[self.object_ix]
                dset_name, obj_name = self.split_full_name(dset_obj_name)
                assert dset_name in ['kit', 'bigbird']
                object_metadata = self.get_object_metadata(dset_name, obj_name)

                rospy.loginfo('Trying to grasp object: %s' % object_metadata["name"])
                choose_success = True

                if dset_name == 'kit' and object_metadata["name"] in KIT_OBJECTS_DATA_GENERATED:
                    choose_success = False
            except:
                self.object_ix += 1

        return object_metadata


# if __name__ == '__main__':
#     a = MetadataHandlerFinalDataGen()
#     for i in range(0, a.get_total_num_objects()):
#         obj = a.choose_next_grasp_object()
#         print(i)
