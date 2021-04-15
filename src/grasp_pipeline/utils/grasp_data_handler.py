import h5py
import os
import numpy as np

MD = 'metadata'
RS = 'recording_sessions'
GT = 'grasp_trials'
G = 'grasps'
GSL = 'grasp_success_label'
C = 'collision'
NC = 'no_collision'


class GraspDataHandlerVae:
    def __init__(self, file_path):
        assert os.path.exists(file_path)
        self.file_path = file_path

    def get_grasps_for_object(self, obj_name, outcome='positive'):
        """ Returns either all grasps for an outcome in [positive, negative, collision, all]. 
        All means all outcomes are combined and returned.
        """
        def grasps_for_outcome(file_path, outcome):
            if outcome == 'collision':
                joint_preshape_name = "desired_preshape_joint_state"
            else:
                joint_preshape_name = "true_preshape_joint_state"

            palm_poses = []
            joint_confs = []
            with h5py.File(file_path, 'r') as hdf:
                print hdf.keys()
                outcomes_gp = hdf[obj_name][outcome]
                for i, grasp in enumerate(outcomes_gp.keys()):
                    grasp_gp = outcomes_gp[grasp]
                    palm_poses.append(grasp_gp["desired_preshape_palm_mesh_frame"][()])
                    joint_confs.append(grasp_gp[joint_preshape_name][()])
                num_pos = i

            return palm_poses, joint_confs, num_pos

        if outcome == 'all':
            palm_poses = []
            joint_confs = []
            num_g = 0
            for oc in ['collision', 'negative', 'positive']:
                palms, joints, num = grasps_for_outcome(self.file_path, oc)
                palm_poses += palms
                joint_confs += joints
                num_g += num
            return palm_poses, joint_confs, num_g
        elif outcome in ['collision', 'negative', 'positive']:
            return grasps_for_outcome(self.file_path, outcome)
        else:
            raise Exception("Wrong outcome. Choose [positive, negative, collision, all]")

    def get_num_success_per_object(self):
        num_success_per_object = {}
        with h5py.File(self.file_path, 'r') as hdf:
            for obj in hdf.keys():
                num_success_per_object[obj] = len(hdf[obj]['positive'].keys())

        return num_success_per_object

    def get_single_successful_grasp(self, obj_name, random=True, idx=None):
        with h5py.File(self.file_path, 'r') as hdf:
            pos_gp = hdf[obj_name]['positive']
            grasp_ids = pos_gp.keys()
            if random:
                idx = np.random.randint(0, len(grasp_ids))
            else:
                idx = idx

            palm_pose = pos_gp[grasp_ids[idx]]["desired_preshape_palm_mesh_frame"][()]
            joint_conf = pos_gp[grasp_ids[idx]]["true_preshape_joint_state"][()]

        return palm_pose, joint_conf


class GraspDataHandler():
    def __init__(self, file_path, sess_name='-1'):
        self.file_path = file_path
        self.set_sess_name(sess_name)

    def set_sess_name(self, sess_name):
        if sess_name != '-1':
            self.sess_name = sess_name
        else:
            with h5py.File(self.file_path, "r") as grasp_file:
                self.sess_name = grasp_file[RS].keys()[-1]

    def check_sess_name(self, grasp_file):
        if self.sess_name is None:
            raise Exception('Self.sess_name not set.')
        else:
            if self.sess_name in grasp_file[RS].keys():
                return
            else:
                raise Exception('Invalid sess_name')

    ### +++++ Part I: Print dataset information +++++ ###
    def print_metadata(self):
        with h5py.File(self.file_path, "r") as grasp_file:
            metadata_gp = grasp_file[MD]
            print("\n***** All metadata information ******")
            for key in metadata_gp.keys():
                print("{:<25} {}".format(key, metadata_gp[key][()]))

            print("")

    def print_object_metadata(self, obj_name, count=False):
        with h5py.File(self.file_path, "r") as hdf:
            obj_metadata_gp = hdf[RS][self.sess_name][GT][obj_name][MD]

            print("\n***** All object metadata ******")
            for key in obj_metadata_gp.keys():
                print("{:<25} {}".format(key, obj_metadata_gp[key][()]))

            print("")

            if count:
                obj_grasp_gp = hdf[RS][self.sess_name][GT][obj_name][G][NC]
                num_pos = 0
                num_neg = 0
                for key in obj_grasp_gp.keys():
                    label = obj_grasp_gp[key]['grasp_success_label'][()]
                    if label == 1:
                        num_pos += 1
                    elif label == 0:
                        num_neg += 1
                print("Number of negative and positive grasps")
                print("{:<20} {}".format('negatives', num_neg))
                print("{:<20} {}".format('positives', num_pos))

    def print_objects(self):
        with h5py.File(self.file_path, "r") as grasp_file:
            self.check_sess_name(grasp_file)

            grasps_gp = grasp_file[RS][self.sess_name][GT]

            print("\n\n***** All object names *****")
            for key in grasps_gp.keys():
                print(key)

            print("")

            return grasps_gp.keys()

    ### +++++ Part II: Access Dataset +++++ ###
    def get_objects_list(self):
        with h5py.File(self.file_path, "r") as grasp_file:
            self.check_sess_name(grasp_file)
            grasps_gp = grasp_file[RS][self.sess_name][GT]
            return grasps_gp.keys()

    def get_successful_grasps_idxs(self, object_name):
        with h5py.File(self.file_path, "r") as grasp_file:
            self.check_sess_name(grasp_file)
            no_coll_gp = grasp_file[RS][self.sess_name][GT][object_name][G]['no_collision']

            # Build a list with all the successful grasps
            return [i + 1 for (i, grasp) in enumerate(no_coll_gp.keys()) \
                if no_coll_gp[grasp][GSL][()]]

    def get_single_successful_grasp(self, object_name, random=False, grasp_idx=-1):
        with h5py.File(self.file_path, "r") as grasp_file:
            self.check_sess_name(grasp_file)
            # Get thr group holding all non-collision grasps
            no_coll_gp = grasp_file[RS][self.sess_name][GT][object_name][G]['no_collision']

            # Build a list with all the successful grasps
            idxs = [i + 1 for (i, grasp) in enumerate(no_coll_gp.keys()) \
                if no_coll_gp[grasp][GSL][()]]

            # Select idx
            if random:
                idx = idxs[np.random.randint(0, len(idxs))]
            elif grasp_idx in idxs:
                idx = grasp_idx
            else:
                raise Exception('Given grasp_idx is invalid')

            # Get grasp group
            grasp_gp = no_coll_gp['grasp_' + str(idx).zfill(4)]

            # Build and return a dict with all information
            grasp_data = {"object_name": object_name}
            for key in grasp_gp.keys():
                grasp_data[key] = grasp_gp[key][()]

            # Grasp data holds the following keys:
            #[u'is_top_grasp', u'lifted_joint_state', u'desired_preshape_joint_state', u'desired_preshape_palm_world_pose', 'object_name', u'true_preshape_joint_state', u'closed_joint_state', u'object_world_sim_pose', u'time_stamp', u'true_preshape_palm_world_pose', u'grasp_success_label']
            return grasp_data


if __name__ == '__main__':
    file_path = os.path.join('/home/vm/', 'grasp_data.h5')
    gdh = GraspDataHandler(file_path=file_path)
    gdh.set_sess_name(sess_name='-1')

    gdh.print_metadata()
    objs = gdh.print_objects()
    # grasp_data = gdh.get_single_successful_grasp(objs[-1], random=True)
    # for key, val in grasp_data.items():
    #     print(key)
    #     print(val)
