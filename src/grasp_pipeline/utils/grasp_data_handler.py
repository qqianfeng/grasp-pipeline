import h5py
import os
import numpy as np

MD = 'metadata'
RS = 'recording_sessions'
GT = 'grasp_trials'
G = 'grasps'
GSL = 'grasp_success_label'


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

    def print_objects(self):
        with h5py.File(self.file_path, "r") as grasp_file:
            self.check_sess_name(grasp_file)

            grasps_gp = grasp_file[RS][self.sess_name][GT]

            print("\n\n***** All object names *****")
            for key in grasps_gp.keys():
                print(key)

            print("")

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
    file_path = os.path.join('/home/vm', 'grasp_data.h5')
    gdh = GraspDataHandler(file_path=file_path)
    gdh.set_sess_name(sess_name='-1')

    gdh.print_metadata()
    gdh.print_objects()
    grasp_data = gdh.get_single_successful_grasp('ycb_008_pudding_box', random=True)
    print(grasp_data["object_name"])