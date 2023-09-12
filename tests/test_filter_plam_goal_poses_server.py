# only call the filter_plam_goal_pose service, assume the scene is ready (objects are spawned, robots are ready

from grasp_pipeline.utils import utils
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient

grasp_client = GraspClient(is_rec_sess=False)

quat = [0,0,0,1]
transl = [0.4,-0.1,0.05]
grasp_pose = utils.get_pose_stamped_from_trans_and_quat(transl,quat)

collision = grasp_client.filter_palm_goal_poses_client([grasp_pose])
