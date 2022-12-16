import h5py

path='/home/yb/new_data_16_12/grasp_data.h5'

file= h5py.File(path,'r')

data1 = file['recording_sessions']['recording_session_0001']

obj = 'bigbird_3m_high_tack_spray_adhesive'
data2 = data1['grasp_trials'][obj]['grasps']['no_collision']
#  

grasp_data = data2['grasp_0001']

print(grasp_data.keys())

print(grasp_data['obstacle1_name'][()])
print(grasp_data['obstacle2_name'][()])
print(grasp_data['obstacle3_name'][()])
print(grasp_data['true_preshape_palm_mesh_frame'][()])