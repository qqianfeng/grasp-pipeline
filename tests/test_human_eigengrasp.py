import rospy
from sensor_msgs.msg import JointState
rospy.init_node('test')

pub = rospy.Publisher("/hithand/joint_cmd", JointState, queue_size=1)
grasp_types = []

# Index, Little, Middle, Ring, Thumb
########## Type 1: thumb abduction min
js = JointState()
js.position = [ 0.0, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0, \
                -0.26179, 0.0, 0.0, 0.0 ]
grasp_types.append(js)
# max
js = JointState()
js.position = [ 0.0, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0, \
                0.26179, 0.0, 0.0, 0.0 ]
grasp_types.append(js)

# ########## Type 2: Finger spread min
js = JointState()
js.position = [ -0.26179, 0.0, 0.0, 0.0, \
                0.26179, 0.0, 0.0, 0.0, \
                -0.13, 0.0, 0.0, 0.0, \
                0.13, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0 ]
grasp_types.append(js)
# max
js = JointState()
js.position = [ 0.09, 0.0, 0.0, 0.0, \
                -0.09, 0.0, 0.0, 0.0, \
                0.03, 0.0, 0.0, 0.0, \
                -0.03, 0.0, 0.0, 0.0, \
                0.0, 0.0, 0.0, 0.0 ]
grasp_types.append(js)

# ########## Type 3: MCP min
js = JointState()
js.position = [ 0.0, 0.087266, 0.0, 0.0, \
                0.0, 0.087266, 0.0, 0.0, \
                0.0, 0.087266, 0.0, 0.0, \
                0.0, 0.087266, 0.0, 0.0, \
                0.0, 0.087266, 0.0, 0.0 ]
grasp_types.append(js)
# max
js = JointState()
js.position = [ 0.0, 1, 0.0, 0.0, \
                0.0, 1, 0.0, 0.0, \
                0.0, 1, 0.0, 0.0, \
                0.0, 1, 0.0, 0.0, \
                0.0, 0.7, 0.0, 0.0 ]
grasp_types.append(js)

# ########## Type 3: PIP min
js = JointState()
js.position = [ 0.0, 0.0, 0.087266, 0.087266, \
                0.0, 0.0, 0.087266, 0.087266, \
                0.0, 0.0, 0.087266, 0.087266, \
                0.0, 0.0, 0.087266, 0.087266, \
                0.0, 0.0, 0.087266, 0.087266 ]
grasp_types.append(js)
# max
js = JointState()
js.position = [ 0.0, 0.0, 1, 1, \
                0.0, 0.0, 1, 1, \
                0.0, 0.0, 1, 1, \
                0.0, 0.0, 1, 1, \
                0.0, 0.0, 1, 1 ]
grasp_types.append(js)

for i in range(3):
    pub.publish(js)
    rospy.sleep(0.2)