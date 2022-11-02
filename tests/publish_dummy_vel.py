import time
import ar_dds as dds
import numpy as np
import os

use_diana7 = int(os.environ["USE_DIANA7"])

if use_diana7:
    # Start velocity stream with ar-toolkit
    from ar_toolkit.robots import Diana7Robot
    robot = Diana7Robot("diana7")
    robot.move_ptp([0,0,0,1.57079632679,0,-1.57079632679,0])
    robot.start_linear_speed_listener(1.0/30, 1) # 30 fps, flange (0: base, 1:flange, 2:tcp)

else:
# Initialize DDS Domain Participant
    participant = dds.DomainParticipant(domain_id=0)

    # Create Domain Participant and a publisher to publish velocity
    publisher = participant.create_publisher("ar::frankenstein_legacy_interfaces::dds::robot::diana::linear_speed_servoing_v1",
                                            'des_cart_vel_msg')
rand_arr = np.random.rand(6) * np.pi

while 1:
    v_des = (0.05 * np.sin(time.time() + rand_arr)).astype(np.double)
    if use_diana7:
        robot.linear_speed_servoing(v_des)
    else:
        publisher.message["dT"] = v_des
        publisher.publish()

    print("Published ", v_des)

    time.sleep(0.033333)
