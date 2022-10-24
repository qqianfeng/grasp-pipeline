import time
import ar_dds as dds
import numpy as np

# Initialize DDS Domain Participant
participant = dds.DomainParticipant(domain_id=0)

# Create Domain Participant and a publisher to publish velocity
publisher = participant.create_publisher("ar::frankenstein_legacy_interfaces::dds::robot::diana::linear_speed_servoing_v1",
                                         'des_cart_vel_msg')
rand_arr = np.random.rand(6) * np.pi

while 1:
    v_des = (0.1 * np.sin(time.time() + rand_arr)).astype(np.double)
    publisher.message["dT"] = v_des
    publisher.publish()

    print("Published ", v_des)

    time.sleep(0.02)
