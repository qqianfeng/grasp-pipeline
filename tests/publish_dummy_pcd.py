import os
import time
import numpy as np
import ar_dds as dds

# DDS subscriber functions
def interrupt_signal_handler(self, _signal_number, _frame):
    """SIGINT/SIGTSTP handler for gracefully stopping an application."""
    print("Caught interrupt signal. Stop application!")
    global shutdown_requested
    shutdown_requested = True

# Init path to save pcd
enc_path = os.environ['OBJECT_PCD_ENC_PATH']
 
# Initialize DDS Domain Participant
participant = dds.DomainParticipant(domain_id=0)

# Create Domain Participant and a publisher to publish data
publisher = participant.create_publisher("ar::dds::pcd_enc::Msg",
                                                   'pcd_enc_msg')

enc_np_center = np.load(enc_path)

# Publish BPS encoded point cloud and centering data
while 1:
    publisher.message["message"] = enc_np_center[0]
    publisher.publish()
    print("Published ", enc_np_center[0])
    time.sleep(0.04)
