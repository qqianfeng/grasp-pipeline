#!/usr/bin/env python
import signal
import sys
import trollius
from trollius import From
import os

import pygazebo
import pygazebo.msg.v9.contacts_pb2
import pickle

home_folder = os.path.expanduser('~')
coll_flag_path = os.path.join(home_folder,'collision_flag.pickle')
loop = trollius.get_event_loop()

def write_to_file(bool):
    with open(coll_flag_path, 'w') as file:
        print('write to file with:', bool)
        b = pickle.dump(bool, file)

@trollius.coroutine
def publish_loop():
    manager = yield From(pygazebo.connect())

    def callback(data):
        message = pygazebo.msg.contacts_pb2.Contacts.FromString(data)
        # print(len(message.contact))
        flag = False
        for idx in range(len(message.contact)):
            if message.contact[idx].collision1[:4]=='hand' or message.contact[idx].collision2[:4]=='hand':
                flag = True
        write_to_file(flag)

    subscriber = manager.subscribe('/gazebo/default/physics/contacts',
                     'gazebo.msgs.Contacts',
                     callback)

    yield From(subscriber.wait_for_connection())
    yield From(trollius.sleep(1.00))
    # print('wait...')

def check_collision():
    if os.path.exists(coll_flag_path):
        with open(coll_flag_path, 'rb') as file:
            b = pickle.load(file)
            return b
    return False

def get_contact():
    loop.run_until_complete(publish_loop())

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    get_contact()
