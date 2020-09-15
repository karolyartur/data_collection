import time
from os import system, name
from datetime import datetime
import signal
import sys
import socket
import threading
import math
import numpy as np
import cv2
import json
import re
from state import State
import struct
import pyrealsense2 as rs
import logging
import realsense_input_module as ri
import egomotion_filter_module as emf

np.set_printoptions(threshold=sys.maxsize)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C! terminating the program...')
    sys.exit(0)


def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

width = 640
height = 480
fps = 30
size = (width,height)


# Get configs for the two streams
config_1 = ri.get_config()
config_2 = ri.get_config()

# Get pipelines for the two streams
pipeline_1 = ri.get_pipeline()
pipeline_2 = ri.get_pipeline()

# Enable t265 an SR300 streams
ri.enable_stream_t265(config_1)
ri.enable_stream_sr300(config_2, width, height, fps)
config_1.enable_record_to_file('t265.bag')
config_2.enable_record_to_file('sr300.bag')

# Start pipelines
pipeline_1.start(config_1)
pipeline_2.start(config_2)



HOST = "192.168.1.107"
PORT = 30003
TIMEOUT = 2
signal.signal(signal.SIGINT, signal_handler)

f = open("robot_states.txt", "w", buffering=16384)
#t265 = open("t265.txt", "w", buffering=16384)
#depth = open("depth.txt", "w", buffering=16384)

#cap = cv2.VideoCapture(0)



states = {}
Output = []
color_img_array = []
depth_img_array = []
pose_array = []


with open('states.json') as json_file:
    states = json.load(json_file)

for iii in states['states']:
    Output.append(State(iii['description'], iii['length'],iii['radian'],
			 iii['start'], iii['valuetype'], iii['visible']))


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(TIMEOUT)
    s.connect((HOST, PORT))

    #i = 0
    try:
        while True:    

            start = time.perf_counter()
            receivedData = s.recv(16384)
 	    
            timestamp = time.time()
            byteData = bytearray(receivedData)

            for iii in Output:
                iii.value = struct.unpack(iii.valuetype, byteData[int(
                    iii.start):int(iii.start)+int(iii.length)])[0]

            #f.write(str(i) + "\n")
            #i = i + 1
            f.write("Timestamp: \t" + str(timestamp) + "\n")
            for iii in Output:
                if iii.visible == 'True':
                    print(iii)
                    f.write(str(iii) + "\n")
            time.sleep(0.033)
            #f.write(str(received_all) + "\n")
            print("\n\ntime taken {}".format(time.perf_counter()-start))

    except (KeyboardInterrupt, SystemExit):
        pass



pipeline_1.stop()
pipeline_2.stop()
sys.exit()

