import pyrealsense2 as rs
from ctypes import *
import tkinter as tk
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

netMain = None
metaMain = None
altNames = None

def stringize(num):
    return str(int(float(num)))

configPath = "./cfg/yolov3.cfg"
weightPath = "./yolov3.weights"
metaPath = "./cfg/coco.data"
if not os.path.exists(configPath):
    raise ValueError("Invalid config path `" +
                        os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
    raise ValueError("Invalid weight path `" +
                        os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
    raise ValueError("Invalid data file path `" +
                        os.path.abspath(metaPath)+"`")
if netMain is None:
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
if metaMain is None:
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
if altNames is None:
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                                re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass
    
start_time = time.time()
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
end_time = time.time()
print("camera ready: ", end_time - start_time)
pipeline.start(config)
print("Starting the multi-threading loop...")
darknet_image = darknet.make_image(darknet.network_width(netMain), 
                                darknet.network_height(netMain),3)

root = tk.Tk()
root.title("Darknet GUI Demo")
text = tk.StringVar()
text.set("Waiting for detections")
tk.Label(root, textvariable = text).pack()

try:
    while True:
        prev_time = time.time()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        frame_rgb = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.15)
        detection = ""
        for detect in detections:
            obj = str(detect)
            obj = obj.replace('(', '').replace(')', '').replace("b'", "").replace("'", "")
            obj = obj.split(',')
            detection += "Object detected: " + str(obj[0]) + "\nCoordinates: (" + stringize(obj[2]) + ", " + stringize(obj[3]) + ", " + stringize(obj[4]) + ", " + stringize(obj[5]) + "\nConfidence: " + str(obj[1] + "\n")
            dist = depth_frame.get_distance(int(float(obj[2])), int(float(obj[3])))
            detection += "Estimated distance from camera " + str(dist) + " \n"
            root.update_idletasks()
            # time.sleep(1)
        print(time.time()-prev_time)
        text.set(detection)

    cap.release()
finally:
    pipeline.stop()
