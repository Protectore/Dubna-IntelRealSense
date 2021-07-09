import cv2
import numpy as np
import pyrealsense2 as rs2
from dark_net import *

pipe = rs2.pipeline()
cfg = rs2.config()
cfg.enable_stream(rs2.stream.depth, WIDTH, HEIGHT, rs2.format.z16, FPS)
cfg.enable_stream(rs2.stream.color, WIDTH, HEIGHT, rs2.format.bgr8, FPS)
profile = pipe.start(cfg)
device = profile.get_device()

set_device_options(profile)
for i in range(5):
    pipe.wait_for_frames()
frame = pipe.wait_for_frames()

depth_frame = frame.get_depth_frame()
color_frame = frame.get_color_frame()


pc = rs2.pointcloud()
points = pc.calculate(depth_frame)
pc.map_to(color_frame)

print(np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)[0])
