import pyrealsense2 as rs2
from frames import *
from measure import Projector
from clean_depth import DepthProcesser
from clean_depth import set_device_options
import numpy as np
from prettytable import PrettyTable


WIDTH, HEIGHT = 1280, 720
FPS = 30

pipe = rs2.pipeline()

cfg = rs2.config()
cfg.enable_stream(rs2.stream.depth, WIDTH, HEIGHT, rs2.format.z16, FPS)

cfg.enable_stream(rs2.stream.color, WIDTH, HEIGHT, rs2.format.bgr8, FPS)

profile = pipe.start(cfg)
device = profile.get_device()
set_device_options(profile)

for _ in range(5):
    pipe.wait_for_frames()

frame, depth_frame, color_frame = get_frame(pipe)

projector = Projector(device, profile, 0.11, 10)
processer = DepthProcesser()

depth_frame = processer.process(frame)
projector.set_values(depth_frame, color_frame)

depth_image, color_image = to_image_representation(depth_frame=depth_frame, color_frame=color_frame)

#projector.color_frame = color_frame
#projector.depth_frame = depth_frame
pt = PrettyTable()
pt.field_names = [i for i in range(0, 721, 10)]
row = list(range(0, 720, 10))
row.insert(0, '-')
pt.add_row(row)

for x in range(0, 1280, 10):
    temp = []
    for y in range(0, 720, 10):
        pixel = projector.color2depth((x, y))
        temp.append(pixel)

        #print(f'x = {x}, y = {y}, result = {pixel.astype(int)}')
    temp.insert(0, x)
    pt.add_row(temp)

print(pt)