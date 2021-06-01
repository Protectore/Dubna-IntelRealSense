import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

pipe = rs.pipeline()
config = rs.config()

profile = pipe.start(config)

for _ in range(5):
    pipe.wait_for_frames()

counter = 0
start = time.time()
while True:
    frame = pipe.wait_for_frames()

    color_frame = np.asanyarray(frame.get_color_frame().get_data())
    depth_frame = frame.get_depth_frame()
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    cv2.imshow('govno',color_frame)
    cv2.imshow('depth_govno', colorized_depth)
    key = cv2.waitKey(1)
    counter += 1
    if key == 27:
        break


stop = time.time()

t = stop - start
print(counter / t)
pipe.stop()
cv2.destroyAllWindows()