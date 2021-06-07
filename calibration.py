import pyrealsense2 as rs2
import numpy as np
import cv2

# Once started successfully, dev object can be casted to rs2.auto_calibrated_device by calling:
pipe = rs2.pipeline()
cfg = rs2.config()
cfg.enable_stream(rs2.stream.depth, 256, 144, rs2.format.z16, 90)
dev = pipe.start(cfg).get_device()

# Once started successfully, dev object can be casted to rs2.auto_calibrated_device by calling:
cal = rs2.auto_calibrated_device(dev)


def cb(progress):
    print(".")


res, health = cal.run_on_chip_calibration("", cb, 15000)
