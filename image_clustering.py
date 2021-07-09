from dark_net import get_objects_in_image, create_yolo_model
from clean_depth import set_device_options, DepthProcesser
import frames
import pyrealsense2 as rs2
from measure import Projector
import numpy as np
import cv2


WIDTH, HEIGHT = 1280, 720
FPS = 30
DEPTH_MIN, DEPTH_MAX = 0.4, 10  # В метрах

LABELS_FILE='./yolo_v4/coco.names'
CONFIG_FILE='./yolo_v4/yolov4.cfg'
WEIGHTS_FILE='./yolo_v4/yolov4.weights'

if __name__ == '__main__':
    net, ln, labels = create_yolo_model()
    
    pipe = rs2.pipeline()
    cfg = rs2.config()
    cfg.enable_stream(rs2.stream.depth, WIDTH, HEIGHT, rs2.format.z16, FPS)
    cfg.enable_stream(rs2.stream.color, WIDTH, HEIGHT, rs2.format.bgr8, FPS)
    profile = pipe.start(cfg)
    set_device_options(profile)

    device = profile.get_device()
    for _ in range(5):
        pipe.wait_for_frames()
    dp = DepthProcesser()

    projector = Projector(device, profile, DEPTH_MIN, DEPTH_MAX)
    while True:
        frame, depth_frame, color_frame = frames.get_frame(pipe)
        depth_frame = dp.process(frame)
        depth_image, color_image = frames.to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
        
        
        boxes, names = get_objects_in_image(color_image, net, ln, labels)
        depth_data = np.asanyarray(depth_frame.get_data())

        for box in boxes:
            (start_x, end_x, start_y, end_y) = box 
            depth_start = projector.color2depth((start_x, start_y)).astype(int)
            depth_end = projector.color2depth((end_x, end_y)).astype(int)
            object_depth = depth_data[depth_start[1]:depth_end[1], depth_start[0]:depth_end[0]]
            cv2.imshow('depth_image', depth_image[depth_start[1]:depth_end[1], depth_start[0]:depth_end[0]])



