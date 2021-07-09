import cv2
import numpy as np
import time
from video import get_video_stream_from_camera
import pyrealsense2 as rs2
import frames
from measure import LineMeasurer, Projector
import numpy as np
from clean_depth import DepthProcesser, set_device_options
from dark_net import *

if __name__ == '__main__':
    net, ln, labels = create_yolo_model()
    
    pipe = rs2.pipeline()
    cfg = rs2.config()
    cfg.enable_stream(rs2.stream.depth, WIDTH, HEIGHT, rs2.format.z16, FPS)
    cfg.enable_stream(rs2.stream.color, WIDTH, HEIGHT, rs2.format.bgr8, FPS)
    profile = pipe.start(cfg)
    device = profile.get_device()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    set_device_options(profile)
    for _ in range(5):
        pipe.wait_for_frames()

    projector = Projector(device, profile, DEPTH_MIN, DEPTH_MAX)
    processer = DepthProcesser()
    model = KMeans(n_clusters=2)

    while True:
        frame, depth_frame, color_frame = frames.get_frame(pipe)
        depth_frame = processer.process(frame, False)
        depth_image, color_image = frames.to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
        
        #print(color_image.shape)
        boxes, names = get_objects_in_image(color_image, net, ln, labels)
        
        #COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

        #prediction = get_predict(color_image, LABELS, net, ln, COLORS)
        depth_data = np.asanyarray(depth_frame.get_data())


        projector.set_values(depth_frame, color_frame)

        
        cv2.imshow('depth', depth_image)

        key = cv2.waitKey(1)
        measurer = LineMeasurer(3)

        for name, box in zip(names, boxes):

            
            if name != 'tvmonitor':
                continue
            (start_x, end_x, start_y, end_y) = box 
            (w, h) = (end_x - start_x, end_y - start_y)
            middle_point = (start_x + w // 2, start_y + h // 2)
            depth_middle = projector.color2depth(middle_point).astype(int)
            #print(start_x, start_y, end_x, end_y)
            depth_start = projector.color2depth((start_x, start_y)).astype(int)
            depth_end = projector.color2depth((end_x, end_y)).astype(int)
            #print(depth_start, depth_end)
            
            #try:
            #object_depth = depth_data[depth_start[1]:depth_end[1], depth_start[0]:depth_end[0]]
            cv2.imshow(name, depth_image[depth_start[1]:depth_end[1], depth_start[0]:depth_end[0]])
            cv2.rectangle(color_image, (start_x,start_y), (end_x, end_y), (0, 255, 0), 2)
            object_depth = depth_data[depth_middle[1], depth_middle[0]]   
             
            start_point = projector.pixel2point((start_x + w // 2, start_y))
            start_point[2] = object_depth *  depth_scale
            end_point = projector.pixel2point((start_x + w // 2, end_y))
            end_point[2] = object_depth *  depth_scale
            
            measurer.start = start_point
            measurer.stop = end_point
            print(f'r = {object_depth * depth_scale}')
            print(f"h = {round(np.linalg.norm(measurer.calculate()), 2)}mm")
            cv2.putText(color_image, f'r = {round(object_depth * depth_scale, 2)}', (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

          #  except Exception as e:
            #    print(e)
        cv2.imshow('color', color_image)
