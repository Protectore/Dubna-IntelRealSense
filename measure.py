from typing import List
import cv2
import pyrealsense2 as rs2
import numpy as np

from frames import *
from clean_depth import *


SELECTION_COLOR = (0, 255, 0)
LINE_THICKNESS = 1

DEPTH_MIN, DEPTH_MAX = 0.4, 10  # В метрах
WIDTH, HEIGHT = 1280, 720
FPS = 30


class LineDrawer:

    def __init__(self, thickness = 1, color = (255, 0, 0)):
        self.image = None
        self.cache = None
        self.start_pt = (0, 0)
        self.end_pt = (0, 0)
        self.color = color
        self.thickness = thickness
        self.drawing = False
    
    
    def fed_image(self, image):
        self.image = image
        self.cache = image.copy()


    def begin(self, x, y):
        if self.image is None:
            print('no image')
            return -1
        if self.cache is None:
            self.cache = self.image.copy()
        else:
            self.image = self.cache.copy()
        self.drawing = True
        self.start_pt = (x, y)
        self.end_pt = self.start_pt
        
        

    def move(self, x, y):
        if self.drawing:
            self.image = self.cache.copy()
            self.end_pt = (x, y)
            cv2.line(self.image, self.start_pt, self.end_pt, color=self.color, thickness=self.thickness)


    def finish(self, x, y):
        if self.drawing:
            self.image = self.cache.copy()
            self.end_pt = (x, y)
            cv2.line(self.image, self.start_pt, self.end_pt, color=self.color, thickness=self.thickness)
            self.drawing = False




class Projector:

    def __init__(self, device, profile, depth_min, depth_max):
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        depth_stream = profile.get_stream(rs2.stream.depth)
        color_stream = profile.get_stream(rs2.stream.color)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()  # Эта штука нужна для первоначального кода
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_to_color_extrinsics = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)
        self.color_to_depth_extrinsics = color_stream.as_video_stream_profile().get_extrinsics_to(depth_stream)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.color_frame = None
        self.depth_frame = None

    def set_values(self, depth_frame, color_frame):
        depth_stream = depth_frame.get_profile().as_video_stream_profile()
        color_stream = color_frame.get_profile().as_video_stream_profile()

        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()  # Эта штука нужна для первоначального кода
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_to_color_extrinsics = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)
        self.color_to_depth_extrinsics = color_stream.as_video_stream_profile().get_extrinsics_to(depth_stream)

        self.depth_frame = depth_frame
        self.color_frame = color_frame



    def check_frames(self):
        if self.color_frame is None:
            print('No color_frame')
            return False
        if self.depth_frame is None:
            print('No depth frame')
            return False
        return True


    def color2depth(self, color_pixel):
        if not self.check_frames():
            return None
        depth_data = self.depth_frame.get_data()
        depth_pixel = rs2.rs2_project_color_pixel_to_depth_pixel(
            depth_data, self.depth_scale,
            self.depth_min, self.depth_max,
            self.depth_intrinsics, self.color_intrinsics, self.depth_to_color_extrinsics,
            self.color_to_depth_extrinsics, color_pixel)
        return np.array(depth_pixel)

    
    def pixel2point(self, pixel):
        if not self.check_frames():
            return None
        #print(np.asanyarray(self.depth_frame.get_data()).shape)
        depth = np.asanyarray(self.depth_frame.get_data())[pixel[1], pixel[0]]
        return np.array(rs2.rs2_deproject_pixel_to_point(self.depth_intrinsics, pixel[::-1], depth))




class LineMeasurer:

    def __init__(self, ndim):
        self.ndim = ndim
        self.start = np.zeros(ndim)
        self.stop = np.zeros(ndim)


    def calculate(self):
        diff = self.stop - self.start
        return np.linalg.norm(diff)




def mouse_events_handler(event, x, y, flags, params):
    (projector, color_drawer, depth_drawer, measurer) = params

    if event == cv2.EVENT_LBUTTONDOWN:
        color_start = (x, y)
        color_drawer.begin(x, y)

        (depth_x, depth_y) = projector.color2depth(color_start).astype(int)
        depth_drawer.begin(depth_x, depth_y)


        #start_coords = projector.pixel2point((depth_y, depth_x))
        start_coords = projector.pixel2point((depth_x, depth_y))
        measurer.start = start_coords
        print(f'start: x = {x}, y = {y}, coords = {start_coords}')

    elif event == cv2.EVENT_MOUSEMOVE:
       if color_drawer.drawing:
           color_stop = (x, y)
           color_drawer.move(x, y)
           (depth_x, depth_y) = projector.color2depth(color_stop).astype(int)
           depth_drawer.move(depth_x, depth_y)

    elif event == cv2.EVENT_LBUTTONUP:
        color_stop = (x, y)
        color_drawer.finish(x, y)
        (depth_x, depth_y) = projector.color2depth(color_stop).astype(int)
        depth_drawer.finish(depth_x, depth_y)
        #stop_coords = projector.pixel2point((depth_y, depth_x))
        stop_coords = projector.pixel2point((depth_x, depth_y))
        measurer.stop = stop_coords
        print(f'stop: x = {x}, y = {y}, coords = {stop_coords}')

        # Если начальная/конечная точки попали туда, где не вычисленна глубина, вычисления невозможно выполнить
        if (np.argmax(measurer.start) == 0 or np.argmax(measurer.stop) == 0):
            print('Eror, start or end depth isn\'t defined')
        else:
            size = measurer.calculate()
            print(f'vector norm = {round(np.linalg.norm(size), 2)}mm')
        print()




def set_frame_params(col_frame, depth_frame, depth_img, col_img, color_drawer, depth_drawer, projector):
    color_drawer.fed_image(col_img)
    depth_drawer.fed_image(depth_img)
    projector.color_frame = col_frame
    projector.depth_frame = depth_frame




if __name__ == '__main__':

    COLOR_WINDOW = 'image'
    DEPTH_WINDOW = 'depth'
    NDIM = 3

    WIDTH, HEIGHT = 1280, 720
    FPS = 30

    # declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs2.pipeline()

    # configure camera pipeline for depth + color streaming
    cfg = rs2.config()
    cfg.enable_stream(rs2.stream.depth, WIDTH, HEIGHT, rs2.format.z16, FPS)
 #enable default depth
    # set color stream format to RGBA 
    # to allow blending of the color frames on top of the depth frames
    cfg.enable_stream(rs2.stream.color, WIDTH, HEIGHT, rs2.format.bgra8, FPS)
    #cfg.enable_stream(rs2.stream.color, rs2.format.rgba8)

    #pipeline_wrapper = rs2.pipeline_wrapper(pipe)
    #pipeline_profile = cfg.resolve(pipeline_wrapper)
    #device = pipeline_profile.get_device()

    # start pipeline 
    profile = pipe.start(cfg)
    device = profile.get_device()
    set_device_options(profile)

    projector = Projector(device, profile, DEPTH_MIN, DEPTH_MAX)
    color_drawer = LineDrawer()
    depth_drawer = LineDrawer()
    measurer = LineMeasurer(NDIM)
    depth_processer = DepthProcesser()

    cv2.namedWindow(DEPTH_WINDOW)
    cv2.namedWindow(COLOR_WINDOW)

    for i in range(5):
        frames, depth_frame, color_frame = get_frame(pipe)
    depth_img, col_img = to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
    set_frame_params(color_frame, depth_frame, depth_img, col_img, color_drawer, depth_drawer, projector)
    
    params = [projector, color_drawer, depth_drawer, measurer]
    cv2.setMouseCallback(COLOR_WINDOW, mouse_events_handler, params)

    while True:
        key = cv2.waitKey(1) 

        cv2.imshow(COLOR_WINDOW, color_drawer.image)
        cv2.imshow(DEPTH_WINDOW, depth_drawer.image)
              
        if key == ord('p') & 0xFF :
            print('taking photo...')
            frames, depth_frame, color_frame = get_frame(pipe)
            depth_frame = depth_processer.process(frames)
            depth_img, col_img = to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
            set_frame_params(color_frame, depth_frame, depth_img, col_img, color_drawer, depth_drawer, projector)

            params[0] = projector
            params[1] = color_drawer
            params[2] = depth_drawer

        elif key == ord('q') & 0xFF or key == 27:
            break

    pipe.stop()
    cv2.destroyAllWindows()



