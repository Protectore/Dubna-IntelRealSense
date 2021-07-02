import cv2
import pyrealsense2 as rs2 
import numpy as np

def colorize_depth(depth: rs2.depth_frame) -> rs2.video_frame: 
    ''' Colorize depth data for vizualization '''

     # use colorizer to visualize depth data
    color_map = rs2.colorizer()
    # use black to white color map
    color_map.set_option(rs2.option.color_scheme, 0) # 0 - JET, 3 - black&white

    colorized = color_map.colorize(depth)
    return colorized
    #np.asanyarray(colorized.get_data())


def get_frame(pipe: rs2.pipeline):
    frames = pipe.wait_for_frames()
    if frames:
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
    return frames, depth, color


def to_image_representation(frames: rs2.frame = None, depth_frame: rs2.depth_frame = None,
 color_frame: rs2.video_frame = None):

    if not depth_frame:
        depth_frame = frames.get_depth_frame()
    if not color_frame:
        color_frame = frames.get_color_frame()

    depth_frame = colorize_depth(depth_frame)

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())

    return depth_img, color_img


if __name__ == '__main__':
    COLOR_WINDOW = 'image'
    DEPTH_WINDOW = 'depth'

    # declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs2.pipeline()

    # configure camera pipeline for depth + color streaming
    cfg = rs2.config()

    profile = pipe.start(cfg)

    while True:
        frames, depth_frame, color_frame = get_frame(pipe)
        depth_img, col_img = to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
        cv2.imshow(COLOR_WINDOW, col_img)
        cv2.imshow(DEPTH_WINDOW, depth_img)
        if cv2.waitKey(1) == 27:
            break

    pipe.stop()
    cv2.destroyAllWindows()





